"""
File name: scheduler.py
Author: Stefan Zeidler
Created on: 3/7/2025
Description: Scheduling program using constraint search to create course plans for students.
Adapted from https://github.com/aimacode/aima-python/blob/master/csp.py.
"""
from typing import override, Tuple, Callable
import scheduler_utils
import csp
import copy
import pandas as pd
import os

import itertools
import random
import re
import string
from collections import defaultdict, Counter
from functools import reduce
from operator import eq, neg

from sortedcontainers import SortedSet

import search
from utils import argmin_random_tie, count, first, extend


class Scheduler:
    COURSE = 'Course'
    ID = 'ID'
    PROGRAM = 'program'
    PREREQUISITES = "Prerequisites"
    TRANSFERS = 'transfers'
    TAKEN = 'taken'
    MAX_TERMS = 'maxterms'
    REQUIRED = 'required_courses'
    CUTOFF = 'electives_cutoff'
    PREFERENCES_TOPICS = "preferences-topics"
    PREFERENCES_INSTRUCTORS = "preferences-instructors"
    AREA = 'area'
    INSTRUCTORS = "Instructors"
    TERMS = 'Terms'
    CREDITS = 'Credits'
    MAX_CREDITS = "maxcredits"

    def __init__(self):
        # super().__init__(None,None,None,None)
        self.initial_season = None
        self.course_list = None
        self.program_requirements = None
        self.student_list = None
        self.current_student = 0
        self.electives = None
        self.required_courses = None


    class SchedulerCSP:
        def __init__(self, variables, domains, constraints):
            self.variables = variables or domains.keys()
            self.constraints = constraints

        def nconflicts(self, var, val, assignment):
            """Return the number of conflicts var=val has with other variables."""

            # Subclasses may implement this more efficiently
            def conflict(var2):
                return var2 in assignment and not self.constraints(var, val, var2, assignment)

            return count(conflict(v) for v in self.neighbors[var])

    class SchedulerConstraint:
        def __init__(self, scope, condition):
            self.scope = scope
            self.condition = condition

        def holds(self, course, term, assignment):
            kwargs = {'scope': self.scope, 'course': course, 'term': term, 'assignment': assignment}
            return self.condition(**kwargs)

    @staticmethod
    def prerequisite_constraint(*,scope: Tuple[int,int], course, term: int, assignment: dict[int, int]):
        prerequisite, dependent = scope
        if prerequisite == 594 and dependent == 595:
            return (term + 1 == assignment[dependent]) if course == prerequisite else (assignment[prerequisite] == term - 1)
        return (term < assignment[dependent]) if course == prerequisite else (term > assignment[prerequisite])

    def max_credits_constraint(self,**kwargs):
        course = kwargs['course']
        term = kwargs['term']
        assignment = kwargs['assignment']
        max_credits = self._get_student_max_credits()
        courses_in_term = [key for key in assignment if (assignment[key] == term and key != course)]
        credits_in_term = sum([self._get_course_credits(course_in_term) for course_in_term in courses_in_term])
        return self._get_course_credits(course) + credits_in_term <= max_credits

    def elective_constraint(self,*, course, term, assignment):
        electives = [elective for elective in self.electives if
                     (elective in assignment.keys() and assignment[elective] is not None)]
        max_electives = self.program_requirements.at[self._get_program(), 'elective_courses_required']
        if course in electives:
            # removing from electives is okay so long as...
            if term is None:
                 return len(electives)-1 >= max_electives
            # changing the course is okay if it's already in electives
            return True
        else:
            #if it's already not in electives, fine to keep it out? Unless there are too few, then we want to assign another,
            # so None would not be valid. Get it in there!
            if term is None:
                return len(electives) < max_electives
            return len(electives)+1 > max_electives

    def season_constraint(self, **kwargs):
        return self._get_season(kwargs['term']) in kwargs['scope']



    def plan(self, initial_season: str, course_file: str, student_file: str):
        self.initial_season = initial_season
        self.course_list, self.program_requirements, self.student_list = scheduler_utils.create_data_tables(course_file,
                                                                                                       student_file,
                                                                                                       "program_requirements.csv")
        for index, row in self.student_list.iterrows():
            self.current_student = index
            program = self.student_list.at[self.current_student,Scheduler.PROGRAM]
            if program == 'BS' or program == 'BS':
                all_courses = self.course_list.index
                self.required_courses = self.program_requirements.at[program, Scheduler.REQUIRED]
                # cutoff = self.program_requirements.at[program, Scheduler.CUTOFF]
                self.electives = list(set(all_courses) - set(self.required_courses))
                # self.electives = [course for course in self.electives if course >= cutoff]
            my_csp = self._create_csp()
            solution = csp.min_conflicts(my_csp)
            print(solution)

    def _create_csp(self):
        lower_bound = 1
        max_terms = self.student_list.at[self.current_student,self.MAX_TERMS]
        constraints = defaultdict(list)
        transfers_and_taken = self.student_list.at[self.current_student, self.TRANSFERS] + self.student_list.at[self.current_student, self.TAKEN]
        variables = []
        domains = defaultdict(list)
        neighbors = defaultdict(list)
        available_courses = list(self._generate_available_courses(transfers_and_taken).index)
        while len(available_courses) > 0:
            variables = variables + available_courses
            for course in available_courses:
                if course in self.electives:
                    neighbors[course].append(course)
                    domains[course] = [None]
                for prerequisite in set(self.course_list.at[course,self.PREREQUISITES]).difference(set(transfers_and_taken)):
                    neighbors[prerequisite].append(course)
                    neighbors[course].append(prerequisite)
                domains[course] += list(range(lower_bound,max_terms+1))
                # for i in range(lower_bound,max_terms+1):
                #     domains[course].append(i)
            lower_bound += 1
            if lower_bound > max_terms:
                return None
            available_courses = list(self._generate_available_courses(variables + transfers_and_taken).index)
        return SchedulerCSP(variables,domains,neighbors,constraints)



    # def constraint(self, course_a: int, term_a: int, course_b: int, assignment):
    #     if course_a == course_b:
    #         return self._elective_constraint(course_a,term_a,assignment)
    #     return self._prerequisite_constraint(course_a, term_a, course_b, assignment)

    # def _elective_constraint(self,course_a, term_a, assignment):
    #     electives = [elective for elective in self.electives if
    #                  (elective in assignment.keys() and assignment[elective] is not None)]
    #     max_electives = self.program_requirements.at[self._get_program(), 'elective_courses_required']
    #     if course_a in electives:
    #         # removing from electives is okay so long as...
    #         if term_a is None:
    #              return len(electives)-1 >= max_electives
    #         return True
    #     else:
    #         #if it's already not in electives, fine to keep it out? Unless there are too few, then we want to assign another,
    #         # so None would not be valid. Get it in there!
    #         if term_a is None:
    #             return len(electives)+1 <= max_electives
    #         return len(electives)+1 > max_electives


    # def _prerequisite_constraint(self,course_a, term_a, course_b, assignment):
    #     term_b = assignment[course_b]
    #     if term_b is None:
    #         return True
    #     if course_a == '594' and course_b == '595':
    #         return term_a + 1 == term_b
    #     return term_a < term_b
    #
    # def _dependency_constraint(self, course_a, term_a, course_b, assignment):
    #     #only electives can be None, and have no further dependents
    #     if term_a is None:
    #         return True
    #     # prereqs do not have None in domain
    #     term_b = assignment[course_b]
    #     if course_a == '595' and course_b == '594':
    #         return term_a - 1 == term_b
    #     return term_b < term_a

    # def _max_credits_constraint(self,course_a, term_a, scope, assignment):
    # def max_credits_constraint(self,course_a, term_b, course_b, assignment):
    #     current_courses = [key for key in assignment if assignment[key] == course_b]
    #     current_credits = sum([self._get_credits(course) for course in current_courses])
    #     return self._get_credits(course_a) + current_credits <= self.student_list.at[self.current_student,Scheduler.MAX_CREDITS]


    def _get_season(self, term):
        if term % 2 == 1:
            return self.initial_season
        else:
            return 'S' if self.initial_season == 'F' else 'F'

    def _get_prerequisites(self,course):
        return self.course_list.at[course, Scheduler.PREREQUISITES]

    def _get_program(self):
        return self.student_list.at[self.current_student, Scheduler.PROGRAM]

    def _get_course_credits(self, course) -> int:
        return self.course_list.at[course,Scheduler.CREDITS]

    def _generate_available_courses(self, taken_courses: list[int]) -> pd.DataFrame:
        """
        Generates list of courses that student may enroll in for the given term.
        :param taken_courses: List of course IDs for all courses already taken by student.
        :return:Filtered copy of entire course_list dataframe containing only available courses.
        """
        course_candidates = self._recursive_deep_copy(self.course_list)
        course_candidates = course_candidates[~course_candidates.index.isin(taken_courses)]
        for course in taken_courses:
            course_candidates[Scheduler.PREREQUISITES] = course_candidates[Scheduler.PREREQUISITES].apply(
                lambda x: Scheduler._remove_prerequisite(x, course))
        course_candidates = course_candidates[course_candidates[Scheduler.PREREQUISITES].apply(lambda x: len(x) == 0)]
        # available_courses = list(set(self.required_courses + self.electives).intersection(set(course_candidates.index)))
        return course_candidates

    @staticmethod
    def _recursive_deep_copy(df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        for col in df_copy.columns:
            df_copy[col] = df_copy[col].apply(copy.deepcopy)
        return df_copy

    @staticmethod
    def _remove_prerequisite(prerequisites: list, course_no: int):
        if course_no in prerequisites:
            prerequisites.remove(course_no)
        return prerequisites

    def _get_preference_value(self, course):
        preference_value = 0
        preferred_instructors = self.student_list.at[self.current_student, Scheduler.PREFERENCES_INSTRUCTORS]
        course_instructors = self.course_list.at[course, Scheduler.INSTRUCTORS]
        preference_value = preference_value + sum(
            [1 for instructor in course_instructors if instructor in preferred_instructors])
        preferred_topics = self.student_list.at[self.current_student, Scheduler.PREFERENCES_TOPICS]
        course_topic = self.course_list.at[course, Scheduler.AREA]
        if course_topic in preferred_topics:
            preference_value += 1
        return preference_value

    def _get_student_max_credits(self):
        return self.student_list.at[self.current_student, Scheduler.MAX_CREDITS]

class SchedulerCSP(csp.CSP):
    def  __init__(self,variables, domains, neighbors, constraints):
        super().__init__(variables, domains, neighbors, constraints)

    def nconflicts(self, var, val, assignment):
        """Return the number of conflicts var=val has with other variables."""

        # Subclasses may implement this more efficiently
        def conflict(var2):
            return var2 in assignment and not self.constraints(var, val, var2, assignment)

        return count(conflict(v) for v in self.neighbors[var])



    # def __call__(self, var, val, assignment):
    #     return self.condition(var, val, self.scope, assignment)


def main():
    my_scheduler = Scheduler()
    my_scheduler.plan('F','courses.csv','students.csv' )

# main()