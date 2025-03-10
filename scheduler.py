"""
File name: scheduler.py
Author: Stefan Zeidler
Created on: 3/7/2025
Description: Scheduling program using constraint search to create course plans for students.
Adapted from https://github.com/aimacode/aima-python/blob/master/csp.py.
"""
from typing import override
import scheduler_utils
import csp
import copy
import pandas as pd
import os
from collections import defaultdict


class Scheduler:
    COURSE = 'Course'
    ID = 'ID'
    PROGRAM = 'program'
    PREREQUISITES = "Prerequisites"
    TRANSFERS = 'transfers'
    TAKEN = 'taken'
    MAX_TERMS = 'maxterms'

    def __init__(self):
        # super().__init__(None,None,None,None)
        self.initial_season = None
        self.course_list = None
        self.program_requirements = None
        self.student_list = None
        self.current_student = 0

    def plan(self, initial_season: str, course_file: str, student_file: str):
        self.initial_season = initial_season
        self.course_list, self.program_requirements, self.student_list = scheduler_utils.create_data_tables(course_file,
                                                                                                       student_file,
                                                                                                       "program_requirements.csv")
        for index, row in self.student_list.iterrows():
            self.current_student = index
            self._create_csp()

    def _create_csp(self):
        term = 1
        max_terms = self.student_list.at[self.current_student,self.MAX_TERMS]
        constraints = defaultdict(list)
        taken_courses = self.student_list.at[self.current_student, self.TRANSFERS] + self.student_list.at[self.current_student, self.TAKEN]
        courses = []
        available_courses = self._generate_available_courses(taken_courses)
        while not available_courses.empty:
            courses = courses + available_courses
            for course in courses:
                for prerequisite in self.course_list.at[course,self.PREREQUISITES]:
                    ...

            available_courses = self._generate_available_courses(courses)

    def prerequisite(self, course_a: int, term_a: int, course_b: int, term_b: int):
        return term_a >= term_b

    def _get_season(self, term):
        if term%2 == 1:
            return self.initial_season
        else:
            return 'S' if self.initial_season == 'F' else 'F'

    def _season_constraint(self,course_a,term_a,course_b,term_b):
        ...

    def _generate_available_courses(self, taken_courses: list[int]) -> pd.DataFrame:
        """
        Generates list of courses that student may enroll in for the given term.
        :param taken_courses: List of course IDs for all courses already taken by student.
        :return:Filtered copy of entire course_list dataframe containing only available courses.
        """
        available_courses = self._recursive_deep_copy(self.course_list)
        available_courses = available_courses[~available_courses.index.isin(taken_courses)]
        for course in taken_courses:
            available_courses[Scheduler.PREREQUISITES] = available_courses[Scheduler.PREREQUISITES].apply(
                lambda x: Scheduler._remove_prerequisite(x, course))
        available_courses = available_courses[available_courses[Scheduler.PREREQUISITES].apply(lambda x: len(x) == 0)]
        return available_courses

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

class SchedulerCSP(csp.CSP):
    def  __init__(self,variables, domains, neighbors, constraints):
        super().__init__(variables, domains, neighbors, constraints)

    @override
    def nconflicts(self,var,val, assignment):

        ...




def main():
    my_scheduler = Scheduler()
    my_scheduler.plan('F','courses.csv','students.csv' )

main()