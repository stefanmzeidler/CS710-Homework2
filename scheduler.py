"""
File name: scheduler.py
Author: Stefan Zeidler
Created on: 3/7/2025
Description: Scheduling program using constraint search to create course plans for students.
Adapted from https://github.com/aimacode/aima-python/blob/master/csp.py.
"""
import copy
from collections import defaultdict
from typing import override, Tuple

import pandas as pd

import scheduler_utils
from aime import csp, search
from aime.utils import *


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
        self.initial_season = None
        self.course_list = None
        self.program_requirements = None
        self.student_list = None
        self.current_student = 0
        self.electives = None
        self.required_courses = None

    class SchedulerCSP(csp.CSP):
        def __init__(self, variables, domains, constraints):
            super().__init__(variables, domains, None, constraints)

        @override
        def nconflicts(self, course, term, assignment):
            """Return the number of conflicts var=val has with other variables."""

            return count((not constraint.holds(course, term, assignment)) for constraint in self.constraints[course])

    class SchedulerConstraint:
        def __init__(self, scope, condition):
            self.scope = scope
            self.condition = condition

        def __repr__(self):
            return self.condition.__name__ +": " + str(self.scope)

        def holds(self, course, term, assignment):
            kwargs = {'scope': self.scope, 'course': course, 'term': term, 'assignment': assignment}
            return self.condition(**kwargs)


    def prerequisite_constraint(self,scope: Tuple[int, int], course, term: int, assignment: dict[int, int]):
        prerequisite, dependent = scope
        if term is None:
            return True
            # return len(self._get_assigned_electives(assignment)) >= self._get_max_electives()
        # if (course == prerequisite and dependent not in assignment.keys()) or (course == dependent and prerequisite not in assignment.keys()):
        #     return True
        if prerequisite not in assignment.keys() or dependent not in assignment.keys():
            return True
        if assignment[dependent] is None:
            return True
        if prerequisite == 594 and dependent == 595:
            return (term + 1 == assignment[dependent]) if course == prerequisite else (
                        assignment[prerequisite] == term - 1)
        return (term < assignment[dependent]) if course == prerequisite else (term > assignment[prerequisite])

    def max_credits_constraint(self, **kwargs):
        course = kwargs['course']
        term = kwargs['term']
        assignment = kwargs['assignment']
        if term is None:
            return True
        max_credits = self._get_student_max_credits()
        courses_in_term = [key for key in assignment if (assignment[key] == term and key != course)]
        credits_in_term = sum([self._get_course_credits(course_in_term) for course_in_term in courses_in_term])
        return self._get_course_credits(course) + credits_in_term <= max_credits

    def solution_to_string(self,solution):
        reverse_dict = defaultdict(list)
        for key, value in solution.items():
            if value is not None:
                reverse_dict[value].append(key)
        data = {'Terms': reverse_dict.keys(), 'Season': [self._get_season(term) for term in reverse_dict.keys()], 'Courses': reverse_dict.values()}
        df = pd.DataFrame(data).set_index('Terms')
        df.sort_index(inplace=True)
        return df.to_string()

    def elective_constraint(self, **kwargs):
        course = kwargs['course']
        term = kwargs['term']
        assignment = kwargs['assignment']
        assigned_electives = self._get_assigned_electives(assignment)
        if len(assigned_electives) > self._get_max_electives():
            return term is None
        if len(assigned_electives) < self._get_max_electives():
            return term is not None
        if course not in assignment.keys():
            return term is None
        return (assignment[course] is not None and term is not None) or (assignment[course] is None and term is None)
        #

    def weighted_elective_constraint(self, **kwargs):
        course = kwargs['course']
        term = kwargs['term']
        assignment = kwargs['assignment']
        assigned_electives = self._get_assigned_electives(assignment)
        if len(assigned_electives) > self._get_max_electives():
            return self._preference_weight(course,term,assigned_electives)
        if len(assigned_electives) < self._get_max_electives():
            return term is not None
        if course not in assignment.keys():
            return term is None
        if term is None:
            return assignment[course] is None
        return self._preference_weight(course, term, assigned_electives)
        # return (assignment[course] is not None and term is not None) or (
        #             assignment[course] is None and term is None)

    def _preference_weight(self, term, course, assigned_electives):
        for elective in assigned_electives:
            if self._get_preference_value(course) > self._get_preference_value(elective):
                return term is not None
        return term is None

        #
        # if course in self._get_assigned_electives(assignment):
        #     return term is not None
        # return term is None

        # if course in assignment.keys():
        #     return term is not None
        # return term == assignment[course]
        # max_electives = self._get_max_electives()
        # if term is None and course in assigned_electives:
        #     assigned_electives.remove(course)
        #     return len(assigned_electives) > max_electives
        # elif term is not None and course not in assigned_electives:
        #     return len(assigned_electives) < max_electives


        # return len(assigned_electives) == max_electives

    def season_constraint(self, **kwargs):
        if(kwargs['term']) is None:
            return True
        return self._get_season(kwargs['term']) in kwargs['scope']

    def min_conflicts(self, csprob, current, max_steps=100000):
        """Solve a CSP by stochastic Hill Climbing on the number of conflicts."""
        # Generate a complete assignment for all variables (probably with conflicts)
        # csprob.current = current = {}
        csprob.current = current
        for var in csprob.variables:
            val = self.min_conflicts_value(csprob, var, current)
            csprob.assign(var, val, current)
        # Now repeatedly choose a random conflicted variable and change it
        for i in range(max_steps):
            no_assigned_electives = len(self._get_assigned_electives(current))
            conflicted = csprob.conflicted_vars(current)
            if not conflicted:
                return current
            var = random.choice(conflicted)
            val = self.min_conflicts_value(csprob, var, current)
            csprob.assign(var, val, current)
        return None

    def min_conflicts_value(self,csp, var, current):
        """Return the value that will give var the least number of conflicts.
        If there is a tie, choose at random."""
        return min(csp.domains[var], key=lambda val: csp.nconflicts(var, val, current))
        # if var in self.required_courses:
        #     return min(csp.domains[var], key=lambda val: csp.nconflicts(var, val, current))
        # return argmax_random_tie(csp.domains[var], key=lambda val: csp.nconflicts(var, val, current))


    def plan(self, initial_season: str, course_file: str, student_file: str):
        self.initial_season = initial_season
        self.course_list, self.program_requirements, self.student_list = scheduler_utils.create_data_tables(course_file,
                                                                                                            student_file,
                                                                                                            "program_requirements.csv")
        for index, row in self.student_list.iterrows():
            self.current_student = index
            program = self.student_list.at[self.current_student, Scheduler.PROGRAM]
            if program == 'BS' or program == 'BS':
                all_courses = self.course_list.index
                self.required_courses = self._get_required_courses()
                # cutoff = self.program_requirements.at[program, Scheduler.CUTOFF]
                self.electives = list(set(all_courses) - set(
                    self.required_courses))  # self.electives = [course for course in self.electives if course >= cutoff]
            my_required_csp, my_elective_csp = self._create_csps()
            if my_required_csp is None:
                print("No solution")
            else:
                required_courses_solution = self.min_conflicts(my_required_csp, {})
                elective_courses_solution = self.min_conflicts(my_elective_csp,required_courses_solution)
                print(self.solution_to_string(elective_courses_solution))

    def _create_csps(self):
        lower_bound = 0
        max_terms = self._get_max_terms()
        constraints = defaultdict(list)
        transfers_and_taken = self._get_transfers() + self._get_taken()
        test = self._topological_sort()
        variables = []
        domains = defaultdict(list)
        available_courses = list(self._generate_available_courses(transfers_and_taken).index)
        while len(available_courses) > 0:
            lower_bound += 1
            if lower_bound > max_terms:
                return None, None
            variables = variables + available_courses
            for course in available_courses:
                if course in self.electives:
                    domains[course] = [None]
                    constraints[course].append(self.SchedulerConstraint(None, self.elective_constraint))
                for prerequisite in set(self._get_prerequisites(course)).difference(
                        set(transfers_and_taken)):
                    temp_constraint = self.SchedulerConstraint((prerequisite, course), self.prerequisite_constraint)
                    constraints[course].append(temp_constraint)
                    constraints[prerequisite].append(temp_constraint)
                if len((seasons := self.course_list.at[course, Scheduler.TERMS])) < 2:
                    constraints[course].append(self.SchedulerConstraint(tuple(seasons), self.season_constraint))
                constraints[course].append(self.SchedulerConstraint(None, self.max_credits_constraint))
                domains[course] += list(range(lower_bound,
                                              max_terms + 1))
            available_courses = list(self._generate_available_courses(variables + transfers_and_taken).index)

        # required_variables = list(set(self._get_required_courses()).intersection(set(variables)))
        required_variables = [course_id for course_id in variables if course_id in self.required_courses]
        elective_variables =  list(set(variables) - set(required_variables))
        return self.SchedulerCSP(variables, domains, constraints),self.SchedulerCSP(elective_variables, domains, constraints)
        # return self.SchedulerCSP(required_variables, domains, constraints),self.SchedulerCSP(elective_variables, domains, constraints)

    def _topological_sort(self):
        variables = []
        available_courses = list(self._generate_available_courses(variables).index)
        while len(available_courses) > 0:
            variables = variables + available_courses
            available_courses = list(self._generate_available_courses(variables).index)
        transfers_and_taken = self._get_transfers() + self._get_taken()
        return [variable for variable in variables if variable not in transfers_and_taken]
        # def _get_constraints_and_domains(self, course, constraints, domains, lower_bound, max_terms, transfers_and_taken):
    #     if course in self.electives:
    #         domains[course] = [None]
    #         constraints[course].append(self.SchedulerConstraint(None, self.elective_constraint))
    #     for prerequisite in set(self._get_prerequisites(course)).difference(
    #             set(transfers_and_taken)):
    #         temp_constraint = self.SchedulerConstraint((prerequisite, course), self.prerequisite_constraint)
    #         constraints[course].append(temp_constraint)
    #         constraints[prerequisite].append(temp_constraint)
    #     if len((seasons := self.course_list.at[course, Scheduler.TERMS])) < 2:
    #         constraints[course].append(self.SchedulerConstraint(tuple(seasons), self.season_constraint))
    #     constraints[course].append(self.SchedulerConstraint(None, self.max_credits_constraint))
    #     domains[course] += list(range(lower_bound,
    #                                   max_terms + 1))

    def _get_season(self, term):
        if term % 2 == 1:
            return self.initial_season
        else:
            return 'S' if self.initial_season == 'F' else 'F'

    def _get_prerequisites(self, course):
        return self.course_list.at[course, Scheduler.PREREQUISITES]

    def _get_program(self):
        return self.student_list.at[self.current_student, Scheduler.PROGRAM]

    def _get_course_credits(self, course) -> int:
        return self.course_list.at[course, Scheduler.CREDITS]

    def _get_taken(self):
        return self.student_list.at[self.current_student, self.TAKEN]

    def _get_transfers(self):
        return self.student_list.at[self.current_student, self.TRANSFERS]

    def _get_required_courses(self):
        return self.program_requirements.at[self._get_program(), Scheduler.REQUIRED]

    def _get_max_terms(self):
        return self.student_list.at[self.current_student, self.MAX_TERMS]

    def _get_max_electives(self):
        return self.program_requirements.at[self._get_program(), 'elective_courses_required']
    def _get_assigned_electives(self,assignment):
        return [elective for elective in self.electives if
         (elective in assignment.keys() and assignment[elective] is not None)]

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

    # def __call__(self, var, val, assignment):  #     return self.condition(var, val, self.scope, assignment)




def main():
    my_scheduler = Scheduler()
    my_scheduler.plan('F', 'courses.csv', 'students.csv')

main()
# seq = [5,None,3,2,4]
# seq.sort()
# print(seq)