"""
File name: scheduler.py
Author: Stefan Zeidler
Created on: 3/7/2025
Description: Scheduling program using constraint search to create course plans for students.
Adapted from https://github.com/aimacode/aima-python/blob/master/csp.py.
"""

from collections import defaultdict
from typing import override, Tuple

import pandas as pd
from database import DataBase


from aime import csp
from aime.utils import *


class Scheduler:
    def __init__(self,initial_season, course_file, student_file, requirements_file):
        self.db = DataBase(course_file, student_file, requirements_file)
        self.initial_season = initial_season
        self.current_student = 0
        self.electives = None
        self.required_courses = None

    class SchedulerCSP(csp.CSP):
        def __init__(self, variables, domains, constraints):
            super().__init__(variables, domains, None, constraints)
            self.current = None

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

    @staticmethod
    def prerequisite_constraint(scope: Tuple[int, int], course, term: int, assignment: dict[int, int]):
        prerequisite, dependent = scope
        if term is None:
            return True
            # return len(self._get_assigned_electives(assignment)) >= self.db.get_max_electives(self.current_student)
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
        max_credits = self.db.get_student_max_credits(self.current_student)
        courses_in_term = [key for key in assignment if (assignment[key] == term and key != course)]
        credits_in_term = sum([self.db.get_course_credits(course_in_term) for course_in_term in courses_in_term])
        return self.db.get_course_credits(course) + credits_in_term <= max_credits

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
        if len(assigned_electives) > self.db.get_max_electives(self.current_student):
            return term is None
        if len(assigned_electives) < self.db.get_max_electives(self.current_student):
            return term is not None
        if course not in assignment.keys():
            return term is None
        return (assignment[course] is not None and term is not None) or (assignment[course] is None and term is None)


    def weighted_elective_constraint(self, **kwargs):
        course = kwargs['course']
        term = kwargs['term']
        assignment = kwargs['assignment']
        assigned_electives = self._get_assigned_electives(assignment)
        if len(assigned_electives) > self.db.get_max_electives(self.current_student):
            return self._preference_weight(course,term,assigned_electives)
        if len(assigned_electives) < self.db.get_max_electives(self.current_student):
            return not self._preference_weight(course,term,assigned_electives)
        if course not in assignment.keys():
            return not self._preference_weight(course,term,assigned_electives)
        if term is None:
            return assignment[course] is None
        return self._preference_weight(course, term, assigned_electives)


    def _preference_weight(self, course, term, assigned_electives):
        if course in assigned_electives:
            unassigned_electives = list(set(self.electives) - set(assigned_electives))
            for elective in unassigned_electives:
                if self.db.get_preference_value(self.current_student,course) > self.db.get_preference_value(self.current_student,elective):
                    return term is None
            return term is not None
        for elective in assigned_electives:
            if self.db.get_preference_value(self.current_student,course) > self.db.get_preference_value(self.current_student,elective):
                return term is not None
        return term is None

    def season_constraint(self, **kwargs):
        if(kwargs['term']) is None:
            return True
        return self._get_season(kwargs['term']) in kwargs['scope']

    def min_conflicts(self, csprob, current, max_steps=100000):
        csprob.current = current
        for var in csprob.variables:
            val = self.min_conflicts_value(csprob, var, current)
            csprob.assign(var, val, current)
        for i in range(max_steps):
            # no_assigned_electives = len(self._get_assigned_electives(current))
            conflicted = csprob.conflicted_vars(current)
            if not conflicted:
                return current
            var = random.choice(conflicted)
            val = self.min_conflicts_value(csprob, var, current)
            csprob.assign(var, val, current)
        return None
    @staticmethod
    def min_conflicts_value(csp, var, current):
        """Return the value that will give var the least number of conflicts.
        If there is a tie, choose at random."""
        return min(csp.domains[var], key=lambda val: csp.nconflicts(var, val, current))

    def plan(self):
        for index, row in self.db.student_rows():
            self.current_student = index
            program = self.db.get_program(self.current_student)
            if program == 'BS' or program == 'BA':
                all_courses = self.db.get_all_courses()
                self.required_courses = self.db.get_required_courses(self.current_student)
                self.electives = list(set(all_courses) - set(
                    self.required_courses))
            my_csp = self._create_csp()
            if my_csp is None:
                print("No solution")
            else:
                solution = self.min_conflicts(my_csp, {})
                print(self.solution_to_string(solution))
            # my_weighted_csp = self._create_csp(weighted=True)
            # if my_weighted_csp is None:
            #     print("No solution")
            # else:
            #     solution = self.min_conflicts(my_weighted_csp, {})
            #     print(self.solution_to_string(solution))

    def _create_csp(self, weighted = False):
        courses, domains = self._topological_sort()
        constraints = self._get_constraints(courses,weighted)
        for key, value in domains.items():
            if len(value) == 0:
                return None
        return self.SchedulerCSP(courses, domains, constraints)

    def _get_constraints(self, courses,weighted):
        constraints = defaultdict(list)
        transfers_and_taken = self.db.get_transfers(self.current_student) + self.db.get_taken(self.current_student)
        for course in courses:
            if course in self.electives:
                if weighted:
                    constraints[course].append(self.SchedulerConstraint(None, self.weighted_elective_constraint))
                else:
                    constraints[course].append(self.SchedulerConstraint(None, self.elective_constraint))
            for prerequisite in set(self.db.get_prerequisites(course)).difference(
                    set(transfers_and_taken)):
                temp_constraint = self.SchedulerConstraint((prerequisite, course), self.prerequisite_constraint)
                constraints[course].append(temp_constraint)
                constraints[prerequisite].append(temp_constraint)
            if len((seasons := self.db.get_seasons(course))) < 2:
                constraints[course].append(self.SchedulerConstraint(tuple(seasons), self.season_constraint))
            constraints[course].append(self.SchedulerConstraint(None, self.max_credits_constraint))
        return constraints

    def _topological_sort(self):
        lower_bound = 0
        domains = defaultdict(list)
        variables = []
        transfers_and_taken = self.db.get_transfers(self.current_student) + self.db.get_taken(self.current_student)
        available_courses = list(self.db.generate_available_courses(transfers_and_taken).index)
        while len(available_courses) > 0:
            lower_bound += 1
            variables = variables + available_courses
            for course in available_courses:
                if course in self.electives:
                    domains[course] = [None]
                domains[course] += list(range(lower_bound,
                                              self.db.get_max_terms(self.current_student) + 1))
            available_courses = list(self.db.generate_available_courses(variables + transfers_and_taken).index)
        return variables, domains

    def _get_season(self, term):
        if term % 2 == 1:
            return self.initial_season
        else:
            return 'S' if self.initial_season == 'F' else 'F'

    def _get_assigned_electives(self,assignment):
        return [elective for elective in self.electives if
         (elective in assignment.keys() and assignment[elective] is not None)]




def main():
    my_scheduler = Scheduler('F', 'courses.csv', 'students.csv','program_requirements.csv')
    my_scheduler.plan()

main()
# seq = [5,None,3,2,4]
# seq.sort()
# print(seq)