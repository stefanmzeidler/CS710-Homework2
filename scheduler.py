"""
File name: scheduler.py
Author: Stefan Zeidler
Created on: 3/7/2025
Description: Scheduling program using constraint search to create course plans for students.
Adapted from https://github.com/aimacode/aima-python/blob/master/csp.py.
"""

from collections import defaultdict
from typing import override, Tuple, Optional

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
        def __init__(self, variables, domains, binary_neighbors, constraints, current):
            super().__init__(variables, domains, binary_neighbors, constraints)
            self.current = current

        @override
        def nconflicts(self, course, term, assignment):
            """Return the number of conflicts var=val has with other variables."""

            return count((not constraint.holds(course, term, assignment)) for constraint in self.constraints[course])


        def check_arc_consistency(self,required_courses):
            self.support_pruning()
            for course, neighbors in self.neighbors.items():
                if course in required_courses:
                    for neighbor in neighbors:
                        if neighbor in required_courses:
                            if len(self._make_arc_consistent(course, neighbor)) == 0:
                                return False
            return True

        def _make_arc_consistent(self, course, neighbor):
            for val1 in self.domains[course]:
                keep = False
                if val1 is not None:
                    for val2 in self.domains[neighbor]:
                        for constraint in self.constraints[course]:
                            if constraint.scope is not None and course in constraint.scope and neighbor in constraint.scope:
                                if constraint.holds(course = course, term = val1, assignment={neighbor: val2}):
                                    keep = True
                                    break
                if not keep:
                    self.prune(course, val1, None)
            return self.curr_domains[course]

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
        if prerequisite not in assignment.keys():
            if course == prerequisite:
                return True
            else:
                return term is None
        if dependent not in assignment.keys():
            return True
        if assignment[prerequisite] is None or assignment[dependent] is None:
            return True
        if prerequisite == 594 and dependent == 595:
            return (term + 1 == assignment[dependent]) if course == prerequisite else (
                        assignment[prerequisite] == term - 1)
        return (term < assignment[dependent]) if course == prerequisite else (term > assignment[prerequisite])


    def min_terms_constraint(self, **kwargs):
        course = kwargs['course']
        term = kwargs['term']
        assignment = kwargs['assignment']
        if term is None or term == 1:
            return True
        previous_term_courses = [previous_course for previous_course in assignment.keys() if (assignment[previous_course] is not None and assignment[previous_course] == term -1)]
        prerequisites = self.db.get_prerequisites(course)
        previous_prerequisites = set(previous_term_courses).intersection(set(prerequisites))
        previous_term_credits = self.get_term_credits(course, term-1, assignment)
        if len(previous_prerequisites) == 0:
            return previous_term_credits + self.db.get_course_credits(course) > self.db.get_student_max_credits(self.current_student)
        return True


    def max_credits_constraint(self, **kwargs):
        course = kwargs['course']
        term = kwargs['term']
        assignment = kwargs['assignment']
        if term is None:
            return True
        max_credits = self.db.get_student_max_credits(self.current_student)
        credits_in_term = self.get_term_credits(course,term,assignment)

        return self.db.get_course_credits(course) + credits_in_term <= max_credits

    def total_credits_constraint(self, **kwargs):
        term = kwargs['term']
        assignment = kwargs['assignment']
        total_credits = sum([self.db.get_course_credits(course) for course in assignment.keys() if assignment[course] is not None])
        if total_credits < 18:
            return term is not None
        if total_credits > 21:
            return term is None
        return True

    def cut_off_credits_constraint(self, **kwargs):
        term = kwargs['term']
        assignment = kwargs['assignment']
        cutoff = self.db.get_cutoff(self.current_student)
        exclusions = self.db.get_course_exclusions(self.current_student)
        cutoff_courses = [key for key in assignment.keys() if (key >= cutoff) and (key not in exclusions)]
        cutoff_credits = sum([self.db.get_course_credits(cutoff_course) for cutoff_course in cutoff_courses])
        if cutoff_credits < 9:
            return term is not None
        return True

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

    def courseplan_local(self,state_creation_mode='topological_min_conflicts') -> Tuple[int, str, Optional[int],Optional[pd.DataFrame]]:
        my_csp = self._create_csp()
        if my_csp is not None:
            if not my_csp.check_arc_consistency(self.required_courses):
                return self.current_student, self.db.get_program(self.current_student), None, None
            solution = self.min_conflicts(my_csp,state_creation_mode=state_creation_mode)
            solution = self._solution_to_dframe(solution)
            return self.current_student, self.db.get_program(self.current_student), my_csp.nassigns, solution
        else:
            return self.current_student, self.db.get_program(self.current_student), None, None


    def min_conflicts(self, csprob, state_creation_mode ='topological_min_conflicts', max_steps=100000):
        current = csprob.current
        # for var in csprob.variables:
        #     val = self.min_conflicts_value(csprob, var, current)
        #     csprob.assign(var, val, current)
        self._create_initial_state(csprob, current, state_creation_mode)
        for i in range(max_steps):
            conflicted = csprob.conflicted_vars(current)
            if not conflicted:
                return current
            var = random.choice(conflicted)
            val = self.min_conflicts_value(csprob, var, current)
            csprob.assign(var, val, current)
        return None


    def _create_initial_state(self, csprob, current, state_creation_mode):
        match state_creation_mode:
            case 'topological_min_conflicts':
                for var in csprob.variables:
                    val = self.min_conflicts_value(csprob, var, current)
                    csprob.assign(var, val, current)
            case 'random':
                for var in shuffled(csprob.variables):
                    val = random.choice(csprob.domains[var])
                    csprob.assign(var, val, current)
            case 'first_fit':
                for var in sorted(csprob.variables):
                    for val in csprob.domains[var]:
                        if val is None:
                            continue
                        if self.get_term_credits(var, val, current) + self.db.get_course_credits(var) < self.db.get_student_max_credits(self.current_student):
                            csprob.assign(var, val, current)

            case _:
                raise ValueError('Invalid mode')


    @staticmethod
    def min_conflicts_value(csp, var, current):
        """Return the value that will give var the least number of conflicts.
        If there is a tie, choose at random."""
        return min(csp.domains[var], key=lambda val: csp.nconflicts(var, val, current))

    def plan(self, state_creation_mode ='topological_min_conflicts', print_to_screen = True):
        solutions = []
        for index, row in self.db.student_rows():
            self.current_student = index
            program = self.db.get_program(self.current_student)
            all_courses = self.db.get_all_courses()
            if program == 'BS' or program == 'BA':
                self.required_courses = self.db.get_required_courses(self.current_student)
                self.electives = list(set(all_courses) - set(
                    self.required_courses))
            else:
                self.required_courses = []
                self.electives = all_courses
            solution_data = self.courseplan_local(state_creation_mode = state_creation_mode)
            if print_to_screen:
                self._print_solution(solution_data)
            solutions.append(solution_data)
        return solutions


    @staticmethod
    def _print_solution(solution_data):
        if solution_data[2] is not None:
            print(f'Student id: {solution_data[0]}\nStudent Program:  {solution_data[1]}\nAssignments used: {solution_data[2]}\nCourse selection:\n {solution_data[3].to_string()}\n')
        else:
            print(f'Student id: {solution_data[0]}\nStudent Program:  {solution_data[1]}\nAssignments used: 0\nCourse selection:\nNo Solution\n')


    def _solution_to_dframe(self, solution) -> pd.DataFrame:
        reverse_dict = defaultdict(list)
        for key, value in solution.items():
            if value is not None:
                reverse_dict[value].append(key)
        data = {'Terms': reverse_dict.keys(), 'Season': [self._get_season(term) for term in reverse_dict.keys()], 'Courses': reverse_dict.values()}
        df = pd.DataFrame(data).set_index('Terms')
        df.sort_index(inplace=True)
        return df

    def _create_csp(self, weighted = False):
        self._check_transfer_credits()
        if self.db.get_student_max_credits(self.current_student) < 4:
            return None
        courses, domains = self._topological_sort()
        for key, value in domains.items():
            if key in self.required_courses and len(value) == 0:
                return None
        if not self._credit_consistency_check():
            return None
        if self.db.get_program(self.current_student).lower() == 'cs-minor':

            constraints,binary_neighbors = self._get_minor_constraints(courses, weighted)
        else:
            constraints,binary_neighbors = self._get_major_constraints(courses, weighted)
        current = self._previously_taken_assignment()
        return self.SchedulerCSP(courses, domains, binary_neighbors,constraints, current)

    def _credit_consistency_check(self):
        transfer_courses = self.db.get_transfers(self.current_student)
        taken_courses = self.db.get_taken(self.current_student)
        max_terms = self.db.get_max_terms(self.current_student)
        max_credits = self.db.get_student_max_credits(self.current_student)
        minimum_credits =  self.db.get_minimum_credits(self.current_student,
                                                                     (taken_courses + transfer_courses))
        return (max_terms  * max_credits) >   minimum_credits

    def _check_transfer_credits(self):
        if self.db.get_program(self.current_student).lower() == 'cs-minor':
            transfer_courses = self.db.get_transfers(self.current_student)
            taken_courses = self.db.get_taken(self.current_student)
            if len(transfer_courses) > 1:
                self.db.update_transfer_credits(self.current_student, max(transfer_courses, key=lambda
                    course: self._check_transfer_credits_helper(course)))

    def _check_transfer_credits_helper(self, course):
        taken_courses = self.db.get_taken(self.current_student)
        dependencies =  self.db.get_dependencies(course)
        return len(list(set(dependencies) - set(taken_courses)))

    def _previously_taken_assignment(self):
        assignment = defaultdict(int)
        for course in (self.db.get_transfers(self.current_student) + self.db.get_taken(self.current_student)):
            assignment[course] = 0
        return assignment

    def _get_minor_constraints(self,courses,weighted):
        constraints = defaultdict(list)
        binary_neighbors = defaultdict(list)
        transfers_and_taken = self.db.get_transfers(self.current_student) + self.db.get_taken(self.current_student)
        for course in courses:
            constraints[course].append(self.SchedulerConstraint(None, self.min_terms_constraint))
            self._get_prerequisite_constraints(course, constraints, binary_neighbors,transfers_and_taken)
            if course > 300 and course != 395:
                self._get_cutoff_credits_constraint(course,constraints)
            self._get_total_credits_constraint(course, constraints)

        return constraints,binary_neighbors


    def _get_major_constraints(self, courses, weighted):
        constraints = defaultdict(list)
        binary_neighbors = defaultdict(list)
        transfers_and_taken = self.db.get_transfers(self.current_student) + self.db.get_taken(self.current_student)
        for course in courses:
            constraints[course].append(self.SchedulerConstraint(None, self.min_terms_constraint))
            if course in self.electives:
                self._get_elective_constraint(course, constraints, weighted)
            self._get_prerequisite_constraints(course, constraints,binary_neighbors,transfers_and_taken)
            self._get_seasons_constraint(course, constraints)
            self._get_max_credits_constraint(course,constraints)
        return constraints,binary_neighbors

    def _get_prerequisite_constraints(self,course,constraints, binary_neighbors,transfers_and_taken):
        course_prerequisites = set(self.db.get_prerequisites(course))
        unassigned_prerequisites = course_prerequisites.difference(
                set(transfers_and_taken))
        for prerequisite in unassigned_prerequisites:
            binary_neighbors[course].append(prerequisite)
            binary_neighbors[prerequisite].append(course)
            temp_constraint = self.SchedulerConstraint((prerequisite, course), self.prerequisite_constraint)
            constraints[course].append(temp_constraint)
            constraints[prerequisite].append(temp_constraint)

    def _get_cutoff_credits_constraint(self, course,constraints):
        constraints[course].append(self.SchedulerConstraint(None,self.cut_off_credits_constraint))

    def _get_total_credits_constraint(self, course,constraints):
        constraints[course].append(self.SchedulerConstraint(None,self.total_credits_constraint))

    def _get_elective_constraint(self,course, constraints, weighted):
        if weighted:
            constraints[course].append(self.SchedulerConstraint(None, self.weighted_elective_constraint))
        else:
            constraints[course].append(self.SchedulerConstraint(None, self.elective_constraint))

    def _get_max_credits_constraint(self,course, constraints): \
        constraints[course].append(self.SchedulerConstraint(None, self.max_credits_constraint))

    def _get_seasons_constraint(self,course, constraints):
        if len((seasons := self.db.get_seasons(course))) < 2:
            constraints[course].append(self.SchedulerConstraint(tuple(seasons), self.season_constraint))

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

    def get_term_credits(self,course,term,assignment):
        courses_in_term = [key for key in assignment if (assignment[key] == term and key != course)]
        return sum([self.db.get_course_credits(course_in_term) for course_in_term in courses_in_term])
