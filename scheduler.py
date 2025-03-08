"""
File name: scheduler.py
Author: Stefan Zeidler
Created on: 3/7/2025
Description: Scheduling program using constraint search to create course plans for students.
Adapted from https://github.com/aimacode/aima-python/blob/master/csp.py.
"""

from csp import *
import copy
import pandas as pd
import os

class Scheduler:
    COURSE = 'Course'
    ID = 'ID'
    PROGRAM = 'program'
    PREREQUISITES = "Prerequisites"

    def __init__(self):
        self.initial_season = None
        self.course_list = None
        self.program_requirements = None
        self.student_list = None


    @staticmethod
    def _create_data_tables(course_file: str, student_file: str, program_requirements_file: str):
        """
        Loads data files into memory as tables for processing.
        :param course_file: The file containing the information for each course.
        :param student_file: The file containing the information for each student.
        :param program_requirements_file: The file containing the information for each program.
        :return: course_list, program_requirements, and student_list tables.
        """
        cwd = os.getcwd()
        course_list = Scheduler._file_to_df(os.path.join(cwd, course_file))
        course_list.set_index(Scheduler.COURSE, inplace=True)
        program_requirements = Scheduler._file_to_df(os.path.join(cwd, program_requirements_file))
        program_requirements.set_index(Scheduler.PROGRAM, inplace=True)
        student_list = Scheduler._file_to_df(os.path.join(cwd, student_file))
        student_list.set_index(Scheduler.ID, inplace=True)
        return course_list, program_requirements, student_list

    @staticmethod
    def _file_to_df(path: str) -> pd.DataFrame:
        if not os.path.isfile(path) and os.path.splitext(path)[1] != ".csv":
            raise ValueError(f"File {path} does not exist or is not a CSV file.")
        return Scheduler._clean_data(pd.read_csv(path))

    @staticmethod
    def _clean_data(dframe):
        dframe.fillna("", inplace=True)
        for col in dframe.select_dtypes(include=[object]).columns:
            dframe[col] = dframe[col].apply(Scheduler._clean_data_helper)
        return dframe

    @staticmethod
    def _clean_data_helper(values: str):
        values = values.replace("\"", "").strip()
        if "[" in values:
            return Scheduler._to_list(values)
        return int(values) if values.isdigit() else values

    @staticmethod
    def _to_list(values):
        values = values.replace("[", "").replace("]", "").strip()
        values = values.split(",")
        if values[0] == "":
            return []
        elif values[0].isdigit():
            return list(map(int, values))
        else:
            return [value.strip() for value in values]

    def plan(self, initial_season: str, course_file: str, student_file: str):
        self.initial_season = initial_season
        self.course_list, self.program_requirements, self.student_list = Scheduler._create_data_tables(course_file,
                                                                                                       student_file,
                                                                                                       "program_requirements.csv")
    def _create_csp(self,student):
        term = 1
        courses = self._generate_available_courses([])
        while not courses.empty:


            ...

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

