import pandas as pd
import os
from typing import Tuple
import copy

class DataBase:
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
    AREA = 'Area'
    INSTRUCTORS = "Instructors"
    TERMS = 'Terms'
    CREDITS = 'Credits'
    MAX_CREDITS = "maxcredits"
    ELECTIVE_COURSES_REQUIRED = 'elective_courses_required'
    COURSE_EXCLUSIONS = 'course_exclusions'
    TOTAL_CREDITS = 'total_credits'

    def __init__(self, course_file,student_file,requirements_file):
        self.course_list, self.program_requirements, self.student_list = DataBase.create_data_tables(course_file,
                                                                                                            student_file,
                                                                                                            requirements_file)
    def get_prerequisites(self, course) -> list[int]:
        return self.course_list.at[course, DataBase.PREREQUISITES]

    def get_program(self,current_student) -> str:
        return self.student_list.at[current_student, DataBase.PROGRAM]

    def get_course_credits(self, course) -> int:
        return self.course_list.at[course, DataBase.CREDITS]

    def get_student_max_credits(self,current_student) -> int:
        return self.student_list.at[current_student, DataBase.MAX_CREDITS]

    def get_taken(self,current_student) -> list[int]:
        return self.student_list.at[current_student, DataBase.TAKEN]

    def get_transfers(self,current_student)-> list[int]:
        return self.student_list.at[current_student, DataBase.TRANSFERS]

    def get_required_courses(self,current_student)-> list[int]:
        return self.program_requirements.at[self.get_program(current_student), DataBase.REQUIRED]

    def get_max_terms(self,current_student) -> int:
        return self.student_list.at[current_student, DataBase.MAX_TERMS]

    def get_max_electives(self,current_student) ->int:
        return self.program_requirements.at[self.get_program(current_student), DataBase.ELECTIVE_COURSES_REQUIRED]

    def get_all_courses(self)-> list[int]:
        return list(self.course_list.index)

    def get_seasons(self,course) -> list[str]:
        return self.course_list.at[course, DataBase.TERMS]

    def get_cutoff(self,current_student) ->int:
        return self.program_requirements.at[self.get_program(current_student), DataBase.CUTOFF]

    def get_course_exclusions(self,current_student) -> list[int]:
        return self.program_requirements.at[self.get_program(current_student), DataBase.COURSE_EXCLUSIONS]

    def get_program_credits_requirement(self,current_student) -> int:
        return self.program_requirements.at[self.get_program(current_student), DataBase.TOTAL_CREDITS]

    def get_minimum_credits(self, current_student, taken_and_transfer_courses) -> int:
        if self.get_program(current_student) == 'CS-minor':
            return 18 - sum(self.get_course_credits(course) for course in taken_and_transfer_courses)
        required_courses = self.get_required_courses(current_student)
        elective_courses = self.list_difference(self.course_list.index,required_courses)
        taken_elective_courses = list(set(elective_courses).intersection( set(taken_and_transfer_courses)))
        remaining_required_courses = self.list_difference(required_courses, taken_and_transfer_courses)
        required_credits_remaining = sum([self.get_course_credits(course) for course in remaining_required_courses])
        elective_courses_required = max(self.get_max_electives(current_student) - len(taken_elective_courses),0)
        return required_credits_remaining + (elective_courses_required * 3)

    def get_preference_value(self, current_student, course) -> int:
        preference_value = 0
        preferred_instructors = self.student_list.at[current_student, DataBase.PREFERENCES_INSTRUCTORS]
        course_instructors = self.course_list.at[course, DataBase.INSTRUCTORS]
        preference_value = preference_value + sum(
            [1 for instructor in course_instructors if instructor in preferred_instructors])
        preferred_topics = self.student_list.at[current_student, DataBase.PREFERENCES_TOPICS]
        course_topic = self.course_list.at[course, DataBase.AREA]
        if course_topic in preferred_topics:
            preference_value += 1
        return preference_value

    def get_dependencies(self, candidate_course:int) -> list[int]:
        dependencies = []
        for course in self.course_list.index:
            if candidate_course in self.get_prerequisites(course):
                dependencies.append(course)
        return dependencies

    def update_transfer_credits(self,current_student, new_value):
        self.student_list.at[current_student, DataBase.TRANSFERS] = [new_value]

    def student_rows(self):
        for index, row in self.student_list.iterrows():
            yield index, row

    def generate_available_courses(self, taken_courses: list[int]) -> pd.DataFrame:
        """
        Generates list of courses that student may enroll in for the given term.
        :param taken_courses: List of course IDs for all courses already taken by student.
        :return:Filtered copy of entire course_list dataframe containing only available courses.
        """
        course_candidates = self._recursive_deep_copy(self.course_list)
        course_candidates = course_candidates[~course_candidates.index.isin(taken_courses)]
        for course in taken_courses:
            course_candidates[DataBase.PREREQUISITES] = course_candidates[DataBase.PREREQUISITES].apply(
                lambda x: DataBase._remove_prerequisite(x, course))
        course_candidates = course_candidates[course_candidates[DataBase.PREREQUISITES].apply(lambda x: len(x) == 0)]
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



    @staticmethod
    def create_data_tables(course_file: str, student_file: str, program_requirements_file: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads data files into memory as tables for processing.
        :param course_file: The file containing the information for each course.
        :param student_file: The file containing the information for each student.
        :param program_requirements_file: The file containing the information for each program.
        :return: course_list, program_requirements, and student_list tables.
        """
        cwd = os.getcwd()
        course_list = DataBase._file_to_df(os.path.join(cwd, course_file))
        course_list.set_index(DataBase.COURSE, inplace=True)
        program_requirements = DataBase._file_to_df(os.path.join(cwd, program_requirements_file))
        program_requirements.set_index(DataBase.PROGRAM, inplace=True)
        student_list = DataBase._file_to_df(os.path.join(cwd, student_file))
        student_list.set_index(DataBase.ID, inplace=True)
        return course_list, program_requirements, student_list

    @staticmethod
    def _file_to_df(path: str) -> pd.DataFrame:
        if not os.path.isfile(path) and os.path.splitext(path)[1] != ".csv":
            raise ValueError(f"File {path} does not exist or is not a CSV file.")
        return DataBase._clean_data(pd.read_csv(path))

    @staticmethod
    def _clean_data(dframe):
        dframe.fillna("", inplace=True)
        for col in dframe.select_dtypes(include=[object]).columns:
            dframe[col] = dframe[col].apply(DataBase._clean_data_helper)
        return dframe

    @staticmethod
    def _clean_data_helper(values: str):
        values = values.replace("\"", "").strip()
        if "[" in values:
            return DataBase._to_list(values)
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

    @staticmethod
    def list_difference(list_a, list_b):
        return list(set(list_a) - set(list_b))