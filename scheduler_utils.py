import os
import pandas as pd

COURSE = 'Course'
ID = 'ID'
PROGRAM = 'program'
PREREQUISITES = "Prerequisites"

def create_data_tables(course_file: str, student_file: str, program_requirements_file: str):
    """
    Loads data files into memory as tables for processing.
    :param course_file: The file containing the information for each course.
    :param student_file: The file containing the information for each student.
    :param program_requirements_file: The file containing the information for each program.
    :return: course_list, program_requirements, and student_list tables.
    """
    cwd = os.getcwd()
    course_list = _file_to_df(os.path.join(cwd, course_file))
    course_list.set_index(COURSE, inplace=True)
    program_requirements = _file_to_df(os.path.join(cwd, program_requirements_file))
    program_requirements.set_index(PROGRAM, inplace=True)
    student_list = _file_to_df(os.path.join(cwd, student_file))
    student_list.set_index(ID, inplace=True)
    return course_list, program_requirements, student_list

def _file_to_df(path: str) -> pd.DataFrame:
    if not os.path.isfile(path) and os.path.splitext(path)[1] != ".csv":
        raise ValueError(f"File {path} does not exist or is not a CSV file.")
    return _clean_data(pd.read_csv(path))


def _clean_data(dframe):
    dframe.fillna("", inplace=True)
    for col in dframe.select_dtypes(include=[object]).columns:
        dframe[col] = dframe[col].apply(_clean_data_helper)
    return dframe


def _clean_data_helper(values: str):
    values = values.replace("\"", "").strip()
    if "[" in values:
        return _to_list(values)
    return int(values) if values.isdigit() else values


def _to_list(values):
    values = values.replace("[", "").replace("]", "").strip()
    values = values.split(",")
    if values[0] == "":
        return []
    elif values[0].isdigit():
        return list(map(int, values))
    else:
        return [value.strip() for value in values]