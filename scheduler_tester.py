import unittest
from scheduler import *
from database import *
class MyTestCase(unittest.TestCase):


    def test_success(self):
        files = ['test_bs_success.csv', 'test_ba_success.csv', 'test_cs_minor_success.csv']
        for file in files:
            my_scheduler = Scheduler('F','courses.csv', file, 'program_requirements.csv')
            solutions = my_scheduler.plan(print_to_screen = False)
            for index, solution in enumerate(solutions):
                with self.subTest(i = index):
                    student_id = solution[0]
                    program = solution[1]
                    n_assignments = solution[2]
                    course_selection = solution[3]
                    self.assertGreater(student_id, 0)
                    self.assertIsNotNone(program)
                    self.assertIsNotNone(n_assignments)
                    self.assertGreaterEqual(n_assignments,0)
                    self.assertIsNotNone(course_selection)
                    terms = course_selection.index
                    max_terms = my_scheduler.db.get_max_terms(student_id)
                    if terms[0] == 0:
                        self.assertLessEqual(len(terms) - 1,max_terms)
                    else:
                        self.assertLessEqual(len(terms), max_terms)

    def test_failure(self):
        files = ['test_bs_failure.csv', 'test_ba_failure.csv', 'test_cs_minor_failure.csv']
        for file_index,file in enumerate(files):
            print(f"Testing round {file_index}:\n")
            my_scheduler = Scheduler('F', 'courses.csv', file, 'program_requirements.csv')
            solutions = my_scheduler.plan(print_to_screen=False)
            for index, solution in enumerate(solutions):
                with self.subTest(i=index):
                    student_id = solution[0]
                    program = solution[1]
                    n_assignments = solution[2]
                    course_selection = solution[3]
                    self.assertGreater(student_id, 0)
                    self.assertIsNotNone(program)
                    self.assertIsNone(n_assignments)
                    self.assertIsNone(course_selection)


if __name__ == '__main__':
    unittest.main()
