import unittest
from scheduler import *

class MyTestCase(unittest.TestCase):

    def setUp(self):
        self.scheduler = Scheduler()
        self.scheduler.initial_season = 'F'
        self.scheduler.course_list,  self.scheduler.program_requirements,  self.scheduler.student_list = scheduler_utils.create_data_tables(
            'courses.csv',
            'students.csv',
            "program_requirements.csv")
        self.scheduler.current_student = 1

    def test_prerequisite_constraint(self):
        assignment = {250:2}
        prerequisite = Scheduler.SchedulerConstraint((250,351), Scheduler.prerequisite_constraint)
        self.assertTrue(prerequisite.holds(351,3,assignment))
        self.assertFalse(prerequisite.holds(351,1,assignment))
        self.assertFalse(prerequisite.holds(351,2,assignment))
        assignment = {351:3}
        self.assertTrue(prerequisite.holds(250,2,assignment))
        self.assertFalse(prerequisite.holds(250,3,assignment))
        self.assertFalse(prerequisite.holds(250,4,assignment))

    def test_594_prerequisite_constraint(self):
        assignment = {594: 3}
        prerequisite = Scheduler.SchedulerConstraint((594, 595), Scheduler.prerequisite_constraint)
        self.assertTrue(prerequisite.holds(595, 4, assignment))
        self.assertFalse(prerequisite.holds(595, 3, assignment))
        self.assertFalse(prerequisite.holds(595, 2, assignment))
        self.assertFalse(prerequisite.holds(595, 1, assignment))
        self.assertFalse(prerequisite.holds(595, 5, assignment))
        assignment = {595: 3}
        self.assertTrue(prerequisite.holds(594, 2, assignment))
        self.assertFalse(prerequisite.holds(594, 1, assignment))
        self.assertFalse(prerequisite.holds(594, 3, assignment))
        self.assertFalse(prerequisite.holds(594, 4, assignment))
        self.assertFalse(prerequisite.holds(594, 5, assignment))

    def test_season_constraint(self):
        assignment = {250:2}
        season_constraint = Scheduler.SchedulerConstraint(('F',), self.scheduler.season_constraint)
        self.assertTrue(season_constraint.holds(250,3,assignment))
        self.assertFalse(season_constraint.holds(250, 2, assignment))
        season_constraint = Scheduler.SchedulerConstraint(('S',), self.scheduler.season_constraint)
        self.assertTrue(season_constraint.holds(250,2,assignment))
        self.assertFalse(season_constraint.holds(250, 3, assignment))

    def test_max_credits_constraint(self):
        assignment = {422:3,423:3,425:3,431:3}
        max_credits_constraint = Scheduler.SchedulerConstraint(None, self.scheduler.max_credits_constraint)
        self.assertTrue(max_credits_constraint.holds(361,2,assignment))
        self.assertFalse(max_credits_constraint.holds(361,3,assignment))
        assignment = {361: 3, 423: 3, 425: 3, 431: 3}
        self.assertTrue(max_credits_constraint.holds(361,3,assignment))
        self.assertTrue(max_credits_constraint.holds(361, 2, assignment))

    def test_electives_constraint(self):
        all_courses = self.scheduler.course_list.index
        program = 'BS'
        self.scheduler.required_courses = self.scheduler.program_requirements.at[program, Scheduler.REQUIRED]
        self.scheduler.electives = list(set(all_courses) - set(self.scheduler.required_courses))
        electives_constraint = Scheduler.SchedulerConstraint(None,self.scheduler.elective_constraint)
        assignment = {422:3,423:3,425:3}
        self.assertTrue(electives_constraint.holds(422,2,assignment))
        self.assertFalse(electives_constraint.holds(552, 2, assignment))
        assignment = {351:3,423:3,425:3}
        self.assertTrue(electives_constraint.holds(422,2,assignment))
        self.assertTrue(electives_constraint.holds(552, 2, assignment))


#
# if __name__ == '__main__':
#     unittest.main()
