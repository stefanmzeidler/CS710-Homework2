from scheduler import Scheduler
import time


scheduler = Scheduler('S', 'courses.csv','student_assignments_test_cases.csv','program_requirements.csv')
scheduler.plan(state_creation_mode='random')
scheduler = Scheduler('S', 'courses.csv','student_assignments_test_cases.csv','program_requirements.csv')
scheduler.plan(state_creation_mode='first_fit')
scheduler = Scheduler('S', 'courses.csv','student_assignments_test_cases.csv','program_requirements.csv')
scheduler.plan(state_creation_mode='topological_min_conflicts')
scheduler = Scheduler('F', 'courses.csv','student_assignments_test_cases.csv','program_requirements.csv')
scheduler.plan(state_creation_mode='topological_min_conflicts')

s = time.process_time_ns()