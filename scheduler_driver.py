from scheduler import Scheduler
import time
import gc
from collections import defaultdict

def get_avg_times(iterations=15):
    times = defaultdict(int)
    state_creation_modes = ['random', 'first_fit', 'topological_min_conflicts']
    for mode in state_creation_modes:
        t = 0
        for i in range(iterations):
            scheduler = Scheduler('S', 'courses.csv', 'students.csv', 'program_requirements.csv')
            gc.collect()
            start = time.process_time()
            scheduler.plan(print_to_screen = False, state_creation_mode=mode)
            end = time.process_time()
            t += end - start
        times[mode] = int((t/iterations) * 1000)
    return times

def get_avg_assignments(iterations=15):
    avg_assigns = defaultdict(int)
    state_creation_modes = ['random', 'first_fit', 'topological_min_conflicts']
    for mode in state_creation_modes:
        n_assigns = 0
        for i in range(iterations):
            scheduler = Scheduler('S', 'courses.csv', 'single_student.csv', 'program_requirements.csv')
            solution = scheduler.plan(print_to_screen=False, state_creation_mode=mode)
            n_assigns += solution[0][3]
        avg_assigns[mode] = int(n_assigns / iterations)
    return avg_assigns

def main():
    print("Solutions using random start:\n")
    scheduler = Scheduler('S', 'courses.csv', 'students.csv', 'program_requirements.csv')
    scheduler.plan(write_to_file = True,state_creation_mode='random')
    print("\nSolutions using first fit:\n")
    scheduler = Scheduler('S', 'courses.csv', 'students.csv', 'program_requirements.csv')
    scheduler.plan(write_to_file = True,state_creation_mode='first_fit')
    print("\nSolutions using topological min conflicts:\n")
    scheduler = Scheduler('S', 'courses.csv', 'students.csv', 'program_requirements.csv')
    scheduler.plan(write_to_file = True,state_creation_mode='topological_min_conflicts')
    print("\nSolutions using topological min conflicts:\n")
    scheduler = Scheduler('S', 'courses.csv', 'test_cs_minor_success.csv', 'program_requirements.csv')
    scheduler.plan(write_to_file=True, state_creation_mode='topological_min_conflicts')
    scheduler = Scheduler('S', 'courses.csv', 'test_ba_success.csv', 'program_requirements.csv')
    scheduler.plan(write_to_file=True, state_creation_mode='topological_min_conflicts')
    print("Getting averages...\n")
    with open ('averages.txt', 'w') as f:
        iterations = 15
        avg_times = get_avg_times(iterations)
        str_1 = f"Average times (ms) over {iterations} iterations by initial state: "
        print(str_1)
        print(str(avg_times))
        f.write(str_1 + "\n")
        f.write(str(avg_times) + "\n")
        avg_assigns = get_avg_assignments(iterations)
        str_2 = f"Average assignments over {iterations} iterations by initial state: "
        print(str_2)
        print(str(avg_assigns))
        f.write(str_2 + "\n")
        f.write(str(avg_assigns) + "\n")

# main()
scheduler = Scheduler('S', 'courses.csv', 'test_ba_success.csv', 'program_requirements.csv')
scheduler.plan(write_to_file=True, state_creation_mode='topological_min_conflicts')