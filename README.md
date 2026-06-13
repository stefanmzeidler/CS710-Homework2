**Constraint Satisfaction**
  
Stefan Zeidler  
CS 710  
Dr. Susan McRoy   
March 15, 2025  
  
# Contents

[1. Introduction](#_Toc192982323)

[2. Background](#_Toc192982324)

[3. Pseudocode](#_Toc192982325)

[3.1 Classes](#_Toc192982326)

[3.2 Constraints](#_Toc192982327)

[3.3 CSP construction](#_Toc192982328)

[3.4 Creating Initial Assignments](#_Toc192982329)

[3.5 Solution Algorithms](#_Toc192982330)

[4. Results and Discussion](#_Toc192982331)

[5. Conclusion](#_Toc192982332)

[6. Appendix A – Test Cases](#_Toc192982333)

[7. Appendix B – Results Data](#_Toc192982334)

[7.1 Random Assignment](#_Toc192982335)

[7.2 First Fit](#_Toc192982336)

[7.3 Topological Min Conflicts](#_Toc192982337)

# Introduction

For this assignment we were tasked with solving a constraint satisfaction problem based on creating student course plans. I will first go over the background information regarding my approach and supply the pseudocode for my main functions. I will then compare the results for three different initial states.

# Background

For the design of the constraint satisfaction problem, I decided to use each course as a variable and each assignment was the term it would be taken in. From experience in homework 1, creating terms as variables and determining which courses were in requires factorial time complexity. I also found it easier to conceptualize constraints such as prerequisites must come before dependents using courses as variables. Using courses as the variables also meant that variables and constraints could be conceptualized as a nearly acyclic graph. If we treated the binary prerequisite constraints as directional, then we could transform the graph into a directed, acyclic graph and use a variation of topological sorting to contours of equal distance from the “start” of a student’s enrollment. Below I show the contours for a student who has no transfers or previously taken courses, but different contours can be generated based on which courses have already been taken.

<img width="779" height="703" alt="image" src="https://github.com/user-attachments/assets/75b34d03-5270-496c-8005-a27e83f7513d" />

If we proceed through these contours as we construct the CSP, then we can easily define a lower bound for the domains of each course as the contour since a course cannot be taken before its prerequisite. Doing this also provides us with a convenient sorting of the variables, and we can minimize the number of assignments needed if we assign courses in order of contour.

To test this hypothesis, I created several different test cases, ranging from students with no previously taken courses to students who have completed all requirements before even enrolling. These are the same test cases as used for Homework 1. I will use the min-conflicts heuristic to solve the problem from three different initial starting states: random variable order and assignment, first-fit assignment, and assignment combining the topological ordering with min-conflicts. The time and number of assignments will be recorded and compared for each test case, to see if there are any students that different algorithms perform better on. I will take the average time and assignments over several iterations for a single student from the three different starting states to measure overall performance.

# Pseudocode

I created a class for the CSP that is an extension of the AIME CSP class. The original implementation could only handle binary constraints, and no unary or global constraints. I had several of these which I will discuss as I introduce their pseudocode.

## Classes

<img width="975" height="643" alt="image" src="https://github.com/user-attachments/assets/4d0cd578-f388-4011-bb3d-896404479213" />


The class keeps track of variables and their domains and constraints. It also records the current assignments for the variables. It provides two functions to count the number of violated constraints for a given course and term assignment, needed for the min conflicts heuristic. The second function will return the list of courses that are still conflicted.

I also also created a class for each constraint that contains the scope of the constraint as well as the condition to check.

<img width="882" height="283" alt="image" src="https://github.com/user-attachments/assets/81043ac5-d2bb-4206-bfef-34137c61b157" />


Given the course, term, and assignment, the holds function will return whether the constraint has been violated. Having a scope let’s us use both unary, binary, and global constraints.

## Constraints

Tests whether prerequisites and dependents are in order. Also tests that 594 immediately follows 595.

<img width="882" height="524" alt="image" src="https://github.com/user-attachments/assets/cc06d670-3995-4a6c-896a-bef5192dfba5" />


Tests that if a term has more credits than the student allows.

<img width="855" height="171" alt="image" src="https://github.com/user-attachments/assets/9306134f-d4e7-471e-8b8a-11bdfc7cf556" />


Tests that the minimum number of terms are used. For an assignment, the constraint checks if the previous term does not contain any immediate course prerequisites. If not, it also checks if the previous term has enough space for the course. Prevents empty terms.

<img width="837" height="201" alt="image" src="https://github.com/user-attachments/assets/a67e1231-60f0-4c48-b099-a948bcf3e541" />

For CS-Minors, tests if the student has enrolled in too few or too many credits. A term assigned violates this constraint if setting an a term enrollment too none would result in too few courses or if setting a term enrollment to any term would result in too many courses.

<img width="830" height="208" alt="image" src="https://github.com/user-attachments/assets/6a1ca066-af5d-4197-9939-a33829240003" />


For CS-Minors, specifically for 300 level courses, checks if the student has enough credits above the 300 level.

<img width="857" height="241" alt="image" src="https://github.com/user-attachments/assets/a8ec8d78-1df6-4014-a54c-3bffac782a57" />


For BA and BS students, this constraint checks if they have too many or too few elective courses, under the assumption that students don’t want to take unnecessary courses.

<img width="834" height="312" alt="image" src="https://github.com/user-attachments/assets/ba961c3d-7390-4d2c-a255-723d11b981ae" />


For courses that are only held in Spring or Fall semesters, checks that the assignment does not violate this.

<img width="698" height="157" alt="image" src="https://github.com/user-attachments/assets/87aa52c0-ea05-4dbe-bb24-b93de3f52173" />


## CSP construction

<img width="849" height="591" alt="image" src="https://github.com/user-attachments/assets/b972eee8-885b-4fba-98b9-9f4f1affb32d" />


## Creating Initial Assignments

Creates one of three initial state types. For random, since the variables were inserted into the CSP in topological order, it shuffles the variables to undo this. It then assigns a random value from that variable’s domain. For first-fit search, it sorts the courses by number and then assigns a course to the first valid term that does not already have too many credits. For topological min conflicts, variables are assigned in topological order and values are chosen by the minimum number of conflicts. In the event of a tie, a lower value is chosen.

<img width="859" height="423" alt="image" src="https://github.com/user-attachments/assets/89863bc4-fc45-4b66-a288-00277efbeff5" />

## Solution Algorithms

This function creates the CSP and then passes it to the min\_conflicts algorithm.

<img width="848" height="177" alt="image" src="https://github.com/user-attachments/assets/c234cf5b-e581-444a-8321-76e3753ebff8" />


Adaption of the AIME min\_conflicts algorithm to include global and unary constraints, such as the maximum credits in a term, and if the a term is the correct season for a course. Allows for different starting assignments.

<img width="864" height="282" alt="image" src="https://github.com/user-attachments/assets/b5a4f388-18da-4ae6-9477-d0f94a94cee7" />

# Results and Discussion

The following table shows the average times in milliseconds and number of assignments required to find a solution over 15 iterations for the same student. Whereas first-fit and topological min-search begin with a deterministic initial state, random start is not and this was used to account for hard and easy cases in the mix.


|  | Random | First-fit | Topological |
| --- | --- | --- | --- |
| Time (ms) | 279 | 228 | 184 |
| Assignments | 53 | 33 | 22 |

The time to find a solution steadily decreased from one starting point to the next but still were very close overall. The number of assignments also steadily decreased between states, with topological requiring less than half the assignments that random start did. However, the difficulty of different cases still needs to be taken into account, so 17 different cases were used including both solvable and unsolvable assignments.

The time for unsolvable assignments was not included since these were caught by the prechecks, and not the algorithm. The full results can be found in Appendix B – Results Data.

The results broadly agree with the averages seen above, with random, first-fit and topological starting states each requiring less time and fewer assignments than the previous.

This is because each starting state type is successively closer to a goal state. Whereas a random assignment will likely place many prerequisites after their dependent courses, fit-first is much less likely to. This is because prerequisites are also lower in course number than their dependents. The maximum term credits constraint was also checked during creation so there were fewer constraints violated at the start.

One very interesting result was topological sort. The number of assignments includes the initial assignment of variables. For a student that has not taken any previous courses, this requires 22 variables to be assigned. This was the student used for the average testing (student 11 in Appendix A – Test Cases), and we can see that topological sorting reached the goal state with the initial assignment. In fact, for all the solvable test cases, the total number of assignments required was equal to the 22 – already taken courses, indicating that the initial assignment was the goal state for all of these.

This is because, by performing a topological ordering of the variables we can superimpose a tree over the graph, in which assigning the variables in order is guaranteed to leave values in the domains of subsequent variables that do not violate any constraints. Combining this with min conflicts heuristic for value selection, we can choose a value that has no conflicts at each step. Additional data for CS-minor students and BA students is included in text files which also support this claim.

# Conclusion

Based on the results above, performing a topological sorting, when possible, results in much higher performance for locl search, with the caveat that some problems are easier to reduce to tree than others. In other instances, a partial assignment such as fit-first may be more efficient if conversion of the constraint graph into a tree is much more expensive. Random selection performed worse than either of the two, which is to be expected since it makes no attempts to get close to the goal state at initial assignment. However, in comparison to the search algorithms from the previous assignment, local search for constraint satisfaction took significantly less than time. This time could be measured in milliseconds instead of seconds. For all starting states, constraint satisfaction generated significantly fewer nodes, never going above double digits, compared to the hundreds for uninformed search.

# Appendix A – Test Cases

| ID | program | transfers | taken | maxcredits | maxterms | | preferences-topics | preferences-instructors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | BS | [] | [] | 16 | | 5 | [AI, theory] | [Mali,Cheng] |
| 2 | BS | [250,251] | [351,317] | 15 | | 5 | [] | [] |
| 3 | BS | [425,459,469,422,557] | [150, 250] | 16 | | 7 | [] | [] |
| 4 | BS | [150,250,251,317,337,351,361,395,431,458,535,537,594,595] | [] | 16 | | 7 | [] | [] |
| 5 | BS | [] | [150,250,251,317,337,351,361,395,431,458,535,537,594,595] | 16 | | 7 | [] | [] |
| 6 | BS | [425,459,557] | [150,250,251,317,337,351,361,395,431,458,535,537,594,595] | 16 | | 7 | [] | [] |
| 7 | BS | [425,459,469,557] | [150,250,251,317,337,351,361,395,431,458,535,537,594,595] | 16 | | 7 | [] | [] |
| 8 | BS | [] | [150, 250] | 16 | | 7 | [] | [] |
| 9 | BS | [] | [150, 250] | 2 | | 7 | [] | [] |
| 10 | BS | [] | [150, 250] | 16 | | 1 | [] | [] |
| 11 | BS | [] | [] | 18 | | 7 | [] | [] |
| 12 | BS | [425,459,469,557] | [150,250,251,317,337,351,361,395,431,458,535,537,594,595] | 16 | | 1 | [] | [] |
| 13 | BS | [425,459,469,557] | [150,250,251,317,337,351,361,395,431,458,535,537,594,595] | 16 | | 0 | [] | [] |
| 14 | BS | [] | [] | 18 | | 0 | [] | [] |
| 15 | BS | [] | [] | 0 | | 7 | [] | [] |
| 16 | BS | [] | [] | 0 | | 0 | [] | [] |
| 17 | BS | [250,251] | [317,351] | 16 | | 8 | [] | [] |

# Appendix B – Results Data

## Random Assignment

Student id: 1

Student Program: BS

Solution Time (ms): N/A

Assignments used: 0

Course selection:

No Solution

Student id: 2

Student Program: BS

Solution Time (ms): 15

Assignments used: 32

Course selection:

Season Courses

Terms

0 F [250, 251, 351, 317]

1 S [431, 458, 150, 337, 459]

2 F [395, 361, 469, 535, 537]

3 S [594, 520]

4 F [595]

Student id: 3

Student Program: BS

Solution Time (ms): 15

Assignments used: 38

Course selection:

Season Courses

Terms

0 F [425, 459, 469, 422, 557, 150, 250]

1 S [317, 395, 251]

2 F [337, 351, 458]

3 S [431, 361, 535, 537]

4 F [594]

5 S [595]

Student id: 4

Student Program: BS

Solution Time (ms): 15

Assignments used: 16

Course selection:

Season Courses

Terms

0 F [150, 250, 251, 317, 337, 351, 361, 395, 431, 458, 535, 537, 594, 595]

1 S [557, 552, 459]

Student id: 5

Student Program: BS

Solution Time (ms): 15

Assignments used: 15

Course selection:

Season Courses

Terms

0 F [150, 250, 251, 317, 337, 351, 361, 395, 431, 458, 535, 537, 594, 595]

1 S [425, 520, 557]

Student id: 6

Student Program: BS

Solution Time (ms): 0

Assignments used: 9

Course selection:

Season Courses

Terms

0 F [425, 459, 557, 150, 250, 251, 317, 337, 351, 361, 395, 431, 458, 535, 537, 594, 595]

Student id: 7

Student Program: BS

Solution Time (ms): 0

Assignments used: 8

Course selection:

Season Courses

Terms

0 F [425, 459, 469, 557, 150, 250, 251, 317, 337, 351, 361, 395, 431, 458, 535, 537, 594, 595]

Student id: 8

Student Program: BS

Solution Time (ms): 31

Assignments used: 46

Course selection:

Season Courses

Terms

0 F [150, 250]

1 S [395, 317, 251]

2 F [351, 458, 337, 557, 459]

3 S [537, 361, 535, 431, 422]

4 F [594]

5 S [595]

Student id: 9

Student Program: BS

Solution Time (ms): N/A

Assignments used: 0

Course selection:

No Solution

Student id: 10

Student Program: BS

Solution Time (ms): N/A

Assignments used: 0

Course selection:

No Solution

Student id: 11

Student Program: BS

Solution Time (ms): 31

Assignments used: 46

Course selection:

Season Courses

Terms

1 S [395, 150, 250]

2 F [251, 317]

3 S [351, 458, 459, 337, 425]

4 F [537, 431, 361, 535]

5 S [520, 594]

6 F [595]

Student id: 12

Student Program: BS

Solution Time (ms): 0

Assignments used: 7

Course selection:

Season Courses

Terms

0 F [425, 459, 469, 557, 150, 250, 251, 317, 337, 351, 361, 395, 431, 458, 535, 537, 594, 595]

Student id: 13

Student Program: BS

Solution Time (ms): N/A

Assignments used: 0

Course selection:

No Solution

Student id: 14

Student Program: BS

Solution Time (ms): N/A

Assignments used: 0

Course selection:

No Solution

Student id: 15

Student Program: BS

Solution Time (ms): N/A

Assignments used: 0

Course selection:

No Solution

Student id: 16

Student Program: BS

Solution Time (ms): N/A

Assignments used: 0

Course selection:

No Solution

Student id: 17

Student Program: BS

Solution Time (ms): 15

Assignments used: 40

Course selection:

Season Courses

Terms

0 F [250, 251, 317, 351]

1 S [361, 535, 395, 337, 458]

2 F [537, 469, 594, 150, 557, 431]

3 S [595, 520]

## First Fit

Student id: 1

Student Program: BS

Solution Time (ms): N/A

Assignments used: 0

Course selection:

No Solution

Student id: 2

Student Program: BS

Solution Time (ms): 0

Assignments used: 29

Course selection:

Season Courses

Terms

0 F [250, 251, 351, 317]

1 S [150, 337, 361, 395, 458]

2 F [423, 431, 535, 537, 594]

3 S [520, 552, 595]

Student id: 3

Student Program: BS

Solution Time (ms): 0

Assignments used: 18

Course selection:

Season Courses

Terms

0 F [425, 459, 469, 422, 557, 150, 250]

1 S [251, 317, 395]

2 F [337, 351, 458]

3 S [361, 431, 535, 537]

4 F [594]

5 S [595]

Student id: 4

Student Program: BS

Solution Time (ms): 15

Assignments used: 14

Course selection:

Season Courses

Terms

0 F [150, 250, 251, 317, 337, 351, 361, 395, 431, 458, 535, 537, 594, 595]

1 S [422, 425, 552]

Student id: 5

Student Program: BS

Solution Time (ms): 15

Assignments used: 15

Course selection:

Season Courses

Terms

0 F [150, 250, 251, 317, 337, 351, 361, 395, 431, 458, 535, 537, 594, 595]

1 S [422, 520, 557]

Student id: 6

Student Program: BS

Solution Time (ms): 0

Assignments used: 10

Course selection:

Season Courses

Terms

0 F [425, 459, 557, 150, 250, 251, 317, 337, 351, 361, 395, 431, 458, 535, 537, 594, 595]

Student id: 7

Student Program: BS

Solution Time (ms): 0

Assignments used: 8

Course selection:

Season Courses

Terms

0 F [425, 459, 469, 557, 150, 250, 251, 317, 337, 351, 361, 395, 431, 458, 535, 537, 594, 595]

Student id: 8

Student Program: BS

Solution Time (ms): 15

Assignments used: 32

Course selection:

Season Courses

Terms

0 F [150, 250]

1 S [251, 317, 395]

2 F [337, 351, 458, 459, 557]

3 S [361, 422, 431, 535, 537]

4 F [594]

5 S [595]

Student id: 9

Student Program: BS

Solution Time (ms): N/A

Assignments used: 0

Course selection:

No Solution

Student id: 10

Student Program: BS

Solution Time (ms): N/A

Assignments used: 0

Course selection:

No Solution

Student id: 11

Student Program: BS

Solution Time (ms): 0

Assignments used: 30

Course selection:

Season Courses

Terms

1 S [150, 250, 395]

2 F [251, 317]

3 S [337, 351, 458, 459]

4 F [361, 423, 431, 535, 537]

5 S [552, 594]

6 F [595]

Student id: 12

Student Program: BS

Solution Time (ms): 0

Assignments used: 8

Course selection:

Season Courses

Terms

0 F [425, 459, 469, 557, 150, 250, 251, 317, 337, 351, 361, 395, 431, 458, 535, 537, 594, 595]

Student id: 13

Student Program: BS

Solution Time (ms): N/A

Assignments used: 0

Course selection:

No Solution

Student id: 14

Student Program: BS

Solution Time (ms): N/A

Assignments used: 0

Course selection:

No Solution

Student id: 15

Student Program: BS

Solution Time (ms): N/A

Assignments used: 0

Course selection:

No Solution

Student id: 16

Student Program: BS

Solution Time (ms): N/A

Assignments used: 0

Course selection:

No Solution

Student id: 17

Student Program: BS

Solution Time (ms): 15

Assignments used: 31

Course selection:

Season Courses

Terms

0 F [250, 251, 317, 351]

1 S [150, 337, 361, 395, 557]

2 F [431, 458, 459, 469, 535, 594]

3 S [537, 595]

## Topological Min Conflicts

Student id: 1

Student Program: BS

Solution Time (ms): N/A

Assignments used: 0

Course selection:

No Solution

Student id: 2

Student Program: BS

Solution Time (ms): 0

Assignments used: 19

Course selection:

Season Courses

Terms

0 F [250, 251, 351, 317]

1 S [150, 337, 361, 395, 422]

2 F [423, 431, 458, 459, 535]

3 S [537, 594]

4 F [595]

Student id: 3

Student Program: BS

Solution Time (ms): 15

Assignments used: 15

Course selection:

Season Courses

Terms

0 F [425, 459, 469, 422, 557, 150, 250]

1 S [251, 317, 395]

2 F [337, 351, 458]

3 S [361, 431, 535, 537]

4 F [594]

5 S [595]

Student id: 4

Student Program: BS

Solution Time (ms): 0

Assignments used: 8

Course selection:

Season Courses

Terms

0 F [150, 250, 251, 317, 337, 351, 361, 395, 431, 458, 535, 537, 594, 595]

1 S [422, 425, 459]

Student id: 5

Student Program: BS

Solution Time (ms): 0

Assignments used: 8

Course selection:

Season Courses

Terms

0 F [150, 250, 251, 317, 337, 351, 361, 395, 431, 458, 535, 537, 594, 595]

1 S [422, 425, 459]

Student id: 6

Student Program: BS

Solution Time (ms): 0

Assignments used: 5

Course selection:

Season Courses

Terms

0 F [425, 459, 557, 150, 250, 251, 317, 337, 351, 361, 395, 431, 458, 535, 537, 594, 595]

Student id: 7

Student Program: BS

Solution Time (ms): 15

Assignments used: 4

Course selection:

Season Courses

Terms

0 F [425, 459, 469, 557, 150, 250, 251, 317, 337, 351, 361, 395, 431, 458, 535, 537, 594, 595]

Student id: 8

Student Program: BS

Solution Time (ms): 0

Assignments used: 20

Course selection:

Season Courses

Terms

0 F [150, 250]

1 S [251, 317, 395]

2 F [337, 351, 458, 459, 469]

3 S [557, 361, 431, 535, 537]

4 F [594]

5 S [595]

Student id: 9

Student Program: BS

Solution Time (ms): N/A

Assignments used: 0

Course selection:

No Solution

Student id: 10

Student Program: BS

Solution Time (ms): N/A

Assignments used: 0

Course selection:

No Solution

Student id: 11

Student Program: BS

Solution Time (ms): 0

Assignments used: 22

Course selection:

Season Courses

Terms

1 S [150, 250, 395]

2 F [251, 317]

3 S [337, 351, 425, 458, 459]

4 F [469, 361, 431, 535, 537]

5 S [594]

6 F [595]

Student id: 12

Student Program: BS

Solution Time (ms): 0

Assignments used: 4

Course selection:

Season Courses

Terms

0 F [425, 459, 469, 557, 150, 250, 251, 317, 337, 351, 361, 395, 431, 458, 535, 537, 594, 595]

Student id: 13

Student Program: BS

Solution Time (ms): N/A

Assignments used: 0

Course selection:

No Solution

Student id: 14

Student Program: BS

Solution Time (ms): N/A

Assignments used: 0

Course selection:

No Solution

Student id: 15

Student Program: BS

Solution Time (ms): N/A

Assignments used: 0

Course selection:

No Solution

Student id: 16

Student Program: BS

Solution Time (ms): N/A

Assignments used: 0

Course selection:

No Solution

Student id: 17

Student Program: BS

Solution Time (ms): 0

Assignments used: 18

Course selection:

Season Courses

Terms

0 F [250, 251, 317, 351]

1 S [150, 337, 361, 395, 422]

2 F [423, 431, 458, 459, 535, 594]

3 S [537, 595]
