import math
import numpy as np
import my_assignment_2

def unit_test(test_number, theta_list, object_position, tolerance, in_collision):
    in_collision_FK = my_assignment_2.ee_in_collision(theta_list, object_position, tolerance)
    # Check position tolerance
    if (in_collision and not in_collision_FK) or (not in_collision and in_collision_FK):
        print(" Failed unit test number ", test_number, " with collision truth:", in_collision, "for output:", in_collision_FK)
        return False

    print(" Passed unit test number ", test_number)
    return True


# format is [theta_1, theta_2, theta_3], object_position, tolerance, in_collision
test_input_output_list = [
[[0, 0, 0], np.array([0,0,99999]), 0, False],
[[0, 0, 0], np.array([0.4,0.3,0.3]), 0.0001, True],
[[0, 0, 0], np.array([0.4,0.3,0.4]), 0.0001, False],
[[0, 0, 0], np.array([0.4,0.3,0.4]), 0.1001, True],
]


######################### MAIN ##########################
print("Running unit tests for assignment 2.2b:")
num_test_successes = 0
test_number = 1
for row in test_input_output_list:
    theta_list = row[0]
    object_position = row[1]
    tolerance = row[2]
    in_collision = row[3]
    if unit_test(test_number, theta_list, object_position, tolerance, in_collision):
        num_test_successes += 1
    test_number += 1

print("---------------")
print("")
print("Num successful tests = ",num_test_successes, " / ", len(test_input_output_list))
print("")


