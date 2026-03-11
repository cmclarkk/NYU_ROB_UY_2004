import math
import numpy as np
import my_assignment_3

TOLERANCE  = 0.001

def unit_test(test_number, ee_position, theta):
    IK = my_assignment_3.inverse_kinematics_with_optimizer(ee_position)

    # Check position tolerance
    error = np.linalg.norm(IK - theta)    
    if error > TOLERANCE:
        print(" Failed unit test number ", test_number, " with position error ", error, "for output ", IK)
        return False

    print(" Passed unit test number ", test_number)
    return True


# format is [ee_position_x, ee_position_y, ee_position_z], [theta_1, theta_2, theta_3]
test_input_output_list = [
[np.array([0.2, -0.09, -0.14]), np.array([1.66818515, -0.11782383, -1.73263758])],
[np.array([0.0, -0.09, -0.14]), np.array([0.02800349, -0.08370324, -0.90483176])],
]


######################### MAIN ##########################
print("Running unit tests for assignment 3.2a:")
num_test_successes = 0
test_number = 1
for row in test_input_output_list:
    ee_position = row[0]
    theta = row[1]
    if unit_test(test_number, ee_position, theta ):
        num_test_successes += 1
    test_number += 1

print("---------------")
print("")
print("Num successful tests = ",num_test_successes, " / ", len(test_input_output_list))
print("")


