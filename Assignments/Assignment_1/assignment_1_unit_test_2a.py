import math
import numpy as np
import my_assignment_1

TOLERANCE  = 0.001

def unit_test(test_number, theta, p_point, q_point_des):
    q_point = my_assignment_1.rotate2D(theta, p_point)
    error = np.linalg.norm(q_point - q_point_des)
    if error < TOLERANCE:
        print(" Passed unit test number ", test_number)
        return True

    print(" Failed unit test number ", test_number, " with error ", error, "for output q ", q_point)
    return False

# format is theta, p_point, q_point
test_input_output_list = [
[0, np.array([1,0]), np.array([1,0])],
[math.pi/2, np.array([1,0]), np.array([0,1])],
[-math.pi/2, np.array([1,0]), np.array([0,-1])],
[-math.pi, np.array([1,0]), np.array([-1,0])],
[2*math.pi, np.array([1,0]), np.array([1,0])],
]


######################### MAIN ##########################
print("Running unit tests for assignment 1.2a:")
num_test_successes = 0
test_number = 1
for row in test_input_output_list:
    theta = row[0]
    p_point = row[1]
    q_point_des = row[2]
    if unit_test(test_number, theta, p_point, q_point_des):
        num_test_successes += 1
    test_number += 1

print("---------------")
print("")
print("Num successful tests = ",num_test_successes, " / ", len(test_input_output_list))
print("")


