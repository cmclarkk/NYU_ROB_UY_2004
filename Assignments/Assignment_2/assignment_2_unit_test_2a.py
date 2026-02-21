import math
import numpy as np
import my_assignment_2

TOLERANCE  = 0.001

def unit_test(test_number, theta_1, theta_2, theta_3, ee_position, ee_R):
    FK = my_assignment_2.get_FK(theta_1, theta_2, theta_3)

    # Check position tolerance
    ee_position_FK = FK[0:3,3] 
    error = np.linalg.norm(ee_position_FK - ee_position)    
    if error > TOLERANCE:
        print(" Failed unit test number ", test_number, " with position error ", error, "for output ", ee_position_FK)
        return False
    
    # Check rotation tolerance
    ee_rotation_FK = FK[0:3,0:3] 
    eye_FK = ee_rotation_FK @ np.linalg.inv(ee_R)
    diff = np.identity(3) - eye_FK

    error = np.linalg.norm(diff)    
    if error > TOLERANCE:
        print(" Failed unit test number ", test_number, " with rotation error ", error, "for output ", ee_rotation_FK)
        return False

    print(" Passed unit test number ", test_number)
    return True


# format is theta_1, theta_2, theta_3, ee_position, ee_Rotation
test_input_output_list = [
[0, 0, 0, np.array([0.4,0.3,0.3]), np.array([[1,0,0],[0,1,0],[0,0,1]])],
[0, 0, -math.pi/2, np.array([0.7,0.3,0.0]), np.array([[0,0,1],[0,1,0],[-1,0,0]])],
[0, math.pi, 0, np.array([-0.4,0.3,-0.3]), np.array([[-1,0,0],[0,1,0],[0,0,-1]])],
[0, math.pi/2, -math.pi/2, np.array([0.0,0.3,0.7]), np.array([[1,0,0],[0,1,0],[0,0,1]])],
[math.pi, 0, -math.pi/2, np.array([-0.7,-0.3,0]), np.array([[0,0,-1],[0,-1,0],[-1,0,0]])],
]


######################### MAIN ##########################
print("Running unit tests for assignment 2.2a:")
num_test_successes = 0
test_number = 1
for row in test_input_output_list:
    theta_1 = row[0]
    theta_2 = row[1]
    theta_3 = row[2]
    ee_position = row[3]
    ee_R= row[4]
    if unit_test(test_number, theta_1, theta_2, theta_3, ee_position, ee_R):
        num_test_successes += 1
    test_number += 1

print("---------------")
print("")
print("Num successful tests = ",num_test_successes, " / ", len(test_input_output_list))
print("")


