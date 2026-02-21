import math
import numpy as np
import my_assignment_2

def unit_test(test_number, path, object_list, path_in_collision_truth):
    path_in_collision_FK = my_assignment_2.path_in_collision(path, object_list)

    if (path_in_collision_FK and not path_in_collision_truth) or (not path_in_collision_FK and path_in_collision_truth):
        print(" Failed unit test number ", test_number, " with truth", path_in_collision_truth, "for output:", path_in_collision_FK)
        return False

    print(" Passed unit test number ", test_number)
    return True


# format is path, object_list, path_in_collision
path_1 = [
    [0,0,0],
    [0,0,-1/5*math.pi/2],
    [0,0,-2/5*math.pi/2],
    [0,0,-3/5*math.pi/2],
    [0,0,-4/5*math.pi/2],
    [0,0,-5/5*math.pi/2],
]
object_list_1 =[
    [np.array([111110,0,0]),0.2],
    [np.array([111110,0,0]),0.2],
]
object_list_2 =[
    [np.array([1,1,1]),0.2],
    [np.array([0.7,0.3,0]),0.2],
]
test_input_output_list = [
    [path_1, object_list_1, False],
    [path_1, object_list_2, True],
]


######################### MAIN ##########################
print("Running unit tests for assignment 2.2c:")
num_test_successes = 0
test_number = 1
for row in test_input_output_list:
    path = row[0]
    object_list = row[1]
    path_in_collision = row[2]
    if unit_test(test_number, path, object_list, path_in_collision):
        num_test_successes += 1
    test_number += 1

print("---------------")
print("")
print("Num successful tests = ",num_test_successes, " / ", len(test_input_output_list))
print("")


