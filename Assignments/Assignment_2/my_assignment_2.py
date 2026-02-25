import numpy as np
import math
import pandas as pd
import copy

def translation(x, y, z):
    return np.array(
        [
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1],
        ]
    )

def get_T01(theta_1): 
    # rotate by theta_1 along the y? axis
    # matrix = np.array([[np.cos(theta_1),0,np.sin(theta_1),0],
    #                   [0,1,0,0],
    #                   [-np.sin(theta_1),0,np.cos(theta_1),0],
    #                   [0,0,0,1]])
    # rotate by theta_1 along the z axis
    matrix = np.array([[np.cos(theta_1),-np.sin(theta_1),0,0],
                      [np.sin(theta_1),np.cos(theta_1),0,0],
                      [0,0,1,0],
                      [0,0,0,1]])
    return matrix

def get_T12(theta_2): 
    # translate 0.3m along y axis
    # rotate 90 degrees along the x axis, 
    # then rotate by theta_2 along the z axis
    tl = translation(0,0.3,0)
    matrix = np.array([[1,0,0,0],
                      [0,np.cos(math.pi/2),-np.sin(math.pi/2),0],
                      [0,np.sin(math.pi/2),np.cos(math.pi/2),0],
                      [0,0,0,1]])
    rot_matrix = np.array([[np.cos(theta_2),-np.sin(theta_2),0,0],
                      [np.sin(theta_2),np.cos(theta_2),0,0],
                      [0,0,1,0],
                      [0,0,0,1]])
    return np.matmul(tl,np.matmul(matrix,rot_matrix))

def get_T23(theta_3):
    # translate by 0.4 along x axis
    # rotate by theta_3 along the z axis
    tl = translation(0.4,0,0)
    matrix = np.array([[np.cos(theta_3),-np.sin(theta_3),0,0],
                      [np.sin(theta_3),np.cos(theta_3),0,0],
                      [0,0,1,0],
                      [0,0,0,1]])
    return np.matmul(tl,matrix)

def get_T34():
    # translate by 0.3 along y axis
    # rotate by -90 degrees along the x axis
    tl = translation(0,0.3,0)
    matrix = np.array([[1,0,0,0],
                      [0,np.cos(-math.pi/2),-np.sin(-math.pi/2),0],
                      [0,np.sin(-math.pi/2),np.cos(-math.pi/2),0],
                      [0,0,0,1]])
    return np.matmul(tl,matrix)

def get_FK(theta_1,theta_2,theta_3):
    t02 = np.matmul(get_T01(theta_1), get_T12(theta_2))
    t24 = np.matmul(get_T23(theta_3), get_T34())
    return np.matmul(t02,t24)

def ee_in_collision(angles_list,p_point,tolerance):
    end_effector_position = get_FK(angles_list[0],angles_list[1],angles_list[2])[0:3,3] 
    if (np.linalg.norm(end_effector_position - p_point) < tolerance): return True
    return False

def path_in_collision(path,object_list):
    our_ees = []
    for i in range(0,len(path)):
        # end_effector_position = get_FK(path[i][0],path[i][1],path[i][2])
        our_ees.append(path[i])
    for j in range(0,len(object_list)):
        print("sphere radius: ", object_list[j][1])
        for k in range(0,len(our_ees)):
            if(ee_in_collision(our_ees[k],object_list[j][0],object_list[j][1]) == True):
                return True
    return False

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

print("Hi!")
print(path_in_collision(path_1,object_list_2))