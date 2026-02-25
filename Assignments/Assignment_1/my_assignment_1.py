import numpy as np
import pandas as pd
import copy
import math

def rotate2D(theta,p_point):
    rotation_array = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    q_point = np.matmul(rotation_array,p_point)
    return q_point

def rotate3D(theta,axis_of_rotation,p_point):
    rotation_array = 0

    if(axis_of_rotation == 'x'):
        rotation_array = np.array([[1,0,0],
                                   [0,np.cos(theta),-np.sin(theta)],
                                   [0,np.sin(theta),np.cos(theta)]])
    elif(axis_of_rotation == 'y'):
        rotation_array = np.array([[np.cos(theta),0,np.sin(theta)],
                                   [0,1,0],
                                   [-np.sin(theta),0,np.cos(theta)]])
    elif(axis_of_rotation == 'z'):
        rotation_array = np.array([[np.cos(theta),-np.sin(theta),0],
                                   [np.sin(theta),np.cos(theta),0],
                                   [0,0,1]])
    else:
        raise Exception("Invalid coordinate frame")
    q_point = np.matmul(rotation_array,p_point)
    return q_point

def rotate3D_many_times(rotations,p_point):
    q_point = copy.deepcopy(p_point)
    for i in range(0,len(rotations)):
        q_point = rotate3D(rotations[i][0],rotations[i][1],q_point)
    return q_point

rotate3D_many_times([[math.pi/2, 'z'], [math.pi/2, 'x']], np.array([1,0,0]))