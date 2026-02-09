import numpy as np


def rotate2D(theta, p_point):
    rot_mat = np.array([[np.cos(theta), np.sin(-theta)],
                        [np.sin(theta), np.cos(theta)]])
    
    q_point = rot_mat @ p_point

    return q_point


def rotate3D(theta, axis_of_rotation, p_point):

    c, s = np.cos(theta), np.sin(theta)
    
    if axis_of_rotation == 'x':
        rot_mat = np.array([[1, 0, 0],
                            [0, c, -s],
                            [0, s, c]])
    if axis_of_rotation == 'y':
        rot_mat = np.array([[c, 0, s],
                            [0, 1, 0],
                            [-s, 0, c]])
        
    if axis_of_rotation == 'z':
        rot_mat = np.array([[c, -s, 0],
                            [s, c, 0],
                            [0, 0, 1]])
        
    q_point = rot_mat @ p_point

    return q_point

def rotate3D_many_times(rotation_list, p_point):
    curr_point = p_point
    
    for (theta, axis_of_rotation) in rotation_list:
        curr_point = rotate3D(theta, axis_of_rotation, curr_point)

    q_point = curr_point

    return q_point