import numpy as np

def get_T01(theta_1):

    c,s = np.cos(theta_1), np.sin(theta_1)

    # rotation theta_1 about z-axis

    T01 = np.array([[c, -s, 0, 0], 
                    [s,  c, 0, 0],
                    [0,  0, 1, 0],
                    [0,  0, 0, 1]])
    
    return T01

def get_T12(theta_2):
    c,s = np.cos(theta_2), np.sin(theta_2)

    # rotation theta_2 about frame 2 z-axis
    # rotation 90° about x-axis
    # translation along frame 1 y-axis

    T12 = np.array([[c, -s, 0, 0],
                    [0,  0, -1, 0.3],
                    [s,  c, 0, 0], 
                    [0,  0, 0, 1]])
    
    return T12

def get_T23(theta_3):
    c,s = np.cos(theta_3), np.sin(theta_3)

    # rotation theta_3 about z-axis
    # translation along x-axis
    T23 = np.array([[c, -s, 0, 0.4],
                    [s,  c, 0, 0],
                    [0,  0, 1, 0],
                    [0,  0, 0, 1]])
    
    return T23

def get_T34():
    # rotation -90° about x-axis
    # translation along y-axis
    T34 = np.array([[1, 0, 0, 0],
                    [0, 0, 1, 0.3],
                    [0, -1, 0, 0],
                    [0, 0, 0, 1]])
    
    return T34

def get_FK(theta_1, theta_2, theta_3):
    FK = get_T01(theta_1) @ get_T12(theta_2) @ get_T23(theta_3) @ get_T34()

    return FK

def get_EE(forward_kinematics):
    return forward_kinematics[0:3,3]

def ee_in_collision(thetas, p_point, tolerance):
    # get the end effector position
    FK = get_FK(thetas[0], thetas[1], thetas[2])
    EE = get_EE(FK)

    distance = np.linalg.norm(EE - p_point)
    
    return distance < tolerance

def path_in_collision(path, object_list):

    collision = False

    for angle in path:
        EE = get_EE(get_FK(angle[0], angle[1], angle[2]))
        for object in object_list:
            if np.linalg.norm(EE - object[0]) < object[1]:
                collision = True

    return collision




