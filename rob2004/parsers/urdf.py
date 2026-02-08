from urdf_parser_py.urdf import URDF, Sphere
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from pylab import rcParams # from package matplotlib
from scipy.spatial.transform import Rotation as Rot
import numpy as np

def hatSO3(u):
    p = u.squeeze()
    p_cross = np.array([[ 0,       -p[2],        p[1]],
                        [ p[2],     0,          -p[0]],
                        [-p[1],     p[0],        0]
                       ])
    return p_cross

def veeSE3(u):
    omega = np.array([-u[1,2], u[0,2], -u[0,1]]).reshape(3,1)
    v = u[0:3,-1].reshape(3,1)
    return np.vstack([v, omega])

def veeSO3(u):
    omega = np.array([-u[1,2], u[0,2], -u[0,1]]).reshape(3,1)
    return omega

def quat2R(q):
    return Rot.from_quat(q).as_matrix()

class URDFParser:
    def __init__(self, urdf_path, floating_base=False):
        self.q = None
        self.urdf_path = urdf_path
        try:
            self.urdf = URDF.from_xml_string(open(self.urdf_path).read())
        except:
            raise Exception('Invalid URDF file!')
        
        self.link_map = self.urdf.link_map
        joints = self.urdf.joints
        self.n_joints = len(joints)
        # Extract all joint information
        self.joints_info = {}
        self.edge_to_joint_map = {}
        self.actuated_joints = {}
        actuator_idx = 0
        for joint in joints:
            
            if joint.joint_type in ['revolute']:
                self.actuated_joints[f'{joint.parent}-{joint.child}']=actuator_idx
                actuator_idx+=1
                
            elif joint.joint_type not in ['fixed']:
                raise NotImplementedError(f'Joint type {joint.joint_type} is not implmented.')
            
            self.joints_info[f'{joint.parent}-{joint.child}'] = {
                                       'axis': joint.axis, 
                                       'joint_type': joint.joint_type, 
                                       'limit': joint.limit, 
                                       'rpy': joint.origin.rpy, 
                                       'xyz': joint.origin.xyz,
                                       'joint_offset': self._get_transform(joint.origin.rpy, joint.origin.xyz),
                                       'parent':joint.parent,
                                       'child': joint.child,
                                       'joint_pose': np.eye(4),
                                      }
            self.edge_to_joint_map[f'{joint.parent}-{joint.child}']=joint.name
        
        # Extract all the edges
        edges = {}
        self.link_names = []
        for i in range(self.n_joints):
            edges[joints[i].child] = joints[i].parent
            if joints[i].child not in self.link_names:
                self.link_names.append(joints[i].child)
            if joints[i].parent not in self.link_names:
                self.link_names.append(joints[i].parent)

        # if the parent link is not a child of any other links, it's the root link
        for parent in edges.values():
            if parent not in edges.keys():
                root_link = parent
        self.edges = edges
        self.root_link = root_link
        # Extract the children list of each parent
        self.children_of_parent = {p:[] for p in self.link_names}
        for link_name in self.link_names:
            if link_name==self.root_link:
                continue
            parent = self.edges[link_name]
            self.children_of_parent[parent].append(link_name)

        # Extract the name of leaf links
        self.leaf_links = []
        for parent in self.children_of_parent.keys():
            if len(self.children_of_parent[parent])==0: # if the parent has no children, it's a leaf node
                self.leaf_links.append(parent)

        # Extract the path from root to each leaf

        self.kinematic_paths = {}
        for leaf_link in self.leaf_links:
            path = []
            child = leaf_link
            while child!=self.root_link:
                path.append(child)
                child = self.edges[child]
            path.append(child)
            path.reverse()
            self.kinematic_paths[leaf_link]=path

        # Extract the link information
        self.links_info={}
        for link_name in self.link_names:
            self.links_info[link_name]=self.extractLinkInfo(self.link_map[link_name])

        self.floating_base = floating_base

    def _get_transform(self, rpy, pos):
        R = Rot.from_euler('xyz', rpy).as_matrix().squeeze()
        t = np.array(pos).reshape(3,1)
        T = np.hstack([R, t])
        return np.vstack([T, np.array([0., 0., 0., 1.])])
    
    def _get_joint_twist(self, ax, joint_type):
        if joint_type=='revolute':
            omega_cross = hatSO3(ax)
            v = np.zeros((3,1))
            xi = np.hstack([omega_cross, v])
            xi = np.vstack([xi, np.zeros((1,4))])
            return xi
        elif joint_type=='prismatic':
            raise NotImplementedError('Prismatic joints are not implemented yet')
        else:
            raise Exception('Joint type is not implemented')

    def visulaize(self):
        G = nx.DiGraph()
        for parent, child in self.edges.items():
            nx.add_path(G, [f'{parent}', f'{child}'])

        rcParams['figure.figsize'] = 14, 10
        pos=graphviz_layout(G, prog='dot')
        nx.draw(G, pos=pos,
                node_color='lightgreen', 
                node_size=1500,
                with_labels=True, 
                arrows=True)

    def extractLinkInfo(self, l):
        try:
            xyz = l.origin.xyz 
            rpy = l.origin.rpy
            link_offset = self._get_transform(rpy, xyz)
        except:
            link_offset = np.eye(4)
        try: 
            inertia = l.inertial
        except:
            inertia = None

        visuals = []
        if len(l.visuals)>0:
            for visual in l.visuals:                
                try:
                    rpy = np.array(visual.origin.rpy).squeeze(),
                    xyz = np.array(visual.origin.xyz).squeeze(),
                    visual_offset = self._get_transform(rpy, xyz)
                except:
                    visual_offset = np.eye(4)
                   
                try:
                    file_path = visual.geometry.filename
                except:
                    type='unsupported'
                    file_path = None
                visuals.append({'visual_offset':visual_offset, 'file_path':file_path})

        collisions = []
        if len(l.collisions) > 0:
            for collision in l.collisions:
                try:
                    rpy = collision.origin.rpy,
                    xyz = collision.origin.xyz,
                    collision_offset = self._get_transform(rpy, xyz)
                except:
                    collision_offset = np.eye(4)
                    
                if isinstance(collision.geometry, Sphere):
                    scale = 3*[collision.geometry.radius]
                    type='shpere'
                else:
                    scale = 3*[0.]
                    type='unsupported'
                
                collisions.append({'collision_offset':collision_offset, 'scale':scale, 'type':type})

        return {'name': l.name, 'link_pose':np.eye(4), 'link_pose_valid': False, 'link_offset':link_offset, 'inertia':inertia, 'visuals':visuals, 'collisions':collisions}

    def _get_joint_transform(self, q_i, axis, joint_type):
        if joint_type=='revolute':
            T = np.eye(4)
            T[:3,:3] = Rot.from_rotvec(np.array(axis)*q_i).as_matrix()
        else:
            raise NotImplementedError(f'Joint type {joint_type} is not implemented.')
        return T
    
    def computeForwardKinematics(self, q):
        if not self.floating_base:
            assert len(q)==len(self.actuated_joints), 'The size of q does not match the number of actuated joints'
            world_T_root = np.eye(4)
            q_ = q
        else:
            assert len(q)==len(self.actuated_joints)+7, 'For a floating base robot, the size of q should be 7+ number of actuators'
            R = Rot.from_quat(q[3:7]).as_matrix()
            t = q[0:3].reshape(3,1)
            world_T_root = np.hstack([R, t])
            world_T_root = np.vstack([world_T_root, np.array([0., 0., 0., 1.])])
            q_ = q[7:]
        # Initialize all link pose flags to false
        for link in self.links_info.values():
            link['link_pose_valid']=False
        # Compute the forward kinematics
        for path in self.kinematic_paths.values():
            for i in range(len(path)-1):
                parent_link = path[i]
                child_link = path[i+1]
                joint_key = f'{parent_link}-{child_link}'
                if self.links_info[child_link]['link_pose_valid']==False:
                    T0 = self.links_info[parent_link]['link_pose']
                    joint_info = self.joints_info[joint_key]
                    joint_offset = joint_info['joint_offset']
                    link_offset = self.links_info[child_link]['link_offset']
                    if joint_info['joint_type']=='revolute':
                        q_i = q_[self.actuated_joints[joint_key]]
                        axis = joint_info['axis']
                        joint_type = joint_info['joint_type']
                        joint_transform = self._get_joint_transform(q_i, axis, joint_type)
                        self.joints_info[joint_key]['joint_transform'] = joint_transform
                        self.joints_info[joint_key]['joint_state'] = q_i
                        T = T0@link_offset@joint_offset@joint_transform
                    elif joint_info['joint_type']=='fixed':
                        T = T0@link_offset@joint_offset
                    else:
                        raise NotImplemented('Joint type is not implemented.')
                    
                    self.links_info[child_link]['link_pose']= world_T_root@T
                    self.links_info[child_link]['link_pose_valid'] = True
        self.q = q

    def computeJacobian(self, J_frame):      
        if self.q is None:
            raise Exception('Compute the forward kinematics before Jacobian')  
        J0_T_ef = self.links_info[J_frame]['link_pose']
        J = np.zeros((6, len(self.actuated_joints.keys())))
        # Compute the Jacobian of the robot with fixed root frame
        for i, pair in enumerate(self.actuated_joints.keys()):
            c = pair.split('-')[1]
            p = pair.split('-')[0]
            T_parent = self.links_info[p]['link_pose']
            T_child = self.links_info[c]['link_pose']
            # Ji frame is the child frame
            J0_T_Ji = T_child
            # get the pose of the EF frame with respect to the Ji frame
            Ji_T_ef = np.linalg.inv(J0_T_Ji)@J0_T_ef
            # compute the twist in joint frame
            ax = np.array(self.joints_info[pair]['axis']).reshape(3,1)
            joint_type = self.joints_info[pair]['joint_type']
            xi = self._get_joint_twist(ax, joint_type)
            Ji_hat = J0_T_Ji@xi@Ji_T_ef
            R_dot = Ji_hat[:3, :3]
            R = J0_T_ef[:3, :3]
            J[:3,i] = Ji_hat[:3, -1]
            J[3:,i] = veeSO3(R_dot@R.T).squeeze()

        # If the root frame is free in the world, extend the fixed root Jacobian
        if self.floating_base:
            world_R_base = Rot.from_quat(self.q[3:7]).as_matrix()
            world_t_base = self.q[:3]
            Jv = J[:3,:]
            Jw = J[3:,:]
            J_fb = np.zeros((6, 6+len(self.actuated_joints.keys())))
            J_fb[:3, :3] = np.eye(3)
            J_fb[:3, 3:6] = - world_R_base@hatSO3(world_t_base)
            J_fb[:3, 6:] =  world_R_base@Jv
            J_fb[3:, :3] = np.zeros((3,3))
            J_fb[3:, 3:6] = world_R_base
            J_fb[3:, 6:] = world_R_base@Jw
            self.q = None
            return J_fb
        else:
            self.q = None
            return J
                        
