import zerorpc
import numpy as np

class RobotInterfaceSim:
    def __init__(self, ip='127.0.0.1', port=4242):
        self.robot = zerorpc.Client()
        # Connect to the server's address
        self.robot.connect(f"tcp://{ip}:{port}")

    def getJointState(self):
        state = self.robot.getJointState()
        if state is not None:
            return {k:np.array(v) for k,v in state.items()}
        else:
            return None
        
    def setJointCommand(self, tau):
        if isinstance(tau, np.ndarray):
            assert tau.shape==(12,), 'The shape of the torque vector tau should be (12,)!'
            tau_ = tau.tolist()
        elif isinstance(tau, list):
            assert len(tau)==12, 'The length of the tau list should be 12!'
            tau_ = tau
        else:
            raise TypeError('Tau can only be a numpy array or python list with length 12')
        
        self.robot.setJointCommand(tau_)


class RobotInterfaceReal:
    def __init__(self, ip, port=4243):
        self.robot = zerorpc.Client()
        # Connect to the server's address
        self.robot.connect(f"tcp://{ip}:{port}")

    def getJointState(self):
        state = self.robot.getJointState()
        if state is not None:
            return {k:np.array(v) for k,v in state.items()}
        else:
            return None
        
    def setJointCommand(self, tau, q_des=None, dq_des=None, kp=None, kd=None):
        
        if isinstance(tau, np.ndarray):
            assert tau.shape==(12,), 'The shape of the torque vector tau should be (12,)!'
            tau_ = tau.tolist()
        elif isinstance(tau, list):
            assert len(tau)==12, 'The length of the tau list should be 12!'
            tau_ = tau
        else:
            raise TypeError('Tau can only be a numpy array or python list with length 12')
        
        if isinstance(kp, np.ndarray):
            assert kp.shape==(12,), 'The shape of the KP gain vector should be (12,)!'
            kp_ = kp.tolist()
        elif isinstance(kp, list):
            assert len(kp)==12, 'The length of the kp list should be 12!'
            kp_ = kp
        else:
            kp_ = None

        if isinstance(kd, np.ndarray):
            assert kd.shape==(12,), 'The shape of the kd vector should be (12,)!'
            kd_ = kd.tolist()
        elif isinstance(kd, list):
            assert len(kd)==12, 'The length of the kd list should be 12!'
            kd_ = kd
        else:
            kd_ = None

        if isinstance(q_des, np.ndarray):
            assert q_des.shape==(12,), 'The shape of the q_des vector should be (12,)!'
            q_des_ = q_des.tolist()
        elif isinstance(q_des, list):
            assert len(q_des)==12, 'The length of the q_des list should be 12!'
            q_des_ = q_des
        else:
            q_des_ = None

        if isinstance(dq_des, np.ndarray):
            assert dq_des.shape==(12,), 'The shape of the dq_des vector should be (12,)!'
            dq_des_ = dq_des.tolist()
        elif isinstance(dq_des, list):
            assert len(dq_des)==12, 'The length of the dq_des list should be 12!'
            dq_des_ = dq_des
        else:
            dq_des_ = None

        
        self.robot.setJointCommand(tau_, q_des_, dq_des_, kp_, kd_)