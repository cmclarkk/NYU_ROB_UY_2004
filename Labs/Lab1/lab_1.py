# Andrea Gonzalez, Pingsheng Zhao, Marshal Yuan

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np
import time
from collections import deque
import signal

JOINT_NAME = "leg_front_l_3" 
JOINT_NAME_LEAD = "leg_front_r_3" 

####
####
KP = 1.6 # YOUR KP VALUE
KI = 0.05 # YOUR KI VALUE
KD = 0.09 # YOUR KD VALUE
####
####
LOOP_RATE = 200  # Hz
DELTA_T = 1 / LOOP_RATE
MAX_TORQUE = 2.0
DEAD_BAND_SIZE = 0.095
PENDULUM_CONTROL = False
LEG_TRACKING_CONTROL = not PENDULUM_CONTROL


class JointStateSubscriber(Node):
    direction = 0

    def __init__(self):
        super().__init__("joint_state_subscriber")
        # Create a subscriber to the /joint_states topic
        self.subscription = self.create_subscription(
            JointState, "/joint_states", self.get_joint_info, 10  # QoS profile history depth
        )
        self.subscription  # prevent unused variable warning

        # Publisher to the /forward_command_controller/commands topic
        self.command_publisher = self.create_publisher(Float64MultiArray, "/forward_command_controller/commands", 10)
        self.command_publisher_lead = self.create_publisher(Float64MultiArray, "/forward_command_controller_lead/commands", 10)
        self.print_counter = 0
        self.calculated_torque = 0
        self.joint_pos = 0
        self.joint_vel = 0
        self.joint_pos_lead = 0
        self.joint_vel_lead = 0
        self.target_joint_pos = 0
        self.target_joint_vel = 0
        # self.torque_history = deque(maxlen=DELAY)
        self.last_joint_error = 0
        self.sum_joint_error = 0

        # Create a timer to run control_loop at the specified frequency
        self.create_timer(1.0 / LOOP_RATE, self.control_loop)

    def get_target_joint_info(self):
        target_joint_pos = -self.joint_pos_lead
        target_joint_vel = 0
        
        return target_joint_pos, target_joint_vel

    def calculate_torque_for_pendulum_control(self, joint_pos):
        if joint_pos > 0.20:
            self.direction = -1
        if joint_pos < -0.10:
            self.direction = 1
        torque = self.direction * 0.2

        return torque

    def calculate_torque_for_leg_tracking(self, joint_pos, joint_vel, target_joint_pos, target_joint_vel, sum_joint_error):
        ####
        #### YOUR CODE HERE
        ####

        sum_joint_error += (target_joint_pos-joint_pos)


        torque = 0
        tq= (target_joint_pos-joint_pos)*KP
        tq2 = (target_joint_pos-joint_pos)*KP + (target_joint_vel - joint_vel)*KD
        tq3 = (target_joint_pos-joint_pos)*KP + (target_joint_vel - joint_vel)*KD + (sum_joint_error)*KI

        torque=tq3

        # Leave this code unchanged
        if torque > 0:
            torque = torque+DEAD_BAND_SIZE
        elif torque < 0:
            torque = torque-DEAD_BAND_SIZE
        
        return torque

    def print_info(self):
        """Print joint information every 2 control loops"""
            
        if self.print_counter == 0:
            self.get_logger().info(
                f"Pos: {self.joint_pos:.2f}, Target Pos: {self.target_joint_pos:.2f}, Tor: {self.calculated_torque:.2f}"
            )
        self.print_counter += 1
        self.print_counter %= 2

    def get_joint_info(self, msg):
        """Callback function to process incoming JointState messages"""
        joint_index = msg.name.index(JOINT_NAME)
        joint_pos = msg.position[joint_index]
        joint_vel = msg.velocity[joint_index]

        joint_index_lead = msg.name.index(JOINT_NAME_LEAD)
        joint_pos_lead = msg.position[joint_index_lead]
        joint_vel_lead = msg.velocity[joint_index_lead]
        
        self.joint_pos = joint_pos
        self.joint_vel = joint_vel        
        self.joint_pos_lead = joint_pos_lead
        self.joint_vel_lead = joint_vel_lead

        return joint_pos, joint_vel, joint_pos_lead, joint_vel_lead

    def control_loop(self):
        """Control control loop to calculate and publish torque commands"""
        if PENDULUM_CONTROL:
            self.calculated_torque = self.calculate_torque_for_pendulum_control(self.joint_pos)
        elif LEG_TRACKING_CONTROL: 
            self.target_joint_pos, self.target_joint_vel = self.get_target_joint_info()
            self.calculated_torque = self.calculate_torque_for_leg_tracking(self.joint_pos, self.joint_vel, self.target_joint_pos, self.target_joint_vel, self.sum_joint_error)
        else:
            self.calculated_torque = 0
            
        self.print_info()
        self.publish_torque(self.calculated_torque)

    def publish_torque(self, torque=0.0):
        # Create a Float64MultiArray message with zero kp and kd values
        command_msg = Float64MultiArray()
        torque = np.clip(torque, -MAX_TORQUE, MAX_TORQUE)
        command_msg.data = [torque, 0.0, 0.0]#[torque, 0.0, 0.0]  # Zero kp and kd values

        command_msg_lead = Float64MultiArray()
        command_msg_lead.data = [0.0, 0.0, 0.0]

        # Publish the message
        self.command_publisher.publish(command_msg)
        self.command_publisher_lead.publish(command_msg_lead)


def main(args=None):
    rclpy.init(args=args)

    # Create the node
    joint_state_subscriber = JointStateSubscriber()

    # Install a SIGINT handler that sends zero torque BEFORE shutting down the context
    def _handle_sigint(sig, frame):
        joint_state_subscriber.get_logger().info("SIGINT received: sending zero torque and shutting down...")
        joint_state_subscriber.publish_torque(0.0)
        time.sleep(0.1) 
        joint_state_subscriber.publish_torque(0.0)
        rclpy.shutdown()

    signal.signal(signal.SIGINT, _handle_sigint)

    rclpy.spin(joint_state_subscriber)
  

if __name__ == "__main__":
    main()
