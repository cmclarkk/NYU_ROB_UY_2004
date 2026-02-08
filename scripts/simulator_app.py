import sys
import argparse
import zerorpc
import signal
from rob2004.sim import MujocoRobot

def main():
    
    parser = argparse.ArgumentParser(description="ZeroRPC Popper Mujoco simulator and visualizer.")
    parser.add_argument("--visualizer", action="store_true", help="Visualizer mode (Physics disabled)")
    parser.add_argument("--fixed_body",  action="store_true", help="Fix the robot's body in the world.")
    parser.add_argument('--port', type=int, default=4242, help="The network port on which the simulator server listens")
    parser.add_argument('--ip', type=str, default="0.0.0.0", help="The IP on which the simulator server listens")
    args = parser.parse_args()

    def signal_handler(sig, frame):
        """
        Custom handler function called when SIGINT is received.
        """
        print("\nCtrl+C detected. Closing the simulator ...")
        # Your cleanup code here
        robot.close()
        sys.exit(0) # Exit the program after cleanup

    port = args.port
    ip = args.ip
    fixed_body=args.fixed_body
    visualizer_mode=args.visualizer

    robot = MujocoRobot(fixed_body=fixed_body, visualizer_mode=visualizer_mode)

    # Create a robot_server instance, exposing the RobotSim class
    robot_server = zerorpc.Server(robot)
    # Bind the robot_server to a TCP address (listen on all interfaces, port 4242)
    robot_server.bind(f"tcp://{ip}:{port}")
    print(f"Simulator is running on port {port}. Use RobotSim(port={port}) to communicate with the robot!")

    # Register the custom signal handler
    signal.signal(signal.SIGINT, signal_handler)
    robot_server.run()

if __name__=="__main__":
    main()
