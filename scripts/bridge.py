import sys
import time
import argparse
import signal
import zerorpc

from rob2004.ros2_robot import Ros2Robot


def main(args):
    robot = None
    server = None

    def _safe_shutdown():
        nonlocal robot, server
        try:
            if robot is not None:
                # Best-effort: send zeros twice
                try:
                    robot.setJointCommand([0.0] * len(robot.joint_order))
                    time.sleep(0.05)
                    robot.setJointCommand([0.0] * len(robot.joint_order))
                    time.sleep(0.05)
                except Exception:
                    pass
                robot.close()
        finally:
            # zerorpc server will stop when process exits; this is just in case
            if server is not None:
                try:
                    server.close()
                except Exception:
                    pass

    def signal_handler(sig, frame):
        print("\nCtrl+C detected. Closing ROS robot server ...")
        _safe_shutdown()
        sys.exit(0)

    # Create ROS-backed robot instance (expects ros2_control stack already launched)
    robot = Ros2Robot(
        controllers_yaml_path=args.controllers_yaml,
        loop_hz=args.loop_hz,
        command_timeout_s=args.timeout,
        node_name=args.node_name,
        joint_states_topic=args.joint_states_topic,
        imu_topic=args.imu_topic if args.imu_topic else None,
    )

    # Expose robot methods via ZeroRPC
    server = zerorpc.Server(robot)
    server.bind(f"tcp://{args.ip}:{args.port}")
    print(
        f"ROS Robot ZeroRPC server running on tcp://{args.ip}:{args.port}\n"
        f"- controllers_yaml: {args.controllers_yaml}\n"
        f"- joints: {len(robot.joint_order)}\n"
        f"Use zerorpc.Client().connect(...) to communicate."
    )
    breakpoint()
    signal.signal(signal.SIGINT, signal_handler)
    server.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZeroRPC server wrapping Ros2Robot (ros2_control client).")
    parser.add_argument("--controllers_yaml", type=str, default="robot_config.yaml",
                        help="Path to the ros2_control controllers YAML (same one used in launch).")

    parser.add_argument("--port", type=int, default=4242,
                        help="Network port for the ZeroRPC server.")
    parser.add_argument("--ip", type=str, default="0.0.0.0",
                        help="IP address to bind the ZeroRPC server.")

    parser.add_argument("--loop_hz", type=float, default=400.0,
                        help="Publish loop rate for commands (Hz).")
    parser.add_argument("--timeout", type=float, default=0.10,
                        help="Command deadman timeout (seconds).")

    parser.add_argument("--node_name", type=str, default="popper_bridge",
                        help="ROS node name used by the wrapper.")
    parser.add_argument("--joint_states_topic", type=str, default="/joint_states",
                        help="JointState topic (from joint_state_broadcaster).")
    parser.add_argument("--imu_topic", type=str, default="/imu_sensor_broadcaster/imu",
                        help="Optional IMU topic. Leave empty to disable.")

    args = parser.parse_args()
    main(args)
