# lab_complete.launch.py

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

import os

def _spawn_all_controllers(context, *args, **kwargs):
    """
    Parse the controllers YAML and create spawner nodes for every controller
    listed under controller_manager.ros__parameters that has a 'type' field.
    """
    controllers_file = LaunchConfiguration("controllers_file").perform(context)

    try:
        import yaml
    except ImportError as e:
        raise RuntimeError(
            "PyYAML is required to parse the controllers file. "
            "Install it (e.g., `sudo apt install python3-yaml`)."
        ) from e

    with open(controllers_file, "r") as f:
        cfg = yaml.safe_load(f)

    cm_params = (
        cfg.get("controller_manager", {})
           .get("ros__parameters", {})
    )

    # Controllers are entries with a 'type' field.
    controller_names = [
        name for name, spec in cm_params.items()
        if isinstance(spec, dict) and "type" in spec
    ]

    # We will spawn JSB explicitly first; optionally spawn IMU explicitly too.
    explicitly_spawned_first = {"joint_state_broadcaster", "imu_sensor_broadcaster"}
    remaining = [c for c in controller_names if c not in explicitly_spawned_first]

    # Spawner nodes for the remaining controllers
    spawners = []
    for cname in remaining:
        spawners.append(
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=[
                    cname,
                    "--controller-manager", "/controller_manager",
                    "--controller-manager-timeout", "30",
                ],
                output="screen",
            )
        )

    return spawners


def generate_launch_description():
    # --- URDF via xacro ---
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [FindPackageShare("pupper_v3_description"), "description", "pupper_v3.urdf.xacro"]
            ),
        ]
    )
    robot_description = {"robot_description": robot_description_content}

    # --- Controllers YAML ---
    default_controllers_file = PathJoinSubstitution(
        [os.path.dirname(__file__), "robot_config.yaml"]  # <- rename to your full config file
    )

    controllers_file_arg = DeclareLaunchArgument(
        "controllers_file",
        default_value=default_controllers_file,
        description="Path to ros2_control controllers YAML",
    )

    control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[robot_description, LaunchConfiguration("controllers_file")],
        output="both",
    )

    robot_state_pub_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description],
    )

    # --- Spawn broadcasters first ---
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager", "/controller_manager",
            "--controller-manager-timeout", "30",
        ],
        output="screen",
    )

    imu_sensor_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "imu_sensor_broadcaster",
            "--controller-manager", "/controller_manager",
            "--controller-manager-timeout", "30",
        ],
        output="screen",
    )

    # After joint_state_broadcaster spawns, spawn everything else in the YAML
    spawn_rest_after_jsb = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[OpaqueFunction(function=_spawn_all_controllers)],
        )
    )

    return LaunchDescription(
        [
            controllers_file_arg,
            control_node,
            robot_state_pub_node,
            joint_state_broadcaster_spawner,
            imu_sensor_broadcaster_spawner,
            spawn_rest_after_jsb,
        ]
    )
