import time
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

try:
    import yaml
except ImportError as e:
    raise RuntimeError("Missing dependency: python3-yaml (PyYAML). Install e.g. `sudo apt install python3-yaml`.") from e


@dataclass(frozen=True)
class JointControllerInfo:
    controller_name: str          # e.g. "forward_command_controller_leg_front_l_3"
    joint_name: str               # e.g. "leg_front_l_3"
    interface_names: Tuple[str]   # e.g. ("effort", "kp", "kd")


def _load_multi_iface_forward_controllers(controllers_yaml_path: str) -> Tuple[List[str], Dict[str, JointControllerInfo]]:
    """
    Returns:
      joint_order: list of joint names (from joint_state_broadcaster if present, else discovered joints)
      joint_to_ctrl: joint_name -> JointControllerInfo
    """
    with open(controllers_yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    cm_params = cfg.get("controller_manager", {}).get("ros__parameters", {})
    if not isinstance(cm_params, dict):
        raise ValueError("controllers YAML: controller_manager.ros__parameters not found or not a dict")

    # 1) Joint order (nice to match your robot ordering)
    joint_order: List[str] = []
    jsb = cfg.get("joint_state_broadcaster", {}).get("ros__parameters", {})
    if isinstance(jsb, dict) and "joints" in jsb:
        joint_order = list(jsb["joints"])

    # 2) Find controllers with a `type`
    typed_controllers = {
        name: spec.get("type")
        for name, spec in cm_params.items()
        if isinstance(spec, dict) and "type" in spec
    }

    # 3) For each MultiInterfaceForwardCommandController, read its per-controller section
    joint_to_ctrl: Dict[str, JointControllerInfo] = {}

    for cname, ctype in typed_controllers.items():
        if ctype != "forward_command_controller/MultiInterfaceForwardCommandController":
            continue

        params = cfg.get(cname, {}).get("ros__parameters", {})
        if not isinstance(params, dict):
            raise ValueError(f"controllers YAML: missing '{cname}: ros__parameters:' block")

        joint = params.get("joint", None)  # for MultiInterfaceForwardCommandController it's singular "joint" :contentReference[oaicite:2]{index=2}
        iface_names = params.get("interface_names", None)

        if not isinstance(joint, str) or not joint:
            raise ValueError(f"controllers YAML: controller '{cname}' missing valid 'joint' parameter")

        if not (isinstance(iface_names, list) and all(isinstance(x, str) for x in iface_names) and len(iface_names) > 0):
            raise ValueError(f"controllers YAML: controller '{cname}' missing valid 'interface_names' list")

        info = JointControllerInfo(
            controller_name=cname,
            joint_name=joint,
            interface_names=tuple(iface_names),
        )
        if joint in joint_to_ctrl:
            raise ValueError(f"Duplicate control mapping: joint '{joint}' appears in multiple controllers")
        joint_to_ctrl[joint] = info

    if not joint_to_ctrl:
        raise ValueError(
            "No forward_command_controller/MultiInterfaceForwardCommandController found in YAML. "
            "Check controller types / YAML path."
        )

    # If joint_order wasn’t specified, make a stable order from discovered joints
    if not joint_order:
        joint_order = sorted(list(joint_to_ctrl.keys()))

    return joint_order, joint_to_ctrl


class Ros2Robot:
    """
    A MuJoCo-like wrapper for a ros2_control robot driven by MultiInterfaceForwardCommandController(s).
    """
    def __init__(
        self,
        controllers_yaml_path: str,
        loop_hz: float = 400.0,
        command_timeout_s: float = 0.10,
        node_name: str = "ros2_robot_api",
        joint_states_topic: str = "/joint_states",
        # IMU topic varies by setup; common pattern is /<controller_name>/imu. See `ros2 topic list`.
        imu_topic: Optional[str] = "/imu_sensor_broadcaster/imu",
    ):
        self.controllers_yaml_path = controllers_yaml_path
        self.loop_hz = float(loop_hz)
        self.dt = 1.0 / self.loop_hz
        self.command_timeout_s = float(command_timeout_s)

        # Discover controllers/joints from YAML
        self.joint_order, self.joint_to_ctrl = _load_multi_iface_forward_controllers(controllers_yaml_path)
        self.nj = len(self.joint_order)

        # Commands (effort/kp/kd default)
        self._latest_command_stamp = 0.0
        self._cmd_effort = np.zeros(self.nj, dtype=np.float64)
        self._cmd_kp = np.zeros(self.nj, dtype=np.float64)
        self._cmd_kd = np.zeros(self.nj, dtype=np.float64)

        # State cache
        self._q = np.zeros(self.nj, dtype=np.float64)
        self._dq = np.zeros(self.nj, dtype=np.float64)
        self._have_state = False

        self._lock = threading.Lock()

        # ROS
        rclpy.init(args=None)
        self.node = Node(node_name)

        # Subscribers
        self.node.create_subscription(JointState, joint_states_topic, self._on_joint_state, 10)

        # Optional IMU
        self._imu_msg = None
        if imu_topic is not None:
            from sensor_msgs.msg import Imu
            self.node.create_subscription(Imu, imu_topic, self._on_imu, 10)

        # Publishers: one per controller (topic is ~/<controller_name>/commands) :contentReference[oaicite:3]{index=3}
        self._pub_by_controller: Dict[str, any] = {}
        for joint, info in self.joint_to_ctrl.items():
            topic = f"/{info.controller_name}/commands"
            self._pub_by_controller[info.controller_name] = self.node.create_publisher(Float64MultiArray, topic, 10)

        # Timer loop (publishes the *stored* command)
        self.node.create_timer(self.dt, self._publish_loop)

        # Spin in background thread
        self._running = True
        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()

    # -------- public API (MuJoCo-ish) --------
    def getJointState(self) -> Dict[str, List[float]]:
        with self._lock:
            return dict(q=self._q.tolist(), dq=self._dq.tolist())

    def getImu(self):
        """Returns latest sensor_msgs/Imu or None (if imu_topic not set / no message yet)."""
        with self._lock:
            if self._imu_msg is None:
                return None
            else:
                qut = self._imu_msg.orientation
                q = [qut.x, qut.y, qut.z, qut.w]
                ang_vel = [self._imu_msg.angular_velocity.x, self._imu_msg.angular_velocity.y, self._imu_msg.angular_velocity.z]
                lin_acc = [self._imu_msg.linear_acceleration.x, self._imu_msg.linear_acceleration.y, self._imu_msg.linear_acceleration.z]
                imu_dict = {
                    "orientation": q,
                    "angular_velocity": ang_vel,
                    "linear_acceleration": lin_acc,
                }
                return imu_dict

    def setJointCommand(
        self,
        effort: Sequence[float],
        kp: Optional[Sequence[float]] = None,
        kd: Optional[Sequence[float]] = None,
    ):
        effort = np.asarray(effort, dtype=np.float64)
        if effort.shape != (self.nj,):
            raise ValueError(f"effort must have shape ({self.nj},) in joint_order={self.joint_order}")

        with self._lock:
            self._cmd_effort[:] = effort
            if kp is not None:
                kp = np.asarray(kp, dtype=np.float64)
                if kp.shape != (self.nj,):
                    raise ValueError(f"kp must have shape ({self.nj},)")
                self._cmd_kp[:] = kp
            else:
                self._cmd_kp[:] = 0.0

            if kd is not None:
                kd = np.asarray(kd, dtype=np.float64)
                if kd.shape != (self.nj,):
                    raise ValueError(f"kd must have shape ({self.nj},)")
                self._cmd_kd[:] = kd
            else:
                self._cmd_kd[:] = 0.0

            self._latest_command_stamp = time.time()

    def close(self):
        # Send zeros once before shutdown
        self.setJointCommand(np.zeros(self.nj))
        time.sleep(0.05)

        self._running = False
        self.node.destroy_node()
        rclpy.shutdown()
        self._spin_thread.join(timeout=1.0)

    # -------- ROS callbacks / loop --------
    def _spin(self):
        while self._running and rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.05)

    def _on_joint_state(self, msg: JointState):
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        with self._lock:
            for j, joint_name in enumerate(self.joint_order):
                if joint_name not in name_to_idx:
                    continue
                i = name_to_idx[joint_name]
                if i < len(msg.position):
                    self._q[j] = msg.position[i]
                if i < len(msg.velocity):
                    self._dq[j] = msg.velocity[i]
            self._have_state = True

    def _on_imu(self, msg):
        with self._lock:
            self._imu_msg = msg

    def _publish_loop(self):
        now = time.time()
        with self._lock:
            stale = (now - self._latest_command_stamp) > self.command_timeout_s
            eff = np.zeros_like(self._cmd_effort) if stale else self._cmd_effort.copy()
            kp = np.zeros_like(self._cmd_kp) if stale else self._cmd_kp.copy()
            kd = np.zeros_like(self._cmd_kd) if stale else self._cmd_kd.copy()

        # Publish to each joint’s controller.
        # For your interface_names=['effort','kp','kd'], we publish [effort, kp, kd] :contentReference[oaicite:4]{index=4}
        for j, joint_name in enumerate(self.joint_order):
            info = self.joint_to_ctrl.get(joint_name, None)
            if info is None:
                # Joint exists in joint_state_broadcaster but has no controller mapped (OK if you want passive joints)
                continue

            vec = []
            for iface in info.interface_names:
                if iface == "effort":
                    vec.append(float(eff[j]))
                elif iface == "kp":
                    vec.append(float(kp[j]))
                elif iface == "kd":
                    vec.append(float(kd[j]))
                else:
                    # Unknown interface -> publish 0.0 by default
                    vec.append(0.0)

            msg = Float64MultiArray()
            msg.data = vec
            self._pub_by_controller[info.controller_name].publish(msg)
