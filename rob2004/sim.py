import time
import mujoco
import mujoco.viewer
import numpy as np
import os
from rob2004 import ASSETS_PATH
import threading

def add_visual_capsule(scene, point1, point2, radius, rgba):
  """Adds one capsule to an mjvScene."""
  if scene.ngeom >= scene.maxgeom:
    return
  scene.ngeom += 1  # increment ngeom
  # initialise a new capsule, add it to the scene using mjv_connector
  mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
                      mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                      np.zeros(3), np.zeros(9), rgba.astype(np.float32))
  mujoco.mjv_connector(scene.geoms[scene.ngeom-1],
                       mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                       point1, point2)

class MujocoRobot:
    def __init__(self, 
                 render=True, 
                 dt=0.001, 
                 xml_path=None,
                 fixed_body=True, # The robot's base is fixed in the world frame, control only for the lerg joints
                 visualizer_mode=False,
                 ):

        self.fixed = fixed_body
        self.visualizer_mode = visualizer_mode
        if xml_path is None:
            if self.fixed:
                self.model = mujoco.MjModel.from_xml_path(
                    os.path.join(ASSETS_PATH, 'mujoco/pupper_v3_complete.fixed.xml')
                )
                self.q0 = np.zeros(12)
            else:
                self.q0 = np.zeros(19)
                self.q0[6]=1.
                self.q0[2]=0.35
                self.model = mujoco.MjModel.from_xml_path(
                    os.path.join(ASSETS_PATH, 'mujoco/pupper_v3_complete.xml')
                )
        else:
            self.model = mujoco.MjModel.from_xml_path(xml_path)

        self.simulated = True
        self.data = mujoco.MjData(self.model)
        self.dt = dt
        _render_dt = 1 / 60
        self.render_ds_ratio = max(1, _render_dt // dt)

        if render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.render = True
            self.viewer.cam.distance = 3.0
            self.viewer.cam.azimuth = 90
            self.viewer.cam.elevation = -45
            self.viewer.cam.lookat[:] = np.array([0.0, -0.25, 0.824])
        else:
            self.render = False

        self.model.opt.gravity[2] = -9.81
        self.model.opt.timestep = dt
        self.renderer = None
        self.render = render
        self.step_counter = 0

        self.reset()
        mujoco.mj_step(self.model, self.data)
        if self.render:
            self.viewer.sync()
        self.nv = self.model.nv
        self.tau_ff = np.zeros(12)
        self.latest_command_stamp = time.time()
        self.scene = self.viewer.user_scn
        self.running = True
        self.sim_thread = threading.Thread(target=self.sim_thread)
        self.sim_thread.start()

    def sim_thread(self):
        while self.running:
            # Reset the torque commands if the last command sent by the user is too old (older than 0.1 seconds)
            tic = time.time()
            if time.time()-self.latest_command_stamp > 0.1: 
                self.tau_ff[:] = 0.
            self.step()
            while time.time()-tic < self.dt:
                time.sleep(2e-4)

    def reset(self, q0=None):
        if q0 is not None:
            q0 = np.array(q0)
            assert q0.shape==(12,), 'Wrong q0 shape! The shape should be (3,)'
            self.data.qpos[:]=q0
        else:
            self.data.qpos[:] = self.q0
        mujoco.mj_step(self.model, self.data)
        if self.render:
            self.viewer.sync()

    def getJointState(self):
        if self.fixed:
            return dict(q=self.data.qpos[:12].tolist(), dq=self.data.qvel[:12].tolist())
        else:
            return dict(q=self.data.qpos[7:].tolist(), dq=self.data.qvel[6:].tolist())

    def setJointCommand(self, torque):
        assert np.array(torque).shape == (12,), 'Wrong torque shape! The torque commnand should be a numpy array with shape (12,)'
        self.latest_command_stamp = time.time()
        self.tau_ff[:] = np.array(torque)

    def step(self):
        self.data.ctrl[:] = self.tau_ff
        self.step_counter += 1
        if not self.visualizer_mode:
            mujoco.mj_step(self.model, self.data)
        # Render every render_ds_ratio steps (60Hz GUI update)
        if self.render and (self.step_counter % self.render_ds_ratio) == 0:
            self.viewer.sync()

    def show(self, q):
        assert self.fixed, 'Visualization can only work for fxied body robots!'
        assert self.visualizer_mode, 'show only works in visualization model!'
        assert np.array(q).shape == (12,), 'Wrong joint_position shape! The q must be a 12D list of floating numbers'
        self.data.qpos[:] = np.array(q)
        self.data.qvel[:] = 0.
        mujoco.mj_step(self.model, self.data)

    def close(self):
        if self.render:
            self.viewer.close()
        self.running = False
        self.sim_thread.join()

    def add_visual_ball(self, ball_pos, color=np.array([0, 1, 0, 1]), radius=0.01):
        pos = np.array([ball_pos[0]-0.3-0.06, ball_pos[2], ball_pos[1]+0.29])
        add_visual_capsule(self.scene, pos, pos+np.array([0., 0., 0.001]), radius, color)
