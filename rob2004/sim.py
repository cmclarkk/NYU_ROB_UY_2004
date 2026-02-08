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

# class MujocoRobot:
#     def __init__(self, 
#                  render=True, 
#                  dt=0.001, 
#                  xml_path=None,
#                  fixed_body=True, # The robot's base is fixed in the world frame, control only for the lerg joints
#                  visualizer_mode=False,
#                  ):

#         self.fixed = fixed_body
#         self.visualizer_mode = visualizer_mode
#         if xml_path is None:
#             if self.fixed:
#                 self.model = mujoco.MjModel.from_xml_path(
#                     os.path.join(ASSETS_PATH, 'mujoco/pupper_v3_complete.fixed.xml')
#                 )
#                 self.q0 = np.zeros(12)
#             else:
#                 self.q0 = np.zeros(19)
#                 self.q0[6]=1.
#                 self.q0[2]=0.35
#                 self.model = mujoco.MjModel.from_xml_path(
#                     os.path.join(ASSETS_PATH, 'mujoco/pupper_v3_complete.xml')
#                 )
#         else:
#             self.model = mujoco.MjModel.from_xml_path(xml_path)

#         self.simulated = True
#         self.data = mujoco.MjData(self.model)
#         self.dt = dt
#         _render_dt = 1 / 60
#         self.render_ds_ratio = max(1, _render_dt // dt)

#         if render:
#             self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
#             self.render = True
#             self.viewer.cam.distance = 3.0
#             self.viewer.cam.azimuth = 90
#             self.viewer.cam.elevation = -45
#             self.viewer.cam.lookat[:] = np.array([0.0, -0.25, 0.824])
#         else:
#             self.render = False

#         self.model.opt.gravity[2] = -9.81
#         self.model.opt.timestep = dt
#         self.renderer = None
#         self.render = render
#         self.step_counter = 0

#         self.reset()
#         mujoco.mj_step(self.model, self.data)
#         if self.render:
#             self.viewer.sync()
#         self.nv = self.model.nv
#         self.tau_ff = np.zeros(12)
#         self.latest_command_stamp = time.time()
#         self.scene = self.viewer.user_scn
#         self.running = True
#         self.sim_thread = threading.Thread(target=self.sim_thread)
#         self.sim_thread.start()

#     def sim_thread(self):
#         while self.running:
#             # Reset the torque commands if the last command sent by the user is too old (older than 0.1 seconds)
#             tic = time.time()
#             if time.time()-self.latest_command_stamp > 0.1: 
#                 self.tau_ff[:] = 0.
#             self.step()
#             while time.time()-tic < self.dt:
#                 time.sleep(2e-4)

#     def reset(self, q0=None):
#         if q0 is not None:
#             q0 = np.array(q0)
#             assert q0.shape==(12,), 'Wrong q0 shape! The shape should be (3,)'
#             self.data.qpos[:]=q0
#         else:
#             self.data.qpos[:] = self.q0
#         mujoco.mj_step(self.model, self.data)
#         if self.render:
#             self.viewer.sync()

#     def getJointState(self):
#         if self.fixed:
#             return dict(q=self.data.qpos[:12].tolist(), dq=self.data.qvel[:12].tolist())
#         else:
#             return dict(q=self.data.qpos[7:].tolist(), dq=self.data.qvel[6:].tolist())

#     def setJointCommand(self, torque):
#         assert np.array(torque).shape == (12,), 'Wrong torque shape! The torque commnand should be a numpy array with shape (12,)'
#         self.latest_command_stamp = time.time()
#         self.tau_ff[:] = np.array(torque)

#     def step(self):
#         self.data.ctrl[:] = self.tau_ff
#         self.step_counter += 1
#         if not self.visualizer_mode:
#             mujoco.mj_step(self.model, self.data)
#         # Render every render_ds_ratio steps (60Hz GUI update)
#         if self.render and (self.step_counter % self.render_ds_ratio) == 0:
#             self.viewer.sync()

#     def show(self, q):
#         assert self.fixed, 'Visualization can only work for fxied body robots!'
#         assert self.visualizer_mode, 'show only works in visualization model!'
#         assert np.array(q).shape == (12,), 'Wrong joint_position shape! The q must be a 12D list of floating numbers'
#         self.data.qpos[:] = np.array(q)
#         self.data.qvel[:] = 0.
#         mujoco.mj_step(self.model, self.data)

#     def close(self):
#         if self.render:
#             self.viewer.close()
#         self.running = False
#         self.sim_thread.join()

#     def add_visual_ball(self, ball_pos, color=np.array([0, 1, 0, 1]), radius=0.01):
#         pos = np.array([ball_pos[0]-0.3-0.06, ball_pos[2], ball_pos[1]+0.29])
#         add_visual_capsule(self.scene, pos, pos+np.array([0., 0., 0.001]), radius, color)


import os
import time
import threading
import queue
import numpy as np
import mujoco
import mujoco.viewer

# you already have these somewhere
# from your_utils import add_visual_capsule, ASSETS_PATH

class MujocoRobot:
    def __init__(self,
                 render=True,
                 dt=0.001,
                 xml_path=None,
                 fixed_body=True,
                 visualizer_mode=False):
        self.fixed = fixed_body
        self.visualizer_mode = visualizer_mode

        # --- Load model ---
        if xml_path is None:
            if self.fixed:
                self.model = mujoco.MjModel.from_xml_path(
                    os.path.join(ASSETS_PATH, 'mujoco/pupper_v3_complete.fixed.xml')
                )
                self.q0 = np.zeros(12, dtype=np.float64)
            else:
                self.q0 = np.zeros(19, dtype=np.float64)
                self.q0[6] = 1.0
                self.q0[2] = 0.35
                self.model = mujoco.MjModel.from_xml_path(
                    os.path.join(ASSETS_PATH, 'mujoco/pupper_v3_complete.xml')
                )
        else:
            self.model = mujoco.MjModel.from_xml_path(xml_path)

        self.data = mujoco.MjData(self.model)

        self.dt = float(dt)
        _render_dt = 1.0 / 60.0
        # Make sure this is int and >= 1
        self.render_ds_ratio = max(1, int(round(_render_dt / self.dt)))

        self.model.opt.gravity[2] = -9.81
        self.model.opt.timestep = self.dt

        # --- Thread-safety primitives ---
        # Protects shared “command/state snapshot” variables.
        self._cmd_lock = threading.Lock()
        # Work queue to run arbitrary functions on the sim thread.
        self._work_q: "queue.Queue[tuple]" = queue.Queue()

        # Latest torque command (shared between threads via _cmd_lock)
        self.tau_ff = np.zeros(12, dtype=np.float64)
        self.latest_command_stamp = time.time()

        # Cached joint state snapshot, updated by sim thread, read by other threads.
        self._state_lock = threading.Lock()
        self._state_q = None
        self._state_dq = None
        self._state_stamp = 0.0

        # Render / viewer objects (ONLY used in sim thread)
        self.render = bool(render)
        self.viewer = None
        self.scene = None

        self.step_counter = 0
        self.running = True

        # Start sim thread
        self._thread = threading.Thread(target=self._sim_loop, name="MujocoRobotSim", daemon=True)
        self._thread.start()

        # Create viewer on sim thread (so viewer usage stays single-threaded)
        if self.render:
            self._call_in_sim(self._init_viewer, wait=True)

        # Reset on sim thread
        self._call_in_sim(lambda: self._reset_impl(None), wait=True)
        self.nv = self.model.nv

    # -------------------------
    # Public thread-safe API
    # -------------------------

    def getJointState(self):
        # Return a consistent snapshot without touching mujoco data from this thread.
        with self._state_lock:
            if self._state_q is None:
                # If called very early before first snapshot, return zeros of correct shape.
                if self.fixed:
                    q = np.zeros(12).tolist()
                    dq = np.zeros(12).tolist()
                else:
                    q = np.zeros(max(0, self.model.nq - 7)).tolist()
                    dq = np.zeros(max(0, self.model.nv - 6)).tolist()
                return dict(q=q, dq=dq, stamp=0.0)

            return dict(
                q=self._state_q.copy().tolist(),
                dq=self._state_dq.copy().tolist(),
                stamp=float(self._state_stamp),
            )

    def setJointCommand(self, torque):
        torque = np.asarray(torque, dtype=np.float64).reshape(-1)
        assert torque.shape == (12,), "Wrong torque shape! Must be (12,)"
        now = time.time()
        with self._cmd_lock:
            self.latest_command_stamp = now
            self.tau_ff[:] = torque

    def reset(self, q0=None, wait=True):
        # Schedule reset on sim thread
        return self._call_in_sim(lambda: self._reset_impl(q0), wait=wait)

    def show(self, q, wait=True):
        # Schedule show on sim thread
        return self._call_in_sim(lambda: self._show_impl(q), wait=wait)

    def add_visual_ball(self, ball_pos, color=[0, 1, 0, 1], radius=0.01, wait=False):
        color = np.asanyarray(color)
        # Schedule on sim thread (scene is not thread-safe)
        ball_pos = np.asarray(ball_pos, dtype=np.float64).reshape(3,)
        color = np.asarray(color, dtype=np.float64).reshape(4,)
        radius = float(radius)
        self._call_in_sim(lambda: self._add_visual_ball_impl(ball_pos, color, radius), wait=wait)
        return 
    
    def close(self):
        # Stop thread, then close viewer on sim thread before exit.
        if not self.running:
            return
        self.running = False
        # Unblock sim thread if it's waiting by pushing a no-op
        try:
            self._work_q.put_nowait((lambda: None, None))
        except Exception:
            pass
        self._thread.join()

        # viewer.close must happen after sim loop ends (viewer is owned by sim thread)
        # But viewer might already be closed; guard it.
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None

    # -------------------------
    # Sim thread internals
    # -------------------------

    def _call_in_sim(self, fn, wait: bool):
        """
        Run `fn()` on the sim thread.
        If wait=True, block until completion and return its result (or raise its exception).
        """
        if not self.running:
            raise RuntimeError("Simulator is not running.")

        if not wait:
            self._work_q.put((fn, None))
            return None

        ev = threading.Event()
        box = {"ok": False, "ret": None, "err": None}

        def wrapped():
            try:
                box["ret"] = fn()
                box["ok"] = True
            except Exception as e:
                box["err"] = e
            finally:
                ev.set()

        self._work_q.put((wrapped, ev))
        ev.wait()
        if box["err"] is not None:
            raise box["err"]
        return box["ret"]

    def _init_viewer(self):
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.cam.distance = 3.0
        self.viewer.cam.azimuth = 90
        self.viewer.cam.elevation = -45
        self.viewer.cam.lookat[:] = np.array([0.0, -0.25, 0.824], dtype=np.float64)
        self.scene = self.viewer.user_scn

    def _reset_impl(self, q0):
        if q0 is not None:
            q0 = np.asarray(q0, dtype=np.float64).reshape(-1)
            assert q0.shape == (12,), "Wrong q0 shape! The shape should be (12,)"
            self.data.qpos[:12] = q0
        else:
            self.data.qpos[:] = self.q0

        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)  # forward is enough for a reset pose
        if self.viewer is not None:
            self.viewer.sync()

    def _show_impl(self, q):
        assert self.fixed, "Visualization can only work for fixed body robots!"
        assert self.visualizer_mode, "show only works in visualization mode!"
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        assert q.shape == (12,), "q must be shape (12,)"

        self.data.qpos[:12] = q
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()

    def _add_visual_ball_impl(self, ball_pos, color, radius):
        if self.scene is None:
            return  # rendering disabled
        # Keep your existing coordinate conversion
        pos = np.array([ball_pos[0], ball_pos[2], ball_pos[1]], dtype=np.float64)
        add_visual_capsule(self.scene, pos, pos + np.array([0., 0., 0.001]), radius, color)

    def _update_state_snapshot(self):
        # Called on sim thread; copy out joint state into cached snapshot.
        if self.fixed:
            q = np.array(self.data.qpos[:12], dtype=np.float64)
            dq = np.array(self.data.qvel[:12], dtype=np.float64)
        else:
            q = np.array(self.data.qpos[7:], dtype=np.float64)
            dq = np.array(self.data.qvel[6:], dtype=np.float64)

        with self._state_lock:
            self._state_q = q
            self._state_dq = dq
            self._state_stamp = time.time()

    def _sim_loop(self):
        while self.running:
            tic = time.time()

            # 1) Execute queued work (reset/show/add_visual_ball/etc.) on this thread
            while True:
                try:
                    fn, ev = self._work_q.get_nowait()
                except queue.Empty:
                    break
                try:
                    fn()
                finally:
                    # ev is only used by wait=True wrapper; wrapper sets it itself.
                    pass

            # 2) Read latest command safely
            with self._cmd_lock:
                tau = self.tau_ff.copy()
                last_stamp = self.latest_command_stamp

            # Reset torque if stale
            if (time.time() - last_stamp) > 0.1:
                tau[:] = 0.0

            # 3) Step mujoco (ONLY here)
            self.data.ctrl[:] = tau
            self.step_counter += 1

            if not self.visualizer_mode:
                mujoco.mj_step(self.model, self.data)
            else:
                # In visualizer_mode, we typically don't advance time, but keep forward consistent:
                mujoco.mj_forward(self.model, self.data)

            # 4) Update snapshot for other threads
            self._update_state_snapshot()

            # 5) Render at ~60Hz
            if self.viewer is not None and (self.step_counter % self.render_ds_ratio) == 0:
                self.viewer.sync()

            # 6) Sleep to maintain dt
            while (time.time() - tic) < self.dt:
                time.sleep(2e-4)
