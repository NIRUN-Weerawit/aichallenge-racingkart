#!/usr/bin/env python3
"""Gymnasium env for PPO-driven 20-step MPC velocity profile control.

RL outputs [v_0, v_1, ..., v_19] — one target velocity per MPC prediction
horizon step. Published to /mpc/ref_vel_profile and consumed directly by the
MPC's _init_problem() loop.

Architecture (same dual-domain pattern as multiplier_env):
  - Domain 0: publishes to /admin/awsim/reset
  - Domain 1: subscribes to /awsim/status, /localization/kinematic_state
              publishes to /mpc/ref_vel_profile + /awsim/control_mode_request_topic

Observation (71 dims): [one-hot section(10) | speed(1) | curvature(20) |
                        width(20) | prev_vel_profile(20)]
Action: 20 absolute velocities in [v_min, v_max] m/s

Reset semantics:
  - env.reset() on boot: hard reset (AWSIM reset → initial pose → engage MPC)
  - env.reset() on TimeLimit: soft reset (state vars only, kart keeps running)
  - step() after collision: deferred AWSIM reset next cycle (same 3-step seq)
  - Stall detection: skipped during 5-tick grace period post-reset
"""

from collections import deque
import os
import threading
import time

import gymnasium as gym
import numpy as np
import rclpy
import rclpy.executors
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import Odometry
from rclpy.context import Context
from rclpy.node import Node
from std_srvs.srv import Trigger
from std_msgs.msg import Bool, Empty, Float32MultiArray


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HORIZON = 20  # MPC prediction horizon (matches config.yaml mpc.N)
_SECTIONS = 10
_DEFAULT_REF_PATH_CSV = os.path.join(
    get_package_share_directory("multi_purpose_mpc_ros"),
    "env/final_ver3/traj_mincurv.csv",
)


def _load_ref_path_data(csv_path: str):
    """Load x, y, psi, kappa from reference path CSV."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    return {
        "x": np.array(df["x_m"], dtype=np.float64),
        "y": np.array(df["y_m"], dtype=np.float64),
        "psi": np.array(df["psi_rad"], dtype=np.float64),
        "kappa": np.array(df["kappa_radpm"], dtype=np.float64),
    }


class _TickGate:
    """Thread-safe event gate — same pattern as multiplier_env."""

    def __init__(self):
        self._lock = threading.RLock()
        self._ev = threading.Event()

    def signal(self):
        with self._lock:
            self._ev.set()

    def is_set(self):
        return self._ev.is_set()

    def clear(self):
        self._ev.clear()

    def __enter__(self):
        self._lock.acquire()
        return self

    def __exit__(self, *_exc):
        self._lock.release()


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class VelocityProfileEnv(gym.Env):
    """20-step MPC velocity profile control environment."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        node: Node,
        executor,
        *,
        horizon: int = _HORIZON,
        v_max: float = 25.0,
        v_min: float = 1.0,
        ref_path_csv: str | None = None,
        smoothness_weight: float = 0.1,
        speed_weight: float = 0.01,
        max_steps: int = 0,  # 0 = unlimited
    ):
        super().__init__()
        self.node = node
        self._executor = executor
        self._horizon = horizon
        self._v_max = v_max
        self._max_steps = max_steps

        # ── Observation space (71 dims) ────────────────────────
        obs_dim = _SECTIONS + 1 + horizon * 3
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # ── Action space — floor prevents instant stall on very early random exploration ──
        self._v_min = v_min
        self.action_space = gym.spaces.Box(
            low=self._v_min, high=v_max, shape=(horizon,), dtype=np.float32
        )

        # ── State tracking ─────────────────────────────────────
        self.section = 0
        self.speed = 0.0
        self.colliding = False
        self._prev_sec = 0
        self._prev_vel_profile = np.zeros(horizon, dtype=np.float32)
        self._gate = _TickGate()
        self._odom_gate = _TickGate()
        self._needs_reset = False       # set by step(), consumed by next step()
        self._first_reset_done = False  # hard reset done once at boot
        self._prev_speed = 0.0
        self._low_speed_count = 0       # consecutive ticks below 0.5 m/s
        self._post_reset_steps = 0      # grace period after AWSIM reset
        self._step_count = 0             # step counter for debug output

        # ── Reward params ──────────────────────────────────────
        self._smoothness_weight = smoothness_weight
        self._speed_weight = speed_weight

        # ── Reference path data ────────────────────────────────
        csv_path = ref_path_csv or _DEFAULT_REF_PATH_CSV
        try:
            ref_data = _load_ref_path_data(csv_path)
            self._kappa = ref_data["kappa"]
            self._ref_path_loaded = True
            print(
                f"[INFO] Reference path loaded: {len(self._kappa)} waypoints "
                f"from {csv_path}"
            )
        except Exception as e:
            print(f"[WARN] Failed to load reference path ({e}), using zero curvature")
            self._kappa = np.zeros(1000, dtype=np.float64)
            self._ref_path_loaded = False

    # ── Internal helpers ───────────────────────────────────────

    def _reset_state(self):
        """Reset internal bookkeeping (no AWSIM touches)."""
        self.section = 0
        self.speed = 0.0
        self.colliding = False
        self._prev_sec = -1         # ensure first section crossing fires
        self._prev_speed = 0.0
        self._low_speed_count = 0
        # Don't clear _needs_reset here — it's consumed by step() after reset()
        self._post_reset_steps = 5  # grace ticks — stall check suspended
        self._step_count = 0
        self._prev_vel_profile = np.zeros(self._horizon, dtype=np.float32)

    def _reset_awsim(self):
        """Full AWSIM reset: 1) reset  2) initial pose  3) engage MPC."""
        # 1. Reset AWSIM (domain 0)
        self.node._pub_reset_d0.publish(Empty())
        time.sleep(5.0)

        # Spin on domain 1 while AWSIM settles
        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline:
            self._executor.spin_once(timeout_sec=0.1)

        # 2. Re-publish initial pose so kart starts at correct position
        if self.node._cli_initial_pose.service_is_ready():
            try:
                self.node._cli_initial_pose.call_async(Trigger.Request())
            except Exception as e:
                print(f"[WARN] initial-pose service failed: {e}")
        time.sleep(5.0)

        # 3. Engage — hand control to MPC
        self.node._pub_control_mode.publish(Bool(data=True))
        time.sleep(5.0)

        # Wait for kart to start moving (up to 5 s)
        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline:
            self._executor.spin_once(timeout_sec=0.1)
            if self.speed > 0.5:
                break

        # Flush any stale gate signals accumulated during reset
        self._gate.clear()
        self._odom_gate.clear()

    def _get_geometry_profiles(self):
        """Curvature + width for the next N waypoints from ref path CSV."""
        n_wps = len(self._kappa)
        if n_wps == 0:
            return np.zeros(self._horizon, dtype=np.float32), \
                   np.full(self._horizon, 6.0, dtype=np.float32)

        wp_id = int((self.section / _SECTIONS) * n_wps) % n_wps
        curvatures = np.zeros(self._horizon, dtype=np.float32)
        widths = np.full(self._horizon, 6.0, dtype=np.float32)

        for i in range(self._horizon):
            idx = (wp_id + i) % n_wps
            curvatures[i] = float(self._kappa[idx])

        return curvatures, widths

    def _obs(self):
        """Build 71-dim observation vector."""
        oh = np.zeros(_SECTIONS, dtype=np.float32)
        idx = self.section if 0 <= self.section < _SECTIONS else 0
        oh[idx] = 1.0

        speed = np.array([self.speed], dtype=np.float32)
        curvatures, widths = self._get_geometry_profiles()

        return np.concatenate([
            oh, speed, curvatures, widths, self._prev_vel_profile
        ]).astype(np.float32)

    def _spin_until_tick(self, timeout=0.5):
        """Block until BOTH status and odometry callbacks fire, or timeout."""
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            got_status = self._gate.is_set()
            got_odom = self._odom_gate.is_set()
            if got_status and got_odom:
                self._gate.clear()
                self._odom_gate.clear()
                return True
            self._executor.spin_once(timeout_sec=0.01)
        # Timeout — use whatever we have
        if self._gate.is_set():
            self._gate.clear()
        if self._odom_gate.is_set():
            self._odom_gate.clear()
        print("[WARN] missed AWSIM tick -> using stale obs")
        return False

    # ── Gym API ────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if not self._first_reset_done:
            # Boot-time hard reset only
            self._reset_awsim()
            self._first_reset_done = True

        with self._gate:
            self._reset_state()
        self._odom_gate.clear()

        return self._obs(), {}

    def step(self, action):
        # ── Deferred post-crash reset (deferred one tick) ──────
        if self._needs_reset:
            self._reset_awsim()
            with self._gate:
                self._reset_state()
            self._needs_reset = False
            self._odom_gate.clear()
            return self._obs(), 0.0, False, False, {"reset": True}

        # ── Wait for AWSIM tick ────────────────────────────────
        self._spin_until_tick(timeout=0.5)
        self._step_count += 1

        # ── Publish velocity profile to MPC ────────────────────
        vel_profile = np.clip(
            action.astype(np.float64), self.action_space.low, self.action_space.high
        )
        if self._step_count % 200 == 0:
            mean_v = float(np.mean(vel_profile))
            print(f"[RL] step={self._step_count:4d} | mean_v={mean_v:.3f} | speed={self.speed:.2f} | section={self.section} | colliding={self.colliding}")
        msg = Float32MultiArray()
        msg.data = [float(v) for v in vel_profile]
        self.node._pub_vel_profile.publish(msg)

        # ── Compute reward + termination ───────────────────────
        with self._gate:
            # Grace period countdown (stall detection disabled during it)
            if self._post_reset_steps > 0:
                self._post_reset_steps -= 1

            crossed = int(self.section != self._prev_sec)
            self._prev_sec = self.section

            curr_speed = max(0.0, self.speed)
            sudden_drop = (self._prev_speed >= 1.0 and
                           (self._prev_speed - curr_speed) >= 1.0)
            if curr_speed < 0.5:
                self._low_speed_count += 1
            else:
                self._low_speed_count = 0

            # Stall only counted AFTER grace period elapses
            stalled = (self._post_reset_steps <= 0) and \
                      (self._low_speed_count >= 10)
            if stalled:
                print(f"STALLING : {self._post_reset_steps}")
            self.colliding = sudden_drop or stalled
            self._prev_speed = curr_speed

            reward = 1.0 * crossed + self._speed_weight * max(0.0, curr_speed)
            # Smoothness penalty: penalize large jumps between consecutive velocity steps
            if self._prev_vel_profile is not None and len(self._prev_vel_profile) == self._horizon:
                diffs = np.diff(vel_profile)
                reward -= self._smoothness_weight * float(np.sum(diffs ** 2))
            if self.colliding:
                reward -= 100.0

        terminated = bool(self.colliding and self.speed < 1.0)
        # Hard step limit: end episode cleanly when max_steps reached
        if self._max_steps > 0 and self._step_count >= self._max_steps:
            terminated = True
        if terminated:
            print(" EPISODE TERMINATED")
            self._needs_reset = True

        # Save action for observation continuity
        self._prev_vel_profile = np.array(vel_profile, dtype=np.float32)

        return self._obs(), float(reward), terminated, False, {
            "section_crossed": bool(crossed),
            "mean_v": float(np.mean(vel_profile)),
        }

    # ── ROS callbacks (called from main-thread spin) ──────────

    def on_status(self, msg):
        with self._gate:
            if len(msg.data) > 3:
                ns = int(msg.data[3]) % _SECTIONS
                if ns != self.section:
                    self.section = ns
        self._gate.signal()

    def on_odom(self, msg):
        with self._odom_gate:
            self.speed = float(msg.twist.twist.linear.x)
        self._odom_gate.signal()


# ---------------------------------------------------------------------------
# Node wrapper (dual-domain ROS 2)
# ---------------------------------------------------------------------------

class VelocityProfileEnvNode(Node):
    """ROS 2 node: dual-domain + velocity profile publisher.

    - Domain 1: subscriptions (/awsim/status, /localization/kinematic_state)
                publishers (/mpc/ref_vel_profile, /awsim/control_mode_request_topic)
    - Domain 0: publisher (/admin/awsim/reset)
    """

    def __init__(self, ctx0: Context, ctx1: Context, **env_kw):
        super().__init__("velocity_profile_env_node",
                         context=ctx1, allow_undeclared_parameters=True)

        # ── Domain 1 executor (used by env for main-thread spin) ──
        self._exec1 = rclpy.executors.SingleThreadedExecutor(context=ctx1)
        self._exec1.add_node(self)

        self.env = VelocityProfileEnv(self, self._exec1, **env_kw)

        # ── Domain 1 subscriptions ─────────────────────────────
        self.create_subscription(
            Float32MultiArray, "/awsim/status", self.env.on_status, 10
        )
        self.create_subscription(
            Odometry, "/localization/kinematic_state", self.env.on_odom, 10
        )

        # ── Domain 1 publishers ────────────────────────────────
        self._pub_control_mode = self.create_publisher(
            Bool, "/awsim/control_mode_request_topic", 10
        )
        self._pub_vel_profile = self.create_publisher(
            Float32MultiArray, "/mpc/ref_vel_profile", 10
        )

        # ── Domain 1 client: initial pose (best-effort) ────────
        self._cli_initial_pose = self.create_client(Trigger, "/set_initial_pose")

        # ── Domain 0: AWSIM reset ──────────────────────────────
        self._d0_node = rclpy.create_node("velocity_profile_env_d0", context=ctx0)
        self._pub_reset_d0 = self._d0_node.create_publisher(
            Empty, "/admin/awsim/reset", 10
        )

        # ── Background spin for domain 0 ───────────────────────
        self._exec0 = rclpy.executors.SingleThreadedExecutor(context=ctx0)
        self._exec0.add_node(self._d0_node)
        self._cancel = threading.Event()
        self._thread_d0 = threading.Thread(target=self._spin_d0, daemon=True)
        self._thread_d0.start()

    def _spin_d0(self):
        while not self._cancel.is_set():
            try:
                self._exec0.spin_once(timeout_sec=0.1)
            except Exception:
                break

    def shutdown(self):
        """Stop background spin and destroy nodes."""
        self._cancel.set()
        self._exec0.shutdown()
        self._exec1.shutdown()
        try:
            self._d0_node.destroy_node()
        except Exception:
            pass
        try:
            self.destroy_node()
        except Exception:
            pass
