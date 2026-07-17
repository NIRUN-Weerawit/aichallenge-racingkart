#!/usr/bin/env python3
# ruff: noqa: E402, F811, E722
"""Gymnasium env for PPO-driven speed multiplier optimization.
Blocks on real AWSIM ticks via main-thread spinning + thread gate.

Architecture:
  - Domain 1: subscribes to /awsim/status (bridged by AWSIM), /localization/kinematic_state, etc.
  - Domain 0: publishes to /admin/awsim/reset (AWSIM's native reset topic)
  - Domain 1: publishes to /awsim/control_mode_request_topic (bridged to AWSIM)
  - Main thread spins rclpy while waiting for gate (no background thread needed for domain 1)
"""

from collections import deque
import threading
import time

import gymnasium as gym
import numpy as np
import rclpy
import rclpy.executors
from rclpy.context import Context
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import Bool, Empty, Float32MultiArray
from nav_msgs.msg import Odometry
from std_srvs.srv import Trigger
from rcl_interfaces.msg import Parameter as RclParameter, ParameterValue
from rcl_interfaces.srv import SetParameters

SECTIONS = ('s1', 's1_1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9')
BASE_VELS = {
    's1': 33.0, 's1_1': 33.0, 's2': 33.0, 's3': 33.0, 's4': 22.0,
    's5': 33.0, 's6': 22.0, 's7': 33.0, 's8': 25.0, 's9': 33.0,
}


class _TickGate:
    """Thread-safe event that blocks until AWSIM publishes a fresh tick."""

    def __init__(self):
        self._lock = threading.RLock()
        self._ev   = threading.Event()

    def signal(self):
        with self._lock:
            self._ev.set()

    def is_set(self):
        return self._ev.is_set()

    def clear(self):
        self._ev.clear()

    def __enter__(self):
        self._lock.acquire();  return self

    def __exit__(self, *_exc):
        self._lock.release()


class MultiplierEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, node: Node, executor, min_mult=0.7, max_mult=1.3):
        super().__init__()
        self.node = node
        self._executor = executor
        n         = len(SECTIONS)
        self._n   = n

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(n + 5 * n + 1,), dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(low=min_mult, high=max_mult, shape=(n,), dtype=np.float32)

        self.section     = 0
        self.speed       = 0.0
        self.colliding   = False
        self._prev_sec   = 0
        self.hist  = deque([np.ones(self._n, dtype=np.float32)] * 5, maxlen=5)
        self._gate = _TickGate()
        self._mpc_ready = False
        self._needs_reset = False
        self._prev_speed = 0.0
        self._low_speed_count = 0
        self._awsim_state = ""

    # ── Gym API ────────────────────────────────────────────────

    def _reset_awsim(self):
        """Full AWSIM reset sequence: reset → wait → initial pose → control mode."""
        self.node._pub_reset_d0.publish(Empty())
        time.sleep(3.0)

        # Wait for AWSIM state to be "Start" (up to 5s)
        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline:
            self._executor.spin_once(timeout_sec=0.1)
            if self._awsim_state == "Start":
                break

        # Call initial pose service (best-effort, don't block if not ready)
        if self.node._cli_initial_pose.service_is_ready():
            try:
                self.node._cli_initial_pose.call_async(Trigger.Request())
                print("[INFO] Initial pose service called")
            except Exception as e:
                print(f"[WARN] Initial pose service call failed: {e}")
        else:
            print("[WARN] Initial pose service not ready")
        time.sleep(1.0)

        # Hand control to MPC
        self.node._pub_control_mode.publish(Bool(data=True))
        time.sleep(1.0)

        # Wait for kart to start moving (up to 5s)
        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline:
            self._executor.spin_once(timeout_sec=0.1)
            if self.speed > 0.5:
                break

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        with self._gate:
            self.section   = 0;  self.speed = 0.0;  self.colliding = False
            self._prev_sec = 0
            self.hist.clear()
            self.hist.extend([np.ones(self._n, dtype=np.float32)] * 5)

        self._reset_awsim()
        return self._obs(), {}

    def step(self, action):
        # Auto-reset if we crashed on the previous step
        if self._needs_reset:
            self._reset_awsim()
            self._needs_reset = False
            with self._gate:
                self.section   = 0;  self.speed = 0.0;  self.colliding = False
                self._prev_sec = 0
                self._prev_speed = 0.0
                self._low_speed_count = 0
                self.hist.clear()
                self.hist.extend([np.ones(self._n, dtype=np.float32)] * 5)
            # Return immediately with zero reward — kart needs time to start moving
            return self._obs(), 0.0, False, False, {"section_crossed": False, "reset": True}

        # Spin rclpy on the main thread while waiting for the gate
        self._spin_until_tick(timeout=0.5)

        mults = np.clip(action.astype(np.float64),
                        self.action_space.low, self.action_space.high)

        # Set parameters on the MPC controller node via remote parameter service
        req = SetParameters.Request()
        for i, m in enumerate(mults):
            pv = ParameterValue(type=3, double_value=float(BASE_VELS[SECTIONS[i]] * m))
            req.parameters.append(RclParameter(
                name=f'ref_vel/{SECTIONS[i]}/ref_vel',
                value=pv,
            ))
        try:
            if not self._mpc_ready:
                # Spin until service is discovered (non-blocking)
                for _ in range(50):
                    if self.node._mpc_param_client.service_is_ready():
                        self._mpc_ready = True
                        break
                    self._executor.spin_once(timeout_sec=0.01)
            if self._mpc_ready:
                self.node._mpc_param_client.call_async(req)
        except Exception:
            pass

        with self._gate:
            crossed     = int(self.section != self._prev_sec)
            self._prev_sec = self.section
            self.hist.append(np.array(mults, dtype=np.float32))

            # Collision detection from speed drops (same logic as CollisionTermination)
            curr_speed = max(0.0, self.speed)
            sudden_drop = (self._prev_speed >= 1.0 and
                           (self._prev_speed - curr_speed) >= 1.5)
            if curr_speed < 0.5:
                self._low_speed_count += 1
            else:
                self._low_speed_count = 0
            self.colliding = sudden_drop or (self._low_speed_count >= 10)
            self._prev_speed = curr_speed

            reward   = 1.0 * crossed + 0.05 * max(0.0, curr_speed)
            if self.colliding:
                reward -= 100.0

        terminated = bool(self.colliding and self.speed < 1.0)
        if terminated:
            self._needs_reset = True
        return self._obs(), float(reward), terminated, False, {"section_crossed": bool(crossed)}

    def _spin_until_tick(self, timeout=0.5):
        """Spin rclpy on the main thread until a tick arrives or timeout."""
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            if self._gate.is_set():
                self._gate.clear()
                return True
            self._executor.spin_once(timeout_sec=0.01)
        # Final check after timeout
        if self._gate.is_set():
            self._gate.clear()
            return True
        print("[WARN] missed AWSIM tick → using stale obs")
        return False

    def _obs(self):
        oh   = np.zeros(self._n, dtype=np.float32)
        idx  = self.section if 0 <= self.section < self._n else 0
        oh[idx] = 1.0
        return np.concatenate([oh,
                                *[np.asarray(v, dtype=np.float32) for v in self.hist],
                                [self.speed]]).astype(np.float32)

    # ── ROS callbacks (called from main-thread spin) ──────────

    def on_status(self, msg):
        with self._gate:
            if len(msg.data) > 3:
                ns = int(msg.data[3]) % self._n
                if ns != self.section:  self.section = ns
        self._gate.signal()

    def on_odom(self, msg):
        with self._gate:  self.speed = float(msg.twist.twist.linear.x)
        self._gate.signal()


class MultiplierEnvNode(Node):
    """ROS 2 node with dual-domain support:
    - Domain 1: subscriptions + control mode publisher (main-thread spin)
    - Domain 0: AWSIM reset publisher (background thread spin)
    """

    def __init__(self, ctx0: Context, ctx1: Context, **env_kw):
        # Main node lives on domain 1
        super().__init__("multiplier_env_node", context=ctx1, allow_undeclared_parameters=True)

        # Create executor for domain 1 (used by env for main-thread spinning)
        self._exec1 = rclpy.executors.SingleThreadedExecutor(context=ctx1)
        self._exec1.add_node(self)

        self.env = MultiplierEnv(self, self._exec1, **env_kw)

        # ── Domain 1 subscriptions (topics bridged by AWSIM) ─────
        self.create_subscription(Float32MultiArray, "/awsim/status", self.env.on_status, 10)
        self.create_subscription(Odometry, "/localization/kinematic_state", self.env.on_odom, 10)

        # ── Domain 1 publisher: control mode request (bridged to AWSIM) ──
        self._pub_control_mode = self.create_publisher(Bool, "/awsim/control_mode_request_topic", 10)

        # ── Domain 1 client: initial pose service (re-publish after reset) ──
        self._cli_initial_pose = self.create_client(Trigger, "/set_initial_pose")

        # ── Remote parameter client for MPC controller ───────────
        # Use the SetParameters service directly (AsyncParameterClient not available in Humble)
        from rcl_interfaces.srv import SetParameters
        self._mpc_param_client = self.create_client(SetParameters, "/mpc_controller/set_parameters")

        # ── Domain 0 publisher: AWSIM reset (native topic) ──────
        self._d0_node = rclpy.create_node("multiplier_env_d0", context=ctx0)
        self._pub_reset_d0 = self._d0_node.create_publisher(Empty, "/admin/awsim/reset", 10)

        # ── Background spin for domain 0 only ────────────────────
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
        try:  self._d0_node.destroy_node()
        except Exception:  pass
        try:  self.destroy_node()
        except Exception:  pass
