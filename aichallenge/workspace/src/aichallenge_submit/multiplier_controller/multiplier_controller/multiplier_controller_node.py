#!/usr/bin/env python3
"""PPO-driven per-section speed multiplier for MPC reference velocity."""

import os

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import Float32MultiArray, Int32

# Base ref_vel (km/h) per section — from config/ref_vel.yaml
BASE_VELS = {
    's1': 33.0, 's1_1': 33.0, 's2': 33.0, 's3': 33.0, 's4': 22.0,
    's5': 33.0, 's6': 22.0, 's7': 33.0, 's8': 25.0, 's9': 33.0,
}

SECTIONS = tuple(BASE_VELS.keys())
N_SECTIONS = len(SECTIONS)


class MultiplierControllerNode(Node):

    def __init__(self):
        super().__init__('multiplier_controller')

        # Parameters
        policy_path = self.declare_parameter('policy_path', '').value
        self._n = int(self.declare_parameter('num_sections', N_SECTIONS).value)
        self._min_m = float(self.declare_parameter('min_mult', 0.7).value)
        self._max_m = float(self.declare_parameter('max_mult', 1.3).value)

        # Load trained PPO policy (or None for random-init baseline)
        self._model = None
        if policy_path and os.path.isfile(str(policy_path)):
            try:
                from stable_baselines3 import PPO
                self._model = PPO.load(str(policy_path), device='cpu')
                self.get_logger().info(f"Loaded policy: {policy_path}")
            except Exception as e:
                self.get_logger().warn(f"Failed to load policy, using random init: {e}")

        # State
        self._section = 0
        self._speed = 0.0
        self._history = [np.ones(self._n)] * 5          # recent multiplier vectors

        # Timers / pubs
        hist_len = len(self._history[-1]) if self._history[0] is not None else self._n
        self.history_pub = self.create_publisher(Float32MultiArray, '/mpc/multiplier_history', 1)

    # ── callbacks ─────────────────────────────────────────────

    def on_status(self, msg: Float32MultiArray):
        if len(msg.data) > 3:
            self._section = int(msg.data[3]) % self._n
        self._step()

    def on_vel(self, msg):
        self._speed = float(msg.linear.x)

    # ── core logic ────────────────────────────────────────────

    def _obs(self) -> np.ndarray:
        """~25-float observation: one-hot(section) + history[-1] + speed."""
        oh = np.zeros(self._n, dtype=np.float32)
        if 0 <= self._section < self._n:
            oh[self._section] = 1.0
        hist = list(self._history[-1]) if self._history[-1] is not None else [1.0]*self._n
        return np.concatenate([oh, hist, [self._speed]]).astype(np.float32)

    def _predict(self) -> np.ndarray:
        o = self._obs()
        if self._model is not None:
            a, _ = self._model.predict(o.reshape(1, -1), deterministic=False)
        else:
            a = np.random.uniform(self._min_m, self._max_m, (1, self._n))
        return np.clip(a[0], self._min_m, self._max_m).astype(float)

    def _apply(self, mults):
        """Push multipliers to MPC via existing ref_vel/{sec}/ref_vel params."""
        params = []
        for i, m in enumerate(mults[:self._n]):
            sec = SECTIONS[i]
            base = BASE_VELS[sec]
            new_vel = base * m
            params.append(Parameter(f'ref_vel/{sec}/ref_vel', value=new_vel))
        if params:
            self.set_parameters(params)

    def _step(self):
        mults = self._predict()
        self._history.append(np.array(mults[:self._n]))
        if len(self._history) > 5:
            self._history.pop(0)
        self._apply(mults)
        # publish for visibility
        arr = Float32MultiArray(data=[float(x) for x in mults[:self._n]])
        self.history_pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = MultiplierControllerNode()

    try:
        # Subscribe to simulator status (section + lap info)
        node.create_subscription(Float32MultiArray, '/awsim/status', node.on_status, 10)
        # Subscribe to velocity for observation
        from geometry_msgs.msg import TwistStamped
        node.create_subscription(TwistStamped, '/localization/kinematic_state/twist', node.on_vel, 10)

        # Also run on timer as fallback (in case status topic is slow)
        node.create_timer(0.05, lambda: None)            # keep node alive if no callbacks fire
        
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
