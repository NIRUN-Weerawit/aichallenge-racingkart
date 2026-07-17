#!/usr/bin/env python3

"""PPO-driven speed multiplier node for MPC per-section reference velocity."""

import os

# ROS 2 imports
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import Float32MultiArray


class MultiplierControllerNode(Node):

    NUM_SECTIONS = 10
    MIN_MULT = 0.7
    MAX_MULT = 1.3

    BASE_VELS_KMH = {
        's1': 33.0, 's1_1': 33.0, 's2': 33.0, 's3': 33.0, 's4': 22.0,
        's5': 33.0, 's6': 22.0, 's7': 33.0, 's8': 25.0, 's9': 33.0,
    }

    SECTION_NAMES = tuple(BASE_VELS_KMH.keys())[:NUM_SECTIONS]

    # ------------------------------------------------------------------ init --

    def __init__(self):
        super().__init__('multiplier_controller_node')

        self._n = int(self.declare_parameter('num_sections', self.NUM_SECTIONS).value)
        self._min_m = float(self.declare_parameter('min_mult', self.MIN_MULT).value)
        self._max_m = float(self.declare_parameter('max_mult', self.MAX_MULT).value)

        # Load trained policy if path is provided
        policy_path = str(
            self.declare_parameter('policy_path', '').get_parameter_value().string_value)
        self._model = None

        if policy_path and os.path.isfile(str(policy_path)):
            from stable_baselines3 import PPO
            try:
                self._model = PPO.load(str(policy_path), device='cpu')
                self.get_logger().info(f"Loaded policy: {policy_path}")
            except Exception as e:
                self.get_logger().warn(f"Failed to load policy, will use random init: {e}")

        # State / bookkeeping
        self._recent_multipliers = [None] * 5  # last N multiplier vectors (10 each)
        self._cur_section_idx = -1
        self._section_pub = self.create_publisher(Int32, '/mpc/current_section', 10)

    # ---------------------------------------------------------------- callbacks --

    def awsim_status_callback(self, msg: Float32MultiArray):
        """Called every step with /awsim/status [section, lap_cnt, lap_time]"""
        section = int(msg.data[3]) if len(msg.data) > 3 else self._cur_section_idx
        self._cur_section_idx = section

    def _make_obs(self) -> 'np.ndarray':
        """Build observation vector per Option 3 spec (~25 floats)."""
        import numpy as np
        o = []
        # one-hot of current section (self._n dim)
        o.append([0.0] * self._cur_section_idx + [1.0] +
                 [0.0] * (self._n - 1 - self._cur_section_idx))
        # scalar: last published multiplier for prev step, or 1.0 default
        if len(self._recent_multipliers) >= 5 and self._recent_multipliers[-1] is not None:
            o.extend(list(self._recent_multipliers[-1]))   # shape (self._n,)
        else:
            o.extend([1.0] * self._n)                           # default
        return np.array(o, dtype=np.float32)

    def _publish_mul_table(self, mults):
        """Apply multiplier vector to MPC via set_parameters batch call."""
        params = []
        for i, m in enumerate(mults):
            sec_name = self.SECTION_NAMES[i]
            base_val = float(self.BASE_VELS_KMH[sec_name])
            clamped = max(self._min_m, min(self._max_m, float(m)))
            new_vel = base_val * clamped
            params.append(Parameter(
                f'ref_vel/{sec_name}/vel', value=new_vel))
        self.set_parameters(params)

    # ------------------------------------------------------------------ run --

    def _inference_step(self):
        """Run one inference + parameter-apply. Hook this into a timer / callback."""
        obs = self._make_obs()
        if self._model is not None:
            action, _ = self._model.predict(obs[None], deterministic=False)
        else:
            # Random uniform in [0.7, 1.3] as default baseline
            import numpy as np
            action = np.random.uniform(self._min_m, self._max_m, (1, self._n))

        action_clamped = np.clip(action[0], self._min_m, self._max_m)
        action_clamped_list = [float(x) for x in action_clamped]
        self._recent_multipliers.append(tuple(action_clamped_list))
        if len(self._recent_multipliers) > 5:
            self._recent_multipliers.pop(0)

        self._publish_mul_table(action_clamped_list)


def main(args=None):
    rclpy.init(args=args)
    node = MultiplierControllerNode()
    try:
        # Timer to run inference every ~25 ms at 40 Hz target control rate
        node.create_timer(0.025, node._inference_step)
        rclpy.spin(node)
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
