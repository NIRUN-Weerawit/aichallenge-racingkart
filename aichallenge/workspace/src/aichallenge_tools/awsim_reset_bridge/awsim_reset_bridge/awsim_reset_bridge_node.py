#!/usr/bin/env python3
"""
ROS2 Domain Bridge Node for forwarding the AWSIM reset command.

Forwards:
  DOMAIN=1 : /awsim/reset      →  DOMAIN=0 : /admin/awsim/reset

This is a thin pass-through, not a controller. It exists so that an
Autoware-side node (e.g. an RL training script) on DOMAIN=1 can request
an AWSIM reset without needing to talk across DDS domains directly.

For full RL training/inference, see
`aichallenge/ml_workspace/reinforcement_learning/` (the SB3-based harness).
This package is only the reset bridge that both setups need.
"""

import threading

import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Empty


class AwsimResetBridgeNode(Node):
    """ROS2 Domain Bridge Node for forwarding AWSIM reset commands.

    Subscribes to /awsim/reset on DOMAIN=1,
    and publishes to /admin/awsim/reset on DOMAIN=0.
    """

    def __init__(self, ctx0: Context, ctx1: Context):
        # This node lives on the DOMAIN=1 side (subscriber).
        super().__init__('awsim_reset_bridge', context=ctx1)

        self.declare_parameter('src_topic', '/awsim/reset')
        self.declare_parameter('dst_topic', '/admin/awsim/reset')

        src_topic = self.get_parameter('src_topic').value
        dst_topic = self.get_parameter('dst_topic').value

        # --- DOMAIN=0 publisher node ---
        self._pub_node = rclpy.create_node('awsim_reset_bridge_pub', context=ctx0)
        self._pub = self._pub_node.create_publisher(Empty, dst_topic, 10)

        # --- DOMAIN=1 subscriber ---
        self.create_subscription(Empty, src_topic, self._reset_cb, 10)

        self.get_logger().info(
            f"AwsimResetBridge ready. "
            f"DOMAIN=1:{src_topic} -> DOMAIN=0:{dst_topic}"
        )

    def _reset_cb(self, msg: Empty):
        """Callback for /awsim/reset on DOMAIN=1."""
        self.get_logger().info("Reset on DOMAIN=1 -> forwarding to DOMAIN=0")
        self._pub.publish(Empty())

    def destroy_node(self):
        self._pub_node.destroy_node()
        super().destroy_node()


def main(args=None):
    # --- Two contexts, one per domain ---
    ctx0 = Context()
    rclpy.init(context=ctx0, domain_id=0, args=args)

    ctx1 = Context()
    rclpy.init(context=ctx1, domain_id=1, args=args)

    node = AwsimResetBridgeNode(ctx0=ctx0, ctx1=ctx1)

    # --- Spin DOMAIN=0 publisher in a worker thread ---
    exec0 = SingleThreadedExecutor(context=ctx0)
    exec0.add_node(node._pub_node)
    t = threading.Thread(target=exec0.spin, daemon=True)
    t.start()

    # --- Spin DOMAIN=1 subscriber on the main thread ---
    exec1 = SingleThreadedExecutor(context=ctx1)
    exec1.add_node(node)

    try:
        exec1.spin()
    except KeyboardInterrupt:
        pass
    finally:
        exec0.shutdown()
        exec1.shutdown()
        node.destroy_node()
        rclpy.shutdown(context=ctx0)
        rclpy.shutdown(context=ctx1)


if __name__ == '__main__':
    main()
