# awsim_reset_bridge

A 90-line ROS 2 node that forwards AWSIM's reset command across DDS domains.

- DOMAIN=1: `/awsim/reset` (subscribed)
- DOMAIN=0: `/admin/awsim/reset` (published)

## What it is

A thin pass-through bridge, not a controller. It exists so that any
Autoware-side node (e.g. the SB3-based RL training script) running on
DOMAIN=1 can request an AWSIM reset without needing to talk across
DDS domains directly.

This is the bridge that's launched by `control_method:=rl_train` in
`reference.launch.xml`. **It does not load any model or drive the kart**
— to do that, see the SB3 harness in
`aichallenge/ml_workspace/reinforcement_learning/`.

## Directory layout

```
awsim_reset_bridge/
  CMakeLists.txt
  package.xml
  launch/
    awsim_reset_bridge.launch.xml
  awsim_reset_bridge/
    __init__.py
    awsim_reset_bridge_node.py
```

## Node summary

Implementation: `awsim_reset_bridge/awsim_reset_bridge_node.py`

- Node name: `awsim_reset_bridge`
- Subscription (parameter): `src_topic` (default: `/awsim/reset`)
- Publication  (parameter): `dst_topic` (default: `/admin/awsim/reset`)
- Message type: `std_msgs/msg/Empty`

Internally creates two ROS 2 Contexts (one per DOMAIN) and runs each
on its own executor.
