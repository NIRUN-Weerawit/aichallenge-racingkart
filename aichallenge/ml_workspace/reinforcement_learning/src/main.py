#!/usr/bin/env python3
"""AWSIM RL entry point — supports standard env + Option 3 multiplier env."""

import rclpy
from pathlib import Path

from config.load_config import load_config
from environment.awsim_env import AWSIMEnv
from select_parts import (
    select_action_adapter,
    select_algorithm,
    select_algorithm_class,
    select_context_manager,
    select_observation_builder,
    select_reward_function,
    select_termination_function,
    select_wrappers,
)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='YAML config path')
    parser.add_argument('--check', action='store_true',  help='Run env checker')
    parser.add_argument('--train', action='store_true',  help='Train PPO/SAC')
    parser.add_argument('--infer', action='store_true',  help='Inference with saved model')
    parser.add_argument('--model-path', type=str, default='awsim_sac_model',
                        help='Path to saved model')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Episodes for inference (default: 5)')
    args = parser.parse_args()

    cfgs = load_config(args.config)

    # Log / model save locations — auto-increment run number
    config_base_dir = Path(args.config).expanduser().resolve().parent if args.config else Path.cwd()
    run_num = 1
    while (config_base_dir / f'run_{run_num}').exists():
        run_num += 1
    run_dir = config_base_dir / f'run_{run_num}'
    run_dir.mkdir(parents=True, exist_ok=True)
    algorithm_cfg   = dict(cfgs['algorithm'])
    algorithm_cfg['save_path']       = str(run_dir / 'model')
    algorithm_cfg['tensorboard_log'] = str(run_dir / 'log')
    print(f"[INFO] Run directory: {run_dir}")

    # Shared components (used by standard env; multiplier env ignores)
    context_manager      = select_context_manager(cfgs['context_manager'])
    action_adapter       = select_action_adapter(cfgs['action_adapter'])
    observation_builder  = select_observation_builder(cfgs['observation_builder'])
    reward_function      = select_reward_function(cfgs['reward'])
    termination_function = select_termination_function(cfgs['termination'])

    # ── Env construction ────────────────────────────────────────
    is_multiplier = 'multiplier_env' in cfgs
    is_velocity_profile = 'velocity_profile_env' in cfgs

    if is_multiplier or is_velocity_profile:
        from rclpy.context import Context
        # Force localhost discovery so our new DDS participant finds AWSIM's bridge
        import os
        os.environ.setdefault('ROS_AUTOMATIC_DISCOVERY_RANGE', 'LOCALHOST')
        # Two contexts: domain 0 for AWSIM reset, domain 1 for everything else
        ctx0 = Context()
        rclpy.init(context=ctx0, domain_id=0)
        ctx1 = Context()
        rclpy.init(context=ctx1, domain_id=1)

        if is_velocity_profile:
            print("[INFO] Velocity profile env active (Option 4)")
            from environment.velocity_profile_env import VelocityProfileEnvNode
            vpe = cfgs['velocity_profile_env']
            env_node = VelocityProfileEnvNode(
                ctx0=ctx0, ctx1=ctx1,
                horizon=int(vpe.get('horizon', 20)),
                v_min=float(vpe.get('v_min', 0.1)),
                v_max=float(vpe.get('v_max', 20.0)),
                smoothness_weight=float(vpe.get('smoothness_weight', 0.1)),
                speed_weight=float(vpe.get('speed_weight', 0.01)),
            )
            env = env_node.env
        else:
            print("[INFO] Multiplier env active (ctx0=domain 0, ctx1=domain 1)")
            from environment.multiplier_env import MultiplierEnvNode
            meg     = cfgs['multiplier_env']
            env_node = MultiplierEnvNode(
                ctx0=ctx0, ctx1=ctx1,
                min_mult=float(meg.get('min_multiplier', 0.7)),
                max_mult=float(meg.get('max_multiplier', 1.3)),
            )
            env = env_node.env
    else:
        env = AWSIMEnv(
            context_manager=context_manager,     action_adapter=action_adapter,
            observation_builder=observation_builder, reward_function=reward_function,
            termination_function=termination_function,
        )

    # Optional Gym wrappers (TimeLimit, etc.)
    env = select_wrappers(algorithm_cfg, env)

    # ── Execution Modes ─────────────────────────────────────────

    if args.check:
        from stable_baselines3.common.env_checker import check_env
        import time as _t
        print("Checking environment against SB3 spec...")
        # Wait for DDS discovery: spin until we receive at least one /awsim/status tick
        if is_multiplier and 'env_node' in locals():
            print("  Waiting for DDS discovery (up to 5s)...")
            deadline = _t.perf_counter() + 5.0
            while _t.perf_counter() < deadline:
                env_node._exec1.spin_once(timeout_sec=0.1)
                if env_node.env._gate.is_set():
                    env_node.env._gate.clear()
                    print("  DDS discovery complete!")
                    break
        check_env(env, warn=True)
        print("Check passed!")

    elif args.train:
        # Spawn the kart: AWSIM needs `/admin/awsim/reset` (domain 0) + Auto control BEFORE PPO starts
        if is_multiplier:
            from std_msgs.msg import Bool, Empty
            import time as _t
            env_node._pub_reset_d0.publish(Empty())
            _t.sleep(3.0)  # AWSIM needs ~3s to reset and stabilize
            env_node._pub_control_mode.publish(Bool(data=True))
            _t.sleep(0.5)
            print("Spawned AWSIM -> Auto control. Starting training...\n")

        # ── Wandb logging ────────────────────────────────────────
        try:
            import wandb
            wandb.init(
                project="aichallenge-option3",
                config=dict(cfgs),
                sync_tensorboard=True,
            )
            print(f"[INFO] Wandb logging enabled (project=aichallenge-option3)")
        except ImportError:
            print("[INFO] wandb not installed — logging to TensorBoard only")

        model = select_algorithm(algorithm_cfg, env)
        total = int(algorithm_cfg.get('total_timesteps', 200_000))

        # ── TensorBoard callback: log mean_v from info dict ─────
        from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

        class LogMeanVCallback(BaseCallback):
            def _on_step(self):
                if "mean_v" in self.locals.get("infos", [{}])[0]:
                    self.logger.record("rollout/mean_v", self.locals["infos"][0]["mean_v"])
                return True

        # ── Checkpoint callback: save every 10k steps ────────────
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=str(run_dir / "checkpoints"),
            name_prefix="sac_model",
        )

        print(f"Starting training for {total} timesteps...\n")
        model.learn(total_timesteps=total,
                    log_interval=int(algorithm_cfg.get('log_interval', 1)),
                    callback=[LogMeanVCallback(), checkpoint_callback])
        model.save(algorithm_cfg['save_path'])
        print(f"\nTraining complete! Policy saved to: {algorithm_cfg['save_path']}")

        # ── Close wandb ──────────────────────────────────────────
        try:
            wandb.finish()
        except Exception:
            pass

    elif args.infer:
        AlgorithmClass = select_algorithm_class(algorithm_cfg)
        model = AlgorithmClass.load(str(args.model_path), env=env)
        print(f"Model loaded from {args.model_path}. "
              f"Running {args.episodes} episode(s)...")

        for ep in range(1, args.episodes + 1):
            obs, info = env.reset()
            ep_reward, done = 0.0, False
            print(f"\n=== Episode {ep}/{args.episodes} ===")

            step = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, rew, tr, trunc, info = env.step(action)
                ep_reward += rew
                step    += 1
                done    = tr or trunc

                spd = info.get('speed', 0.0)
                sec = info.get('section', '?')
                print(f"  {step:4d} | reward={rew:+7.2f} | "
                      f"speed={spd:5.2f} m/s | section={sec}")

            print(f"> Episode complete: total_reward={ep_reward:.1f} over {step} steps.")

    else:
        if is_multiplier:
            print("[INFO] Multiplier env active. Use --train or --infer to run!")
        else:
            print("Running 200-step random sanity loop...")
            obs, info = env.reset()
            for i in range(200):
                action = env.action_space.sample()
                obs, rew, tr, trunc, info = env.step(action)
                if (i + 1) % 50 == 0:
                    print(f"Steps done: {i + 1}")

    # Cleanup
    env.close()
    if is_multiplier:
        try:
            env_node.shutdown()   # stop background spin safely
        except Exception:
            pass
        # Shutdown both domain contexts
        if rclpy.ok(context=ctx0):
            rclpy.shutdown(context=ctx0)
        if rclpy.ok(context=ctx1):
            rclpy.shutdown(context=ctx1)
