#!/usr/bin/env python3
"""Reward function for 20-step velocity profile RL.

reward = lap_progress_bonus 
       - smoothness_weight * sum(|v_n - v_{n-1}|^2) 
       + speed_weight * mean(v)
       - collision_penalty (if colliding)
"""


class VelocityProfileReward:
    """Compute reward for velocity profile actions.
    
    Can be called independently by the env or via select_reward_function routing.
    Note: The VelocityProfileEnv computes its own reward inline, but this class
    exists for external training scripts that may want to decouple reward logic.
    """

    def __init__(self, smoothness_weight=0.1, speed_weight=0.01, collision_penalty=100.0):
        self._smoothness_weight = float(smoothness_weight)
        self._speed_weight = float(speed_weight)
        self._collision_penalty = float(collision_penalty)

    def __call__(self, action, prev_action, speed, colliding, section_crossed=False):
        """Compute reward from state and action.
        
        Args:
            action: current 20-velocity profile (numpy array)
            prev_action: previous velocity profile (or None for first step)
            speed: current kart speed in m/s
            colliding: boolean collision flag
            section_crossed: boolean — did we cross a section boundary this step
            
        Returns:
            reward: float
        """
        import numpy as np
        
        reward = 0.0
        
        # Section crossing bonus (lap progress)
        if section_crossed:
            reward += 1.0
        
        # Smoothness penalty between consecutive velocity steps
        action_arr = np.asarray(action, dtype=np.float64)
        diffs = np.diff(action_arr)
        reward -= self._smoothness_weight * np.sum(diffs ** 2)
        
        # Speed encouragement (encourage higher average velocity)
        reward += self._speed_weight * float(np.mean(action_arr))
        
        # Collision penalty
        if colliding:
            reward -= self._collision_penalty
        
        return reward
