#!/usr/bin/env python3
"""Progress-speed reward for multiplier env: section bonus + small speed reward - collision penalty."""

from __future__ import annotations

from context.context_types import StepContext
from reward.interfaces import RewardFunction


class ProgressSpeedReward(RewardFunction):
    def __init__(
        self,
        progress_bonus: float = 1.0,
        speed_reward_scale: float = 0.05,
        collision_penalty: float = 100.0,
    ) -> None:
        self.progress_bonus = progress_bonus
        self.speed_reward_scale = speed_reward_scale
        self.collision_penalty = collision_penalty

    def compute(self, context: StepContext,) -> tuple[float, StepContext]:
        speed = float(context.env_state.get_value("vehicle_speed_mps", 0.0))
        section_now = int(context.env_state.get_value("awsim_section", 0))
        section_prev = int(context.info.get("_prev_section", section_now))

        section_crossed = 1.0 if section_now != section_prev else 0.0
        context.info["_prev_section"] = section_now
        context.info["reward_breakdown"] = {
            "progress": self.progress_bonus * section_crossed,
            "speed": self.speed_reward_scale * speed,
            "collision": -self.collision_penalty if context.collision else 0.0,
        }

        return (
            self.progress_bonus * section_crossed + self.speed_reward_scale * speed -
            (self.collision_penalty if context.collision else 0.0),
            context,
        )
