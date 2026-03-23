from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

from .commands import MotionCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class RSIStage(TypedDict, total=False):
  """A single stage in the RSI noise curriculum."""

  step: int
  pose_range: dict[str, tuple[float, float]]
  velocity_range: dict[str, tuple[float, float]]
  joint_position_range: tuple[float, float]


def motion_rsi_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  stages: list[RSIStage],
) -> dict[str, torch.Tensor]:
  """Gradually increase RSI noise during training.

  At the start of training, the policy is untrained and outputs near-zero
  actions, which target the default standing pose (due to
  use_default_offset=True). If the robot is RSI-initialized to a dance
  pose with large noise, the PD controller immediately fights back toward
  standing, causing large end-effector errors and early termination.

  This curriculum starts with zero RSI noise so the robot begins exactly
  at the reference. The policy learns basic corrections first, then
  progressively larger perturbations are introduced.
  """
  del env_ids
  command_term = env.command_manager.get_term(command_name)
  cfg = cast(MotionCommandCfg, command_term.cfg)

  for stage in stages:
    if env.common_step_counter > stage["step"]:
      if "pose_range" in stage:
        cfg.pose_range = stage["pose_range"]
      if "velocity_range" in stage:
        cfg.velocity_range = stage["velocity_range"]
      if "joint_position_range" in stage:
        cfg.joint_position_range = stage["joint_position_range"]

  return {
    "pose_range_x": torch.tensor(cfg.pose_range.get("x", (0.0, 0.0))[1]),
    "velocity_range_x": torch.tensor(cfg.velocity_range.get("x", (0.0, 0.0))[1]),
    "joint_position_range": torch.tensor(cfg.joint_position_range[1]),
  }


class TerminationThresholdStage(TypedDict):
  """A single stage in the termination threshold curriculum."""

  step: int
  threshold: float


def termination_threshold_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  termination_name: str,
  stages: list[TerminationThresholdStage],
) -> torch.Tensor:
  """Gradually tighten termination thresholds during training.

  Starts with loose thresholds so the untrained policy can survive
  longer, then tightens as the policy improves.
  """
  del env_ids
  term_cfg = env.termination_manager.get_term_cfg(termination_name)

  for stage in stages:
    if env.common_step_counter > stage["step"]:
      term_cfg.params["threshold"] = stage["threshold"]

  return torch.tensor(term_cfg.params["threshold"])


class RewardWeightStage(TypedDict):
  """A single stage in a reward weight curriculum."""

  step: int
  weight: float


def reward_weight_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str,
  stages: list[RewardWeightStage],
) -> torch.Tensor:
  """Gradually change a reward term's weight during training."""
  del env_ids
  reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)
  for stage in stages:
    if env.common_step_counter > stage["step"]:
      reward_term_cfg.weight = stage["weight"]
  return torch.tensor(reward_term_cfg.weight)
