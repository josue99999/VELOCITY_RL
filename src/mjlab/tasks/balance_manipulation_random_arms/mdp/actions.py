"""Joint position action that randomizes upper-body (arm) targets.

The action keeps the same ``action_dim`` as the standard
``JointPositionAction`` (all joints), so checkpoints trained with the
original task load without modification.  Policy outputs for arm joints
are discarded and replaced with positions sampled uniformly from the
soft joint limits.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions.actions import (
  JointPositionAction,
  JointPositionActionCfg,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class RandomArmsJointPositionActionCfg(JointPositionActionCfg):
  """Joint position action that overrides arm joints with random targets.

  ``arm_joint_patterns`` identifies which joints are "arms".  Those
  joints receive uniformly-random position targets instead of the
  policy output, while the remaining joints (lower body / waist) stay
  under policy control.  Because ``actuator_names`` still matches all
  joints, ``action_dim`` is unchanged and existing checkpoints are
  compatible.
  """

  arm_joint_patterns: tuple[str, ...] = field(default_factory=tuple)
  """Regex patterns identifying arm joints to randomize."""

  resample_interval_steps: int = 25
  """Env steps between random-target resampling (~0.5 s at 50 Hz)."""

  def build(self, env: ManagerBasedRlEnv) -> RandomArmsJointPositionAction:
    return RandomArmsJointPositionAction(self, env)


class RandomArmsJointPositionAction(JointPositionAction):
  """Joint position action that randomizes arm joint targets."""

  cfg: RandomArmsJointPositionActionCfg

  def __init__(
    self,
    cfg: RandomArmsJointPositionActionCfg,
    env: ManagerBasedRlEnv,
  ):
    super().__init__(cfg=cfg, env=env)

    arm_indices = [
      i
      for i, name in enumerate(self._target_names)
      if any(re.fullmatch(p, name) for p in cfg.arm_joint_patterns)
    ]
    self._arm_idx = torch.tensor(arm_indices, device=self.device, dtype=torch.long)

    arm_target_ids = self._target_ids[self._arm_idx]
    self._arm_lo = self._entity.data.soft_joint_pos_limits[:, arm_target_ids, 0]
    self._arm_hi = self._entity.data.soft_joint_pos_limits[:, arm_target_ids, 1]

    self._arm_targets = self._sample(env_ids=None)
    # Stagger initial counters so envs don't resample in sync.
    self._counter = torch.randint(
      0,
      cfg.resample_interval_steps,
      (self.num_envs,),
      device=self.device,
    )

  # ------------------------------------------------------------------
  # Helpers
  # ------------------------------------------------------------------

  def _sample(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
    lo = self._arm_lo if env_ids is None else self._arm_lo[env_ids]
    hi = self._arm_hi if env_ids is None else self._arm_hi[env_ids]
    return lo + torch.rand_like(lo) * (hi - lo)

  # ------------------------------------------------------------------
  # Overrides
  # ------------------------------------------------------------------

  def process_actions(self, actions: torch.Tensor) -> None:
    super().process_actions(actions)

    self._counter += 1
    need = self._counter >= self.cfg.resample_interval_steps
    if need.any():
      ids = need.nonzero(as_tuple=False).view(-1)
      self._arm_targets[ids] = self._sample(ids)
      self._counter[ids] = 0

    self._processed_actions[:, self._arm_idx] = self._arm_targets

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    if isinstance(env_ids, torch.Tensor):
      n = env_ids.shape[0]
      self._counter[env_ids] = torch.randint(
        0, self.cfg.resample_interval_steps, (n,), device=self.device
      )
      self._arm_targets[env_ids] = self._sample(env_ids)
    else:
      n = self.num_envs
      self._counter[:] = torch.randint(
        0, self.cfg.resample_interval_steps, (n,), device=self.device
      )
      self._arm_targets[:] = self._sample(env_ids=None)
