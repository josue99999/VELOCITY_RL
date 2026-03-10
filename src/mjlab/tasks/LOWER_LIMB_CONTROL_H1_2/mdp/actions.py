from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class NoisyJointPositionActionCfg(JointPositionActionCfg):
  """Joint position action with additive action noise and optional delay."""

  noise_std: float = 0.02
  delay_min_steps: int = 0
  delay_max_steps: int = 3

  def build(self, env: "ManagerBasedRlEnv") -> "NoisyJointPositionAction":
    return NoisyJointPositionAction(self, env)


class NoisyJointPositionAction(JointPositionAction):
  """Wrapper around JointPositionAction that injects bounded noise and delay."""

  def __init__(self, cfg: NoisyJointPositionActionCfg, env: "ManagerBasedRlEnv"):
    super().__init__(cfg, env)
    max_delay = max(int(cfg.delay_max_steps), 0)
    min_delay = max(int(cfg.delay_min_steps), 0)
    self._delay_min_steps = min_delay
    self._delay_max_steps = max_delay
    if max_delay > 0:
      buffer_len = max_delay + 1
      self._delay_buffer = torch.zeros(
        (self.num_envs, buffer_len, self.action_dim),
        device=self.device,
        dtype=torch.float32,
      )
      self._delay_current = torch.randint(
        low=min_delay,
        high=max_delay + 1,
        size=(self.num_envs,),
        device=self.device,
      )
    else:
      self._delay_buffer = None
      self._delay_current = None

  def _apply_delay(self, actions: torch.Tensor) -> torch.Tensor:
    if self._delay_buffer is None or self._delay_current is None:
      return actions
    # Roll buffer and insert newest action at index 0.
    self._delay_buffer = torch.roll(self._delay_buffer, shifts=1, dims=1)
    self._delay_buffer[:, 0, :] = actions
    # Select per-env delayed index.
    max_idx = self._delay_buffer.shape[1] - 1
    idx = self._delay_current.clamp(max=max_idx)
    env_indices = torch.arange(self.num_envs, device=self.device)
    delayed = self._delay_buffer[env_indices, idx]
    return delayed

  def process_actions(self, actions: torch.Tensor) -> None:
    # Uniform noise in [-noise_std, noise_std] to simulate actuator imperfections.
    cfg = cast(NoisyJointPositionActionCfg, self.cfg)
    if cfg.noise_std > 0.0:
      noise = (torch.rand_like(actions) * 2.0 - 1.0) * cfg.noise_std
      actions = actions + noise
    actions = self._apply_delay(actions)
    super().process_actions(actions)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids=env_ids)
    if self._delay_buffer is None:
      return
    if env_ids is None or isinstance(env_ids, slice):
      self._delay_buffer[:] = 0.0
      self._delay_current = torch.randint(
        low=self._delay_min_steps,
        high=self._delay_max_steps + 1,
        size=(self.num_envs,),
        device=self.device,
      )
    else:
      assert isinstance(env_ids, torch.Tensor)
      if self._delay_current is None:
        return
      self._delay_buffer[env_ids] = 0.0
      self._delay_current[env_ids] = torch.randint(
        low=self._delay_min_steps,
        high=self._delay_max_steps + 1,
        size=(env_ids.shape[0],),
        device=self.device,
      )
