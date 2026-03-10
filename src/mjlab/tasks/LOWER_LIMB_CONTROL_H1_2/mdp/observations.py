from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def foot_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # (num_envs, num_sites)


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  # Clamp to 1 s: beyond that the foot is falling, not swinging.
  # Prevents unbounded critic input during jumps or falls.
  return torch.clamp(current_air_time, max=1.0)


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


def gait_phase_clock(
  env: ManagerBasedRlEnv,
  cycle_freq_hz: float = 1.5,
) -> torch.Tensor:
  """Sinusoidal gait phase clock for periodic locomotion.

  Returns a (num_envs, 2) tensor of (sin(phase), cos(phase)) where phase is
  derived from elapsed episode time at a fixed stride frequency. The clock
  resets with each episode, giving the policy a consistent periodic reference
  to synchronize footstep timing without needing an explicit RNN.

  The sin/cos encoding is bounded in [-1, 1] and smooth across phase wrapping.
  Typical walking cadence for H1_2 is ~1.5 Hz (one full stride per ~0.67 s).
  """
  t = env.episode_length_buf.float() * env.step_dt  # elapsed time [num_envs]
  phase = 2.0 * math.pi * cycle_freq_hz * t  # [num_envs]
  return torch.stack([torch.sin(phase), torch.cos(phase)], dim=-1)  # [num_envs, 2]
