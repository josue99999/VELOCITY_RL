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
  result = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
  return torch.nan_to_num(result, nan=0.0)


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return torch.clamp(torch.nan_to_num(current_air_time, nan=0.0), max=1.0)


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
  forces_flat = torch.nan_to_num(forces_flat, nan=0.0)
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
  t = env.episode_length_buf.float() * env.step_dt
  phase = 2.0 * math.pi * cycle_freq_hz * t
  return torch.stack([torch.sin(phase), torch.cos(phase)], dim=-1)
