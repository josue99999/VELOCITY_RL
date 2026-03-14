from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _sanitize(t: torch.Tensor, fallback: float = 0.0) -> torch.Tensor:
  """Replace NaN/Inf with *fallback* so a single unstable env cannot corrupt the batch."""
  return torch.nan_to_num(t, nan=fallback, posinf=fallback, neginf=fallback)


def joint_torques_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize squared L2 norm of joint torques for energy efficiency."""
  asset: Entity = env.scene[asset_cfg.name]
  torques = asset.data.actuator_force[:, asset_cfg.actuator_ids]
  return _sanitize(torch.sum(torch.square(torques), dim=1))


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the commanded base linear velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  lin_vel_error = _sanitize(xy_error + z_error)
  return torch.exp(-lin_vel_error / std**2)


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward heading error for heading-controlled envs, angular velocity for others."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  ang_vel_error = _sanitize(z_error + xy_error)
  return torch.exp(-ang_vel_error / std**2)


def flat_orientation(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward flat base orientation (robot being upright)."""
  asset: Entity = env.scene[asset_cfg.name]

  if asset_cfg.body_ids:
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]
    if body_quat_w.dim() == 3 and body_quat_w.shape[1] > 1:
      body_quat_w = body_quat_w[:, 0, :]
    else:
      body_quat_w = body_quat_w.squeeze(1)
    gravity_w = asset.data.gravity_vec_w
    projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)
    xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
  else:
    xy_squared = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
  return torch.exp(-_sanitize(xy_squared) / std**2)


def self_collision_cost(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Penalize self-collisions."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return _sanitize(sensor.data.found.squeeze(-1).float())


def body_angular_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize excessive body angular velocities."""
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
  ang_vel = _sanitize(ang_vel.squeeze(1))
  ang_vel_xy = ang_vel[:, :2]
  return torch.sum(torch.square(ang_vel_xy), dim=1)


def angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  max_magnitude: float = 100.0,
) -> torch.Tensor:
  """Penalize whole-body angular momentum."""
  angmom_sensor: BuiltinSensor = env.scene[sensor_name]
  angmom = angmom_sensor.data
  angmom = _sanitize(angmom)
  angmom = torch.clamp(angmom, min=-1000.0, max=1000.0)
  angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
  angmom_magnitude_sq = torch.clamp(angmom_magnitude_sq, min=0.0, max=max_magnitude**2)
  angmom_magnitude = torch.sqrt(angmom_magnitude_sq + 1e-8)
  _mean = torch.nanmean(angmom_magnitude).nan_to_num(nan=0.0)
  env.extras["log"]["Metrics/angular_momentum_mean"] = float(
    torch.clamp(_mean, max=500.0).item()
  )
  angmom_magnitude = torch.clamp(angmom_magnitude, max=max_magnitude)
  return angmom_magnitude * angmom_magnitude


def feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold_min: float = 0.05,
  threshold_max: float = 0.5,
  command_name: str | None = None,
  command_threshold: float = 0.5,
) -> torch.Tensor:
  """Reward feet air time."""
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  current_air_time = _sanitize(current_air_time)
  in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
  reward = torch.sum(in_range.float(), dim=1)
  in_air = current_air_time > 0
  num_in_air = torch.sum(in_air.float())
  mean_air_time = torch.sum(current_air_time * in_air.float()) / torch.clamp(
    num_in_air, min=1
  )
  env.extras["log"]["Metrics/air_time_mean"] = mean_air_time.nan_to_num(nan=0.0)
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      scale = (total_command > command_threshold).float()
      reward *= scale
  return reward


def feet_clearance(
  env: ManagerBasedRlEnv,
  target_height: float,
  command_name: str | None = None,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation from target clearance height, weighted by foot velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  foot_z = _sanitize(asset.data.site_pos_w[:, asset_cfg.site_ids, 2])
  foot_vel_xy = _sanitize(asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2])
  vel_norm = torch.norm(foot_vel_xy, dim=-1)
  delta = torch.abs(foot_z - target_height)
  cost = torch.sum(delta * vel_norm, dim=1)
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


def feet_clearance_improved(
  env: ManagerBasedRlEnv,
  target_height: float = 0.08,
  min_height: float = 0.03,
  command_name: str | None = None,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  sensor_name: str = "feet_ground_contact",
) -> torch.Tensor:
  """Improved foot clearance reward focusing on swing phase and dragging."""
  asset: Entity = env.scene[asset_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]

  foot_z = _sanitize(asset.data.site_pos_w[:, asset_cfg.site_ids, 2])
  foot_vel_xy = _sanitize(asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2])
  vel_norm = torch.norm(foot_vel_xy, dim=-1)

  assert sensor.data.found is not None
  in_contact = sensor.data.found > 0
  is_swing = (~in_contact) & (vel_norm > 0.1)

  delta = torch.abs(foot_z - target_height)
  clearance_penalty = torch.where(is_swing, delta * vel_norm, torch.zeros_like(delta))

  too_low = (foot_z < min_height) & is_swing
  dragging_penalty = torch.where(
    too_low, (min_height - foot_z) * 10.0, torch.zeros_like(foot_z)
  )

  total = torch.sum(clearance_penalty + dragging_penalty, dim=1)

  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      active = (linear_norm > command_threshold).float()
      total = total * active

  return total


class feet_swing_height:
  """Penalize deviation from target swing height, evaluated at landing."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.sensor_name = cfg.params["sensor_name"]
    self.site_names = cfg.params["asset_cfg"].site_names
    self.peak_heights = torch.zeros(
      (env.num_envs, len(self.site_names)),
      device=env.device,
      dtype=torch.float32,
    )
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    target_height: float,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    foot_heights = _sanitize(asset.data.site_pos_w[:, asset_cfg.site_ids, 2])
    in_air = contact_sensor.data.found == 0
    self.peak_heights = torch.where(
      in_air,
      torch.maximum(self.peak_heights, foot_heights),
      self.peak_heights,
    )
    # Prevent latent NaN from persisting across steps.
    self.peak_heights = _sanitize(self.peak_heights)
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()
    error = self.peak_heights / target_height - 1.0
    cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active
    num_landings = torch.sum(first_contact.float())
    peak_heights_at_landing = self.peak_heights * first_contact.float()
    mean_peak_height = torch.sum(peak_heights_at_landing) / torch.clamp(
      num_landings, min=1
    )
    env.extras["log"]["Metrics/peak_height_mean"] = mean_peak_height.nan_to_num(nan=0.0)
    self.peak_heights = torch.where(
      first_contact,
      torch.zeros_like(self.peak_heights),
      self.peak_heights,
    )
    return _sanitize(cost)

  def reset(self, env_ids: torch.Tensor) -> None:
    self.peak_heights[env_ids] = 0.0


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot sliding (xy velocity while in contact)."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  active = (total_command > command_threshold).float()
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()
  foot_vel_xy = _sanitize(asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2])
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)
  vel_xy_norm_sq = torch.square(vel_xy_norm)
  cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
  num_in_contact = torch.sum(in_contact)
  mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
    num_in_contact, min=1
  )
  env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel.nan_to_num(nan=0.0)
  return _sanitize(cost)


class single_foot_contact:
  """Reward alternating single-foot contacts to discourage hopping."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.sensor_name: str = cfg.params["sensor_name"]
    self.command_name: str = cfg.params["command_name"]
    self.command_threshold: float = cfg.params.get("command_threshold", 0.1)
    grace_period: float = cfg.params.get("grace_period", 0.2)
    buffer_len = max(1, int(grace_period / env.step_dt))
    self.buffer = torch.zeros(
      (env.num_envs, buffer_len),
      device=env.device,
      dtype=torch.float32,
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    command_threshold: float,
    grace_period: float,
  ) -> torch.Tensor:
    del grace_period

    sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    linear_norm = torch.norm(command[:, :2], dim=1)
    active = (linear_norm > command_threshold).float()

    assert sensor.data.found is not None
    contact = sensor.data.found
    num_feet_in_contact = torch.sum(contact > 0, dim=1).float()

    single_contact = (num_feet_in_contact == 1).float()

    self.buffer = torch.roll(self.buffer, shifts=-1, dims=1)
    self.buffer[:, -1] = single_contact

    had_single_contact = (torch.max(self.buffer, dim=1).values > 0).float()
    return had_single_contact * active

  def reset(self, env_ids: torch.Tensor) -> None:
    self.buffer[env_ids] = 0.0


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize high impact forces at landing to encourage soft footfalls."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = _sanitize(sensor_data.force)
  force_magnitude = torch.norm(forces, dim=-1)
  force_magnitude = torch.clamp(force_magnitude, max=10000.0)
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)
  landing_impact = force_magnitude * first_contact.float()
  cost = torch.sum(landing_impact, dim=1)
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force.nan_to_num(
    nan=0.0
  )
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


def survival_bonus(
  env: ManagerBasedRlEnv,
  bonus_per_step: float = 1.0,
) -> torch.Tensor:
  """Constant per-step bonus for staying alive (not terminated)."""
  return (
    torch.ones(env.num_envs, device=env.device, dtype=torch.float32) * bonus_per_step
  )


class variable_posture:
  """Penalize deviation from default pose with speed-dependent tolerance."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

    _, _, std_standing = resolve_matching_names_values(
      data=cfg.params["std_standing"],
      list_of_strings=joint_names,
    )
    self.std_standing = torch.tensor(
      std_standing, device=env.device, dtype=torch.float32
    )

    _, _, std_walking = resolve_matching_names_values(
      data=cfg.params["std_walking"],
      list_of_strings=joint_names,
    )
    self.std_walking = torch.tensor(std_walking, device=env.device, dtype=torch.float32)

    _, _, std_running = resolve_matching_names_values(
      data=cfg.params["std_running"],
      list_of_strings=joint_names,
    )
    self.std_running = torch.tensor(std_running, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    std_walking,
    std_running,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_running

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    cmd = _sanitize(command)
    linear_speed = torch.norm(cmd[:, :2], dim=1)
    angular_speed = torch.abs(cmd[:, 2])
    total_speed = linear_speed + angular_speed

    standing_mask = (total_speed < walking_threshold).float()
    walking_mask = (
      (total_speed >= walking_threshold) & (total_speed < running_threshold)
    ).float()
    running_mask = (total_speed >= running_threshold).float()

    std = (
      self.std_standing * standing_mask.unsqueeze(1)
      + self.std_walking * walking_mask.unsqueeze(1)
      + self.std_running * running_mask.unsqueeze(1)
    )

    current_joint_pos = _sanitize(asset.data.joint_pos[:, asset_cfg.joint_ids])
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)

    return torch.exp(-torch.mean(error_squared / (std**2 + 1e-8), dim=1))


def stable_upright_under_disturbance(
  env: ManagerBasedRlEnv,
  phases: dict,
  upright_std: float = 0.25,
  upright_threshold: float = 0.95,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Bonus when upright under active disturbances."""
  from mjlab.tasks.LOWER_LIMB_CONTROL_H1_2.mdp.events import (
    get_current_phase,
  )

  phase = get_current_phase(env, phases)
  if phase.get("push_velocity", 0) <= 0:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

  upright = flat_orientation(env, std=upright_std, asset_cfg=asset_cfg)
  return (upright > upright_threshold).float()


def zmp_stability_reward(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  support_polygon_margin: float = 0.05,
) -> torch.Tensor:
  """Heuristic COM-based stability reward (ZMP-inspired)."""
  asset: Entity = env.scene[asset_cfg.name]
  com_pos = _sanitize(asset.data.root_com_pos_w)
  base_pos = _sanitize(asset.data.root_link_pos_w)

  com_offset_xy = com_pos[:, :2] - base_pos[:, :2]
  com_dist = torch.norm(com_offset_xy, dim=1)

  length_scale = max(1e-3, 2.0 * support_polygon_margin)
  return torch.exp(-com_dist / length_scale)


def base_height_penalty(
  env: ManagerBasedRlEnv,
  target_height: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation of COM height from a target value above the terrain."""
  asset: Entity = env.scene[asset_cfg.name]
  com_pos = _sanitize(asset.data.root_com_pos_w)
  terrain_z = env.scene.env_origins[:, 2]
  height_above_terrain = com_pos[:, 2] - terrain_z
  height_error = height_above_terrain - target_height
  return torch.square(height_error)


def contact_force_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 800.0,
) -> torch.Tensor:
  """Penalize contact forces above a threshold for softer contacts."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = _sanitize(sensor_data.force)
  force_magnitude = torch.norm(forces, dim=-1)
  excess = torch.clamp(force_magnitude - force_threshold, min=0.0)
  return torch.sum(excess, dim=1)


def stance_contact_reward(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Reward appropriate stance contact."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()
  num_in_contact = torch.sum(in_contact, dim=1)
  stance_score = torch.clamp(num_in_contact, min=0.0, max=2.0)

  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      active = (linear_norm > command_threshold).float()
      stance_score = stance_score * active

  return stance_score
