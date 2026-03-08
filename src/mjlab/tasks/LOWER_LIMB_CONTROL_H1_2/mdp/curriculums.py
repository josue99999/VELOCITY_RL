from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .events import get_current_phase
from .velocity_command import UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


class VelocityStage(TypedDict):
  step: int
  lin_vel_x: tuple[float, float] | None
  lin_vel_y: tuple[float, float] | None
  ang_vel_z: tuple[float, float] | None


class RewardWeightStage(TypedDict):
  step: int
  weight: float


def terrain_levels_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]

  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  command = env.command_manager.get_command(command_name)
  assert command is not None

  # Compute the distance the robot walked.
  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
  )

  # Robots that walked far enough progress to harder terrains.
  move_up = distance > terrain_generator.size[0] / 2

  # Robots that walked less than half of their required distance go to simpler
  # terrains.
  move_down = (
    distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
  )
  move_down *= ~move_up

  # Update terrain levels.
  terrain.update_env_origins(env_ids, move_up, move_down)

  return torch.mean(terrain.terrain_levels.float())


def commands_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor]:
  del env_ids  # Unused.
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
  for stage in velocity_stages:
    if env.common_step_counter > stage["step"]:
      if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
        cfg.ranges.lin_vel_x = stage["lin_vel_x"]
      if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
        cfg.ranges.lin_vel_y = stage["lin_vel_y"]
      if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
        cfg.ranges.ang_vel_z = stage["ang_vel_z"]
  return {
    "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
    "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
    "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]),
    "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),
    "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]),
    "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),
  }


def _get_current_phase_key(env: ManagerBasedRlEnv, phases: dict) -> str:
  """Return current phase key (name) for threshold lookup."""
  episodes = env.common_step_counter / env.max_episode_length
  offset = getattr(env, "_episode_offset", 0)
  effective_episodes = episodes - offset
  for phase_name, phase_info in phases.items():
    min_ep, max_ep = phase_info["episode_range"]
    if min_ep <= effective_episodes < max_ep:
      return phase_name
  return list(phases.keys())[-1]


# Thresholds (reward_threshold, fell_over_threshold) per phase for gatekeeping.
_PHASE_THRESHOLDS: dict[str, tuple[float, float]] = {
  "light_disturbances": (45.0, 0.12),
  "moderate_disturbances": (48.0, 0.15),
  "full_robustness": (50.0, 0.18),
}
_DEFAULT_THRESHOLDS = (45.0, 0.15)


def update_episode_offset(
  env: ManagerBasedRlEnv,
  metrics: dict | None,
  phases: dict,
  offset_increment: float = 10.0,
  max_offset: float = 500.0,
) -> None:
  """Gatekeeping: increase episode_offset when metrics are bad so phase advance slows.
  Sets env._phase_frozen for WandB logging. Call from runner with env._curriculum_metrics.
  """
  if metrics is None:
    setattr(env, "_phase_frozen", False)
    return

  phase_key = _get_current_phase_key(env, phases)
  reward_threshold, fell_over_threshold = _PHASE_THRESHOLDS.get(
    phase_key, _DEFAULT_THRESHOLDS
  )

  mean_reward = metrics.get("mean_reward")
  value_loss = metrics.get("value_loss")
  fell_over_rate = metrics.get("fell_over_rate")

  bad = False
  if mean_reward is not None and mean_reward < reward_threshold:
    bad = True
  if value_loss is not None:
    v = value_loss
    if isinstance(v, torch.Tensor):
      if torch.isnan(v).any() or torch.isinf(v).any():
        bad = True
    elif v != v or v == float("inf"):
      bad = True
  if fell_over_rate is not None and fell_over_rate > fell_over_threshold:
    bad = True

  if bad:
    offset = getattr(env, "_episode_offset", 0)
    setattr(env, "_episode_offset", min(offset + offset_increment, max_offset))
    setattr(env, "_phase_frozen", True)
  else:
    setattr(env, "_phase_frozen", False)


def gatekeeping_phase_control(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  phases: dict,
) -> dict[str, torch.Tensor]:
  """Curriculum term that updates episode offset based on metrics.
  Expects env._curriculum_metrics to be set by the runner after each iteration.
  """
  metrics = getattr(env, "_curriculum_metrics", None)
  update_episode_offset(env, metrics, phases)
  return {
    "episode_offset": torch.tensor(
      float(getattr(env, "_episode_offset", 0)), dtype=torch.float32
    ),
    "phase_frozen": torch.tensor(
      1.0 if getattr(env, "_phase_frozen", False) else 0.0,
      dtype=torch.float32,
    ),
  }


def update_teleop_pushes(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  phases: dict,
  push_event_name: str = "push_robot",
) -> dict[str, torch.Tensor]:
  """Update push event parameters based on curriculum phases.
  Stress-aware: reduce push when robot already oscillating (high ang_vel).
  """
  del env_ids  # Unused.
  phase = get_current_phase(env, phases)

  push_vel = phase.get("push_velocity", 0.0)
  push_interval = phase.get("push_interval", (1.0, 3.0))

  # Stress-aware scaling: reduce push when robot already oscillating (ang_vel high).
  if push_vel > 0:
    asset = env.scene["robot"]
    ang_vel_mag = torch.norm(asset.data.root_link_ang_vel_w, dim=-1)
    stress_factor = torch.clamp(1.0 - (ang_vel_mag / 5.0), 0.3, 1.0)
    effective_push_vel = push_vel * stress_factor.mean().item()
  else:
    effective_push_vel = 0.0

  # Get the push event config and scale all velocity components.
  try:
    event_cfg = env.event_manager.get_term_cfg(push_event_name)

    if "velocity_range" in event_cfg.params:
      v_range = event_cfg.params["velocity_range"]
      # Base scale: 0.5 for x/y; scale z, roll, pitch, yaw proportionally.
      scale = effective_push_vel / 0.5 if push_vel > 0 else 0.0
      v_range["x"] = (-effective_push_vel, effective_push_vel)
      v_range["y"] = (-effective_push_vel, effective_push_vel)
      v_range["z"] = (-0.4 * scale, 0.4 * scale)
      v_range["roll"] = (-0.52 * scale, 0.52 * scale)
      v_range["pitch"] = (-0.52 * scale, 0.52 * scale)
      v_range["yaw"] = (-0.78 * scale, 0.78 * scale)

    if isinstance(push_interval, (tuple, list)):
      event_cfg.interval_range_s = push_interval
    elif isinstance(push_interval, (float, int)):
      event_cfg.interval_range_s = (push_interval * 0.8, push_interval * 1.2)

  except ValueError:
    pass

  return {
    "push_velocity": torch.tensor(effective_push_vel),
    "is_pushing": torch.tensor(1.0 if effective_push_vel > 0 else 0.0),
  }


def update_arm_teleop_continuous(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  phases: dict,
  event_name: str = "arm_pose_continuous_teleop",
) -> dict[str, torch.Tensor]:
  """Update arm continuous teleop event params from curriculum phase (for training)."""
  del env_ids  # Unused.
  phase = get_current_phase(env, phases)
  max_delta = phase.get("arm_teleop_max_delta", 0.0)
  main_arm_scale = phase.get("arm_teleop_main_arm_scale", 1.0)
  interval_range_s = phase.get("arm_teleop_interval_range_s", (0.02, 0.02))
  try:
    event_cfg = env.event_manager.get_term_cfg(event_name)
    event_cfg.params["max_delta"] = max_delta
    event_cfg.params["main_arm_scale"] = main_arm_scale
    if isinstance(interval_range_s, (tuple, list)) and len(interval_range_s) >= 2:
      event_cfg.interval_range_s = (interval_range_s[0], interval_range_s[1])
  except ValueError:
    pass
  return {
    "arm_teleop_max_delta": torch.tensor(max_delta),
    "arm_teleop_main_arm_scale": torch.tensor(main_arm_scale),
    "arm_teleop_active": torch.tensor(1.0 if max_delta > 0 else 0.0),
  }


def log_phase_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  phases: dict,
) -> dict[str, torch.Tensor]:
  """Return current phase metrics for WandB (arm pose, mass, phase index, gatekeeping)."""
  del env_ids  # Unused.
  episodes = env.common_step_counter / env.max_episode_length
  offset = getattr(env, "_episode_offset", 0)
  effective_episodes = episodes - offset

  phase_names = list(phases.keys())
  current_phase_key = None
  phase_index = 0
  for i, phase_name in enumerate(phase_names):
    min_ep, max_ep = phases[phase_name]["episode_range"]
    if min_ep <= effective_episodes < max_ep:
      current_phase_key = phase_name
      phase_index = i
      break
  if current_phase_key is None:
    phase_index = len(phase_names) - 1
    current_phase_key = phase_names[-1]
  phase = phases[current_phase_key]

  arm_rand = 1.0 if phase.get("arm_randomization", False) else 0.0
  pose_range = float(phase.get("arm_pose_range", 0.0))
  mass_range = phase.get("arm_mass_range")
  if mass_range is not None:
    mass_min, mass_max = float(mass_range[0]), float(mass_range[1])
  else:
    mass_min, mass_max = 0.0, 0.0
  arm_teleop_max_delta = float(phase.get("arm_teleop_max_delta", 0.0))
  arm_teleop_main_arm_scale = float(phase.get("arm_teleop_main_arm_scale", 1.0))
  arm_log = getattr(env, "_arm_teleop_log", {})
  delta_norm = arm_log.get("delta_norm", 0.0)
  delta_mean_abs = arm_log.get("delta_mean_abs", 0.0)
  pose_sample_mean = arm_log.get("pose_sample_mean", 0.0)
  pose_sample_std = arm_log.get("pose_sample_std", 0.0)

  return {
    "phase_index": torch.tensor(phase_index + 1),
    "arm_randomization": torch.tensor(arm_rand),
    "arm_pose_range": torch.tensor(pose_range),
    "arm_mass_min": torch.tensor(mass_min),
    "arm_mass_max": torch.tensor(mass_max),
    "episodes_approx": torch.tensor(episodes),
    "episodes_effective": torch.tensor(effective_episodes),
    "phase_frozen": torch.tensor(1.0 if getattr(env, "_phase_frozen", False) else 0.0),
    "episode_offset": torch.tensor(
      float(getattr(env, "_episode_offset", 0)),
      dtype=torch.float32,
    ),
    "arm_teleop_max_delta": torch.tensor(arm_teleop_max_delta),
    "arm_teleop_main_arm_scale": torch.tensor(arm_teleop_main_arm_scale),
    "arm_teleop_active": torch.tensor(1.0 if arm_teleop_max_delta > 0 else 0.0),
    "arm_teleop_last_delta_norm": torch.tensor(delta_norm),
    "arm_teleop_last_delta_mean_abs": torch.tensor(delta_mean_abs),
    "arm_teleop_last_pose_mean": torch.tensor(pose_sample_mean),
    "arm_teleop_last_pose_std": torch.tensor(pose_sample_std),
  }
