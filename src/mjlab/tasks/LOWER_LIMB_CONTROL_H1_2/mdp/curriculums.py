from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

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


def get_current_phase(env: ManagerBasedRlEnv, phases: dict) -> dict:
  """Helper to determine the current curriculum phase based on episode count."""
  # Convert global steps to estimated episodes
  episodes = env.common_step_counter / env.max_episode_length

  current_phase_key = None
  for phase_name, phase_info in phases.items():
    min_ep, max_ep = phase_info["episode_range"]
    if min_ep <= episodes < max_ep:
      current_phase_key = phase_name
      break

  # Default to the last phase if we exceed all ranges
  if current_phase_key is None:
    current_phase_key = list(phases.keys())[-1]

  return phases[current_phase_key]


def update_teleop_pushes(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  phases: dict,
  push_event_name: str = "push_robot",
) -> dict[str, torch.Tensor]:
  """Update push event parameters based on curriculum phases."""
  del env_ids  # Unused.
  phase = get_current_phase(env, phases)

  push_vel = phase.get("push_velocity", 0.0)
  push_interval = phase.get("push_interval", (1.0, 3.0))

  # Get the push event config
  try:
    event_cfg = env.event_manager.get_term_cfg(push_event_name)

    # Update physical parameters in the config
    # This affects future triggers of the 'interval' mode event
    if "velocity_range" in event_cfg.params:
      v_range = event_cfg.params["velocity_range"]
      # Update x, y, z ranges based on push_vel
      # We preserve the signs/directions from the original config
      v_range["x"] = (-push_vel, push_vel)
      v_range["y"] = (-push_vel, push_vel)
      # We can also scale others if desired, but x/y are most critical for stability

    # Update interval if specified
    if isinstance(push_interval, (tuple, list)):
      event_cfg.interval_range_s = push_interval
    elif isinstance(push_interval, (float, int)):
      # If it's a single number, we create a small range around it
      event_cfg.interval_range_s = (push_interval * 0.8, push_interval * 1.2)

  except ValueError:
    # Push event might not exist in this specific robot config
    pass

  return {
    "push_velocity": torch.tensor(push_vel),
    "is_pushing": torch.tensor(1.0 if push_vel > 0 else 0.0),
  }


def log_phase_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  phases: dict,
) -> dict[str, torch.Tensor]:
  """Return current phase metrics for WandB (arm pose, mass, phase index)."""
  del env_ids  # Unused.
  episodes = env.common_step_counter / env.max_episode_length
  phase_names = list(phases.keys())
  current_phase_key = None
  phase_index = 0
  for i, phase_name in enumerate(phase_names):
    min_ep, max_ep = phases[phase_name]["episode_range"]
    if min_ep <= episodes < max_ep:
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

  # phase_index 1-based (1..4) so "fase 4" shows as 4 in logs/WandB.
  return {
    "phase_index": torch.tensor(phase_index + 1),
    "arm_randomization": torch.tensor(arm_rand),
    "arm_pose_range": torch.tensor(pose_range),
    "arm_mass_min": torch.tensor(mass_min),
    "arm_mass_max": torch.tensor(mass_max),
    "episodes_approx": torch.tensor(episodes),
  }
