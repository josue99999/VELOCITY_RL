from __future__ import annotations

import collections
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


_GK_WINDOW_SIZE = 200  # Rolling window size in completed episodes.
_GK_MIN_WINDOW = 100  # Minimum episodes required before evaluating conditions.


def _init_gk_state(env: ManagerBasedRlEnv, phases: dict) -> None:
  """Initialise metric-gated curriculum state on env (idempotent)."""
  if hasattr(env, "_gk_initialized"):
    return
  setattr(env, "_gk_phase_key", list(phases.keys())[0])
  setattr(env, "_gk_phase_episodes", 0)
  setattr(env, "_gk_fell_window", collections.deque(maxlen=_GK_WINDOW_SIZE))
  setattr(env, "_gk_ep_len_window", collections.deque(maxlen=_GK_WINDOW_SIZE))
  setattr(env, "_phase_frozen", True)  # Frozen until enough data is collected.
  setattr(env, "_episode_offset", 0.0)  # Kept for backward-compat logging only.
  setattr(env, "_gk_initialized", True)


def _can_advance_phase(
  phase_episodes: int,
  fell_window: collections.deque[float],
  ep_len_window: collections.deque[float],
  max_episode_length: int,
  phase: dict,
  metrics: dict | None,
) -> bool:
  """Return True when ALL phase-advancement conditions are satisfied.

  Conditions (all must hold):
  1. Minimum completed episodes in this phase (safety floor).
  2. Enough data in the rolling window for statistical significance.
  3. Rolling success rate (fraction without falls) >= threshold.
  4. Rolling mean episode-length ratio >= threshold.
  5. (Optional) Mean reward >= threshold, sourced from runner metrics.
  """
  # Condition 1: minimum episodes.
  if phase_episodes < phase.get("min_episodes_in_phase", 1000):
    return False

  # Condition 2: window has enough data.
  n = len(fell_window)
  if n < _GK_MIN_WINDOW:
    return False

  # Condition 3: rolling success rate.
  success_rate = 1.0 - sum(fell_window) / n
  if success_rate < phase.get("success_rate_min", 0.85):
    return False

  # Condition 4: rolling episode-length ratio.
  ep_len_ratio = (sum(ep_len_window) / n) / max(1, max_episode_length)
  if ep_len_ratio < phase.get("ep_length_ratio_min", 0.75):
    return False

  # Condition 5: optional reward threshold from runner.
  reward_threshold = phase.get("reward_threshold")
  if reward_threshold is not None and metrics is not None:
    mean_reward = metrics.get("mean_reward")
    if mean_reward is None or mean_reward < reward_threshold:
      return False

  return True


def gatekeeping_phase_control(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  phases: dict,
) -> dict[str, torch.Tensor]:
  """Metric-gated curriculum phase control.

  Phases advance only when ALL of the following conditions are simultaneously
  satisfied:
  - Minimum completed episodes in the current phase (safety floor so phase
    transitions cannot happen on statistical noise alone).
  - Rolling success rate (fraction of episodes without falls) >= threshold.
  - Rolling mean episode-length ratio (actual / max episode steps) >= threshold.
  - (Optional) Mean reward >= threshold, supplied by the runner each iteration.

  Episode-level data (fall flag, episode length) is read directly from
  ``env.reset_terminated[env_ids]`` and ``env.episode_length_buf[env_ids]``,
  which are guaranteed valid at the time curriculum terms are called (before
  episode buffers are zeroed). This removes any dependency on runner-provided
  metrics for the critical fall-rate gate.
  """
  _init_gk_state(env, phases)
  phase_names = list(phases.keys())

  fell_window: collections.deque[float] = getattr(env, "_gk_fell_window")
  ep_len_window: collections.deque[float] = getattr(env, "_gk_ep_len_window")
  phase_episodes: int = getattr(env, "_gk_phase_episodes")
  phase_key: str = getattr(env, "_gk_phase_key")

  # --- Update rolling windows from the batch of newly completed episodes. ---
  n_new = len(env_ids)
  if n_new > 0:
    # reset_terminated: True = terminated (fell), False = timed-out (survived).
    fell = env.reset_terminated[env_ids]
    ep_lens = env.episode_length_buf[env_ids].float()
    for i in range(n_new):
      fell_window.append(float(fell[i].item()))
      ep_len_window.append(float(ep_lens[i].item()))
    phase_episodes += n_new
    setattr(env, "_gk_phase_episodes", phase_episodes)

  # --- Check phase-advancement conditions. ---
  current_phase = phases[phase_key]
  current_idx = phase_names.index(phase_key)
  is_last = current_idx >= len(phase_names) - 1
  metrics = getattr(env, "_curriculum_metrics", None)
  can_advance = _can_advance_phase(
    phase_episodes,
    fell_window,
    ep_len_window,
    env.max_episode_length,
    current_phase,
    metrics,
  )

  if can_advance and not is_last:
    # Advance: reset per-phase counters and windows for the new phase.
    setattr(env, "_gk_phase_key", phase_names[current_idx + 1])
    setattr(env, "_gk_phase_episodes", 0)
    fell_window.clear()
    ep_len_window.clear()
    setattr(env, "_phase_frozen", False)
  else:
    setattr(env, "_phase_frozen", not can_advance and not is_last)

  # --- Compute rolling metrics for logging. ---
  n = len(fell_window)
  if n >= _GK_MIN_WINDOW:
    success_rate = 1.0 - sum(fell_window) / n
    ep_len_ratio = (sum(ep_len_window) / n) / max(1, env.max_episode_length)
  else:
    success_rate = 0.0
    ep_len_ratio = 0.0

  return {
    "phase_frozen": torch.tensor(
      1.0 if getattr(env, "_phase_frozen") else 0.0, dtype=torch.float32
    ),
    "gk_phase_index": torch.tensor(float(current_idx + 1), dtype=torch.float32),
    "gk_phase_episodes": torch.tensor(float(phase_episodes), dtype=torch.float32),
    "gk_success_rate": torch.tensor(success_rate, dtype=torch.float32),
    "gk_ep_len_ratio": torch.tensor(ep_len_ratio, dtype=torch.float32),
    "episode_offset": torch.tensor(
      float(getattr(env, "_episode_offset", 0.0)), dtype=torch.float32
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

  # Use metric-gated phase key when available (set by gatekeeping_phase_control,
  # which runs earlier in the curriculum dict).
  phase_names = list(phases.keys())
  current_phase_key: str = getattr(env, "_gk_phase_key", None) or phase_names[0]
  if current_phase_key not in phases:
    current_phase_key = phase_names[-1]
  phase_index = phase_names.index(current_phase_key)
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

  # Gatekeeping metrics from _gk_* state (set by gatekeeping_phase_control).
  _fell_w: collections.deque[float] = getattr(
    env, "_gk_fell_window", collections.deque()
  )
  n = len(_fell_w)
  if n >= _GK_MIN_WINDOW:
    _ep_len_w: collections.deque[float] = getattr(
      env, "_gk_ep_len_window", collections.deque()
    )
    gk_success_rate = 1.0 - sum(_fell_w) / n
    gk_ep_len_ratio = (sum(_ep_len_w) / n) / max(1, env.max_episode_length)
  else:
    gk_success_rate = 0.0
    gk_ep_len_ratio = 0.0

  # Success threshold for current phase (for reference in WandB).
  success_threshold = float(phase.get("success_rate_min", 0.85))
  min_episodes_threshold = float(phase.get("min_episodes_in_phase", 1000))

  return {
    "phase_index": torch.tensor(float(phase_index + 1)),
    "arm_randomization": torch.tensor(arm_rand),
    "arm_pose_range": torch.tensor(pose_range),
    "arm_mass_min": torch.tensor(mass_min),
    "arm_mass_max": torch.tensor(mass_max),
    "phase_frozen": torch.tensor(1.0 if getattr(env, "_phase_frozen", False) else 0.0),
    "gk_phase_episodes": torch.tensor(
      float(getattr(env, "_gk_phase_episodes", 0)), dtype=torch.float32
    ),
    "gk_success_rate": torch.tensor(gk_success_rate, dtype=torch.float32),
    "gk_ep_len_ratio": torch.tensor(gk_ep_len_ratio, dtype=torch.float32),
    "gk_success_threshold": torch.tensor(success_threshold, dtype=torch.float32),
    "gk_min_episodes_threshold": torch.tensor(
      min_episodes_threshold, dtype=torch.float32
    ),
    "arm_teleop_max_delta": torch.tensor(arm_teleop_max_delta),
    "arm_teleop_main_arm_scale": torch.tensor(arm_teleop_main_arm_scale),
    "arm_teleop_active": torch.tensor(1.0 if arm_teleop_max_delta > 0 else 0.0),
    "arm_teleop_last_delta_norm": torch.tensor(arm_log.get("delta_norm", 0.0)),
    "arm_teleop_last_delta_mean_abs": torch.tensor(arm_log.get("delta_mean_abs", 0.0)),
    "arm_teleop_last_pose_mean": torch.tensor(arm_log.get("pose_sample_mean", 0.0)),
    "arm_teleop_last_pose_std": torch.tensor(arm_log.get("pose_sample_std", 0.0)),
  }
