import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.events import randomize_field
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg


def get_current_phase(env: ManagerBasedRlEnv, phases: dict) -> dict:
  """Helper to determine the current curriculum phase based on episode count.
  Uses episode_offset to slow phase advance when metrics are bad (gatekeeping).
  """
  episodes = env.common_step_counter / env.max_episode_length
  offset = getattr(env, "_episode_offset", 0)
  effective_episodes = episodes - offset

  current_phase_key = None
  for phase_name, phase_info in phases.items():
    min_ep, max_ep = phase_info["episode_range"]
    if min_ep <= effective_episodes < max_ep:
      current_phase_key = phase_name
      break

  if current_phase_key is None:
    current_phase_key = list(phases.keys())[-1]

  return phases[current_phase_key]


def randomize_arm_pose_phase_based(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  phases: dict,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
  """Randomize the arm joint positions at reset according to the current phase.

  Called once per environment per reset (i.e. at the start of each new episode).
  Only the env_ids that are reset in this step get new random arm poses.
  """
  phase = get_current_phase(env, phases)

  if not phase.get("arm_randomization", False):
    return

  pose_range = phase.get("arm_pose_range", 0.0)
  if pose_range <= 0.0:
    return

  asset = env.scene[asset_cfg.name]

  joint_ids, _ = asset.find_joints(asset_cfg.joint_names)
  if len(joint_ids) == 0:
    return

  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None
  soft_joint_pos_limits = asset.data.soft_joint_pos_limits
  assert soft_joint_pos_limits is not None

  joint_pos = default_joint_pos[env_ids][:, joint_ids].clone()
  offsets = (torch.rand_like(joint_pos, device=env.device) * 2.0 - 1.0) * pose_range
  new_joint_pos = joint_pos + offsets
  pos_limits = soft_joint_pos_limits[env_ids][:, joint_ids]
  new_joint_pos = new_joint_pos.clamp_(pos_limits[..., 0], pos_limits[..., 1])

  joint_ids_tensor = torch.tensor(joint_ids, device=env.device, dtype=torch.long)
  velocity = torch.zeros_like(new_joint_pos, device=env.device)
  asset.write_joint_state_to_sim(
    new_joint_pos, velocity, joint_ids=joint_ids_tensor, env_ids=env_ids
  )
  asset.set_joint_position_target(
    new_joint_pos, joint_ids=joint_ids_tensor, env_ids=env_ids
  )


def arm_pose_continuous_teleop(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  asset_cfg: SceneEntityCfg,
  max_delta: float = 0.015,
  main_arm_scale: float = 3.0,
) -> None:
  """Apply a small random delta to current arm joint positions (continuous motion).

  Call this every step (or at short intervals) so arms move smoothly as if
  teleoperated. Shoulder and elbow joints (main 2 DOF) get main_arm_scale x more range.
  """
  if len(env_ids) == 0:
    return
  asset = env.scene[asset_cfg.name]
  joint_ids, joint_names = asset.find_joints(asset_cfg.joint_names)
  if len(joint_ids) == 0:
    return
  soft_joint_pos_limits = asset.data.soft_joint_pos_limits
  assert soft_joint_pos_limits is not None
  # Read current positions (from sim state), not default.
  joint_pos = asset.data.joint_pos[env_ids][:, joint_ids].clone()
  # Gaussian deltas (more continuous); clip to avoid rare large jumps.
  delta = torch.randn_like(joint_pos, device=env.device) * (max_delta * 0.5)
  delta = delta.clamp_(-max_delta, max_delta)
  # Scale up the two main arm DOF (shoulder and elbow) for more angular motion.
  for i, name in enumerate(joint_names):
    if "shoulder" in name or "elbow" in name:
      delta[:, i] *= main_arm_scale
  new_joint_pos = (joint_pos + delta).clamp_(
    soft_joint_pos_limits[env_ids][:, joint_ids, 0],
    soft_joint_pos_limits[env_ids][:, joint_ids, 1],
  )
  joint_ids_tensor = torch.tensor(joint_ids, device=env.device, dtype=torch.long)
  velocity = torch.zeros_like(new_joint_pos, device=env.device)
  asset.write_joint_state_to_sim(
    new_joint_pos, velocity, joint_ids=joint_ids_tensor, env_ids=env_ids
  )
  # Sync PD target so actuators hold this pose (play-only event; no fight to 0).
  asset.set_joint_position_target(
    new_joint_pos, joint_ids=joint_ids_tensor, env_ids=env_ids
  )
  # Store stats for WandB: verify deltas/poses are random and changing.
  pose0 = new_joint_pos[0]
  setattr(
    env,
    "_arm_teleop_log",
    {
      "delta_norm": delta.norm().item(),
      "delta_mean_abs": delta.abs().mean().item(),
      "pose_sample_mean": pose0.mean().item(),
      "pose_sample_std": pose0.std().item() if pose0.numel() > 1 else 0.0,
    },
  )


@requires_model_fields("body_mass")
def randomize_arm_mass_phase_based(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  phases: dict,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
  """Randomize the mass of the arms at reset according to the current phase."""
  phase = get_current_phase(env, phases)

  mass_range = phase.get("arm_mass_range", None)
  if mass_range is None or mass_range[0] == mass_range[1] == 0.0:
    return

  asset = env.scene[asset_cfg.name]

  body_ids_local, _ = asset.find_bodies(asset_cfg.body_names)
  if len(body_ids_local) == 0:
    return

  body_ids_tensor = torch.tensor(body_ids_local, device=env.device, dtype=torch.long)
  global_body_ids = asset.data.indexing.body_ids[body_ids_tensor]
  nominal_mass = env.sim.model.body_mass[0, global_body_ids].float().to(env.device)

  min_mult, max_mult = mass_range
  multipliers = (
    torch.rand((len(env_ids), 1), device=env.device) * (max_mult - min_mult) + min_mult
  )
  new_mass = nominal_mass.unsqueeze(0) * multipliers
  env.sim.model.body_mass[env_ids.unsqueeze(-1), global_body_ids] = new_mass


@requires_model_fields("body_mass")
def randomize_link_masses(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  mass_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
  """Domain randomization of link masses via unified randomize_field helper.

  Applies a multiplicative factor sampled uniformly from ``mass_range`` to the
  nominal body masses for the specified asset/bodies. Safe to call on every reset
  thanks to use of stored default fields inside ``randomize_field``.
  """
  randomize_field(
    env,
    env_ids,
    field="body_mass",
    ranges=mass_range,
    distribution="uniform",
    operation="scale",
    asset_cfg=asset_cfg,
  )


@requires_model_fields("body_inertia")
def randomize_link_inertia(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  inertia_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
  """Domain randomization of link inertias via unified randomize_field helper.

  Applies a multiplicative factor sampled uniformly from ``inertia_range`` to the
  nominal body inertia tensors for the specified asset/bodies.
  """
  randomize_field(
    env,
    env_ids,
    field="body_inertia",
    ranges=inertia_range,
    distribution="uniform",
    operation="scale",
    asset_cfg=asset_cfg,
  )
