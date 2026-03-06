import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.event_manager import requires_model_fields


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
