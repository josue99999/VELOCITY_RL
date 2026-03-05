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
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> None:
    """Randomize the arm positions at reset according to the current phase."""
    phase = get_current_phase(env, phases)
    
    if not phase.get("arm_randomization", False):
        return
        
    pose_range = phase.get("arm_pose_range", 0.0)
    if pose_range <= 0.0:
        return
        
    asset = env.scene[asset_cfg.name]
    
    # We want to identify the arm joints. Assuming standard naming convention like "*_arm_*" or similar.
    # Alternatively, any joint NOT in the CONTROLLED_JOINTS could be treated as an arm joint.
    # We will let the config pass the specific joint names in asset_cfg.joint_names
    
    # Getting the joint indices
    joint_ids = asset.joints(asset_cfg.joint_names)
    if len(joint_ids) == 0:
        return
        
    # Get current joint positions
    joint_pos = asset.data.qpos[env_ids][:, asset.qpos_joint_indices[joint_ids]]
    
    # Sample random offsets uniformly in [-pose_range, pose_range]
    offsets = (torch.rand_like(joint_pos) * 2.0 - 1.0) * pose_range
    
    # Set the new positions
    new_joint_pos = joint_pos + offsets
    
    # TODO: Respect joint limits if needed, or simply let the physics engine clamp it.
    
    asset.write_joint_state_to_sim(
        position=new_joint_pos, 
        joint_ids=joint_ids, 
        env_ids=env_ids
    )

@requires_model_fields("body_mass")
def randomize_arm_mass_phase_based(
    env: ManagerBasedRlEnv, 
    env_ids: torch.Tensor, 
    phases: dict,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> None:
    """Randomize the mass of the arms at reset according to the current phase."""
    phase = get_current_phase(env, phases)
    
    mass_range = phase.get("arm_mass_range", None)
    if mass_range is None or mass_range[0] == mass_range[1] == 0.0:
        return
        
    asset = env.scene[asset_cfg.name]
    
    # Get the body names for the arms from asset_cfg
    body_ids = asset.bodies(asset_cfg.body_names)
    if len(body_ids) == 0:
        return
        
    # Get nominal mass from the generic model
    nominal_mass = asset.mj_model.body_mass[body_ids]
    nominal_mass = torch.tensor(nominal_mass, device=env.device, dtype=torch.float32)
    
    # Broadcast to envs
    # Env specific model fields are shape [num_envs, field_dim]
    
    # Sample random mass multipliers
    min_mult, max_mult = mass_range
    multipliers = torch.rand((len(env_ids), 1), device=env.device) * (max_mult - min_mult) + min_mult
    
    # Calculate new mass
    new_mass = nominal_mass.unsqueeze(0) * multipliers
    
    # Assign new mass to the specific environments and bodies
    env.sim.model.body_mass[env_ids.unsqueeze(-1), body_ids] = new_mass
