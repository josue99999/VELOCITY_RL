"""Velocity task configuration.

This module provides a factory function to create a base velocity task config.
Robot-specific configurations call the factory and customize as needed.
"""

import math
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.LOWER_LIMB_CONTROL_H1_2 import mdp
from mjlab.tasks.LOWER_LIMB_CONTROL_H1_2.mdp import UniformVelocityCommandCfg
from mjlab.tasks.LOWER_LIMB_CONTROL_H1_2.mdp.actions import NoisyJointPositionActionCfg
from mjlab.terrains import TerrainImporterCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

# Controlled joints (Lower limb + torso for stability).
# H1 v2: 6 leg patterns × 2 sides = 12 DOF.
# 1 torso joint (yaw) = 1 DOF.
# Total = 13 DOF.
CONTROLLED_JOINTS = (
  ".*_hip_pitch_joint",
  ".*_hip_roll_joint",
  ".*_hip_yaw_joint",
  ".*_knee_joint",
  ".*_ankle_pitch_joint",
  ".*_ankle_roll_joint",
  "torso_joint",
)

# Observed joints: controlled (legs + torso) + arms/hands for perturbation awareness.
OBSERVED_JOINTS = CONTROLLED_JOINTS + (
  ".*_shoulder_.*",
  ".*_elbow_joint",
  ".*_wrist_.*",
  ".*_thumb_.*",
  ".*_index_.*",
  ".*_middle_.*",
  ".*_ring_.*",
  ".*_pinky_.*",
)

# Joints and bodies for curriculum perturbations (arms, hands, fingers)
ARM_AND_HAND_JOINTS = (
  ".*_shoulder_.*",
  ".*_elbow_joint",
  ".*_wrist_.*",
  ".*_thumb_.*",
  ".*_index_.*",
  ".*_middle_.*",
  ".*_ring_.*",
  ".*_pinky_.*",
)

ARM_AND_HAND_BODIES = (
  ".*_shoulder_.*",
  ".*_elbow_.*",
  ".*_wrist_.*",
  ".*_hand_.*",
  ".*_thumb_.*",
  ".*_index_.*",
  ".*_middle_.*",
  ".*_ring_.*",
  ".*_pinky_.*",
)

# Metric-gated 6-phase curriculum.
#
# Phase advancement requires ALL conditions to hold simultaneously:
#   - min_episodes_in_phase: completed episodes in this phase (safety floor).
#   - success_rate_min: rolling fraction of episodes without falls (last 200 eps).
#   - ep_length_ratio_min: rolling mean(ep_len)/max_ep_len (last 200 eps).
#   - reward_threshold: (optional) mean reward from runner metrics.
#
# episode_range is kept as metadata for backward-compat logging; the actual
# gate is the per-phase metric conditions above, evaluated by gatekeeping_phase_control.
CURRICULUM_PHASES = {
  "pure_walking": {
    "episode_range": (0, 100_000),
    # Phase-advancement conditions.
    "min_episodes_in_phase": 800,  # Safety floor; real gate is success_rate.
    "success_rate_min": 0.85,  # < 15 % fall rate required.
    "ep_length_ratio_min": 0.80,  # Episodes mostly run to time-out.
    "reward_threshold": None,
    # Arm / disturbance parameters.
    "arm_randomization": False,
    "arm_pose_range": 0.0,
    "arm_mass_range": (1.0, 1.0),
    "push_velocity": 0.0,
    "push_interval": (10.0, 20.0),
    "arm_teleop_max_delta": 0.0,
    "arm_teleop_main_arm_scale": 1.0,
    "arm_teleop_interval_range_s": (0.1, 0.1),
  },
  "arm_randomization": {
    "episode_range": (100_000, 200_000),
    "min_episodes_in_phase": 1200,
    "success_rate_min": 0.82,
    "ep_length_ratio_min": 0.78,
    "reward_threshold": None,
    "arm_randomization": True,
    "arm_pose_range": 0.25,
    "arm_mass_range": (0.9, 1.1),
    "push_velocity": 0.0,
    "push_interval": (10.0, 20.0),
    "arm_teleop_max_delta": 0.0,
    "arm_teleop_main_arm_scale": 1.0,
    "arm_teleop_interval_range_s": (0.1, 0.1),
  },
  "arm_pose_exploration": {
    "episode_range": (200_000, 300_000),
    "min_episodes_in_phase": 1200,
    "success_rate_min": 0.80,
    "ep_length_ratio_min": 0.75,
    "reward_threshold": None,
    "arm_randomization": True,
    "arm_pose_range": 0.5,
    "arm_mass_range": (0.8, 1.2),
    "push_velocity": 0.0,
    "push_interval": (10.0, 20.0),
    "arm_teleop_max_delta": 0.02,
    "arm_teleop_main_arm_scale": 1.5,
    "arm_teleop_interval_range_s": (0.05, 0.05),
  },
  "light_disturbances": {
    "episode_range": (300_000, 400_000),
    "min_episodes_in_phase": 1500,
    "success_rate_min": 0.77,
    "ep_length_ratio_min": 0.70,
    "reward_threshold": None,
    "arm_randomization": True,
    "arm_pose_range": 0.5,
    "arm_mass_range": (0.8, 1.2),
    "push_velocity": 0.08,
    "push_interval": (8.0, 15.0),
    "arm_teleop_max_delta": 0.03,
    "arm_teleop_main_arm_scale": 2.0,
    "arm_teleop_interval_range_s": (0.03, 0.03),
  },
  "moderate_disturbances": {
    "episode_range": (400_000, 500_000),
    "min_episodes_in_phase": 1500,
    "success_rate_min": 0.74,
    "ep_length_ratio_min": 0.65,
    "reward_threshold": None,
    "arm_randomization": True,
    "arm_pose_range": 0.75,
    "arm_mass_range": (0.7, 1.3),
    "push_velocity": 0.15,
    "push_interval": (5.0, 8.0),
    "arm_teleop_max_delta": 0.05,
    "arm_teleop_main_arm_scale": 2.5,
    "arm_teleop_interval_range_s": (0.02, 0.02),
  },
  "full_robustness": {
    "episode_range": (500_000, float("inf")),
    "min_episodes_in_phase": 0,  # Last phase — no advancement.
    "success_rate_min": 0.0,
    "ep_length_ratio_min": 0.0,
    "reward_threshold": None,
    "arm_randomization": True,
    "arm_pose_range": 1.0,
    "arm_mass_range": (0.6, 1.5),
    "push_velocity": 0.25,
    "push_interval": (3.0, 5.0),
    "arm_teleop_max_delta": 0.08,
    "arm_teleop_main_arm_scale": 3.0,
    "arm_teleop_interval_range_s": (0.01, 0.01),
  },
}


def make_velocity_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base velocity tracking task configuration for lower limb control."""

  ##
  # Observations
  ##

  actor_terms = {
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
      history_length=6,
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
      history_length=6,
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
      history_length=6,
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=OBSERVED_JOINTS)},
      noise=Unoise(n_min=-0.01, n_max=0.01),
      history_length=6,
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=OBSERVED_JOINTS)},
      noise=Unoise(n_min=-1.5, n_max=1.5),
      history_length=6,
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "twist"},
    ),
    "gait_phase_clock": ObservationTermCfg(
      func=mdp.gait_phase_clock,
      params={"cycle_freq_hz": 1.5},
    ),
    "height_scan": ObservationTermCfg(
      func=envs_mdp.height_scan,
      params={"sensor_name": "terrain_scan"},
      noise=Unoise(n_min=-0.1, n_max=0.1),
      clip=(-1.0, 1.0),
    ),
  }

  critic_terms = {
    **actor_terms,
    "height_scan": ObservationTermCfg(
      func=envs_mdp.height_scan,
      params={"sensor_name": "terrain_scan"},
      clip=(-1.0, 1.0),
    ),
    "foot_height": ObservationTermCfg(
      func=mdp.foot_height,
      params={"asset_cfg": SceneEntityCfg("robot", site_names=())},  # Set per-robot.
    ),
    "foot_air_time": ObservationTermCfg(
      func=mdp.foot_air_time,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact": ObservationTermCfg(
      func=mdp.foot_contact,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact_forces": ObservationTermCfg(
      func=mdp.foot_contact_forces,
      params={"sensor_name": "feet_ground_contact"},
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
      nan_policy="sanitize",
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
      nan_policy="sanitize",
    ),
  }

  ##
  # Actions
  ##

  joint_actuator_names = [pattern for pattern in CONTROLLED_JOINTS]
  actions: dict[str, ActionTermCfg] = {
    "joint_pos": NoisyJointPositionActionCfg(
      entity_name="robot",
      actuator_names=joint_actuator_names,
      scale=0.25,  # Override per-robot.
      use_default_offset=True,
    )
  }

  ##
  # Commands
  ##

  commands: dict[str, CommandTermCfg] = {
    "twist": UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(3.0, 8.0),
      rel_standing_envs=0.1,
      rel_heading_envs=0.3,
      heading_command=True,
      heading_control_stiffness=0.5,
      debug_vis=True,
      init_velocity_prob=0.2,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-1.0, 1.0),
        lin_vel_y=(-1.0, 1.0),
        ang_vel_z=(-0.5, 0.5),
        heading=(-math.pi, math.pi),
      ),
    )
  }

  ##
  # Events
  ##

  events = {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (0.01, 0.05),
          "yaw": (-3.14, 3.14),
        },
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINTS),
      },
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 3.0),
      params={
        "velocity_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (-0.4, 0.4),
          "roll": (-0.52, 0.52),
          "pitch": (-0.52, 0.52),
          "yaw": (-0.78, 0.78),
        },
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
        "operation": "abs",
        "field": "geom_friction",
        "ranges": (0.3, 1.2),
        "shared_random": True,  # All foot geoms share the same friction.
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=mdp.randomize_encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.015, 0.015),
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
        "operation": "add",
        "field": "body_ipos",
        "ranges": {
          0: (-0.025, 0.025),
          1: (-0.025, 0.025),
          2: (-0.03, 0.03),
        },
      },
    ),
    "randomize_link_masses": EventTermCfg(
      mode="reset",
      func=mdp.randomize_link_masses,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "mass_range": (0.8, 1.2),
      },
    ),
    "randomize_link_inertia": EventTermCfg(
      mode="reset",
      func=mdp.randomize_link_inertia,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "inertia_range": (0.8, 1.2),
      },
    ),
    "randomize_pd_gains": EventTermCfg(
      mode="reset",
      func=mdp.randomize_pd_gains,
      params={
        "kp_range": (0.8, 1.2),
        "kd_range": (0.8, 1.2),
        # Apply PD gain randomization to all actuators of the robot.
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    "randomize_arm_pose": EventTermCfg(
      func=mdp.events.randomize_arm_pose_phase_based,
      mode="reset",
      params={
        "phases": CURRICULUM_PHASES,
        "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_AND_HAND_JOINTS),
      },
    ),
    "randomize_arm_mass": EventTermCfg(
      func=mdp.events.randomize_arm_mass_phase_based,
      mode="reset",
      params={
        "phases": CURRICULUM_PHASES,
        "asset_cfg": SceneEntityCfg("robot", body_names=ARM_AND_HAND_BODIES),
      },
    ),
    "arm_pose_continuous_teleop": EventTermCfg(
      func=mdp.events.arm_pose_continuous_teleop,
      mode="interval",
      interval_range_s=(0.02, 0.02),
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_AND_HAND_JOINTS),
        "max_delta": 0.04,
        "main_arm_scale": 2.0,
      },
    ),
  }

  ##
  # Rewards
  ##

  rewards = {
    "track_linear_velocity": RewardTermCfg(
      func=mdp.track_linear_velocity,
      weight=1.0,
      params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    "track_angular_velocity": RewardTermCfg(
      func=mdp.track_angular_velocity,
      weight=0.5,
      params={"command_name": "twist", "std": math.sqrt(0.5)},
    ),
    "upright": RewardTermCfg(
      func=mdp.flat_orientation,
      weight=2.0,
      params={
        "std": math.sqrt(0.3),
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
      },
    ),
    "pose": RewardTermCfg(
      func=mdp.variable_posture,
      weight=0.8,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINTS),
        "command_name": "twist",
        "std_standing": {},  # Set per-robot (env_cfgs fills these).
        "std_walking": {},  # Set per-robot.
        "std_running": {},  # Set per-robot.
        "walking_threshold": 0.1,
        "running_threshold": 1.5,
      },
    ),
    "body_ang_vel": RewardTermCfg(
      func=mdp.body_angular_velocity_penalty,
      weight=-0.05,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # Set per-robot.
    ),
    "angular_momentum": RewardTermCfg(
      func=mdp.angular_momentum_penalty,
      weight=-0.005,
      params={"sensor_name": "robot/root_angmom"},
    ),
    "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-0.5),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),
    "joint_torques": RewardTermCfg(
      func=mdp.joint_torques_l2,
      weight=-0.0001,
      params={
        "asset_cfg": SceneEntityCfg("robot", actuator_names=list(CONTROLLED_JOINTS))
      },
    ),
    "joint_acceleration": RewardTermCfg(
      func=mdp.joint_acc_l2,
      weight=-0.00001,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINTS)},
    ),
    "zmp_stability": RewardTermCfg(
      func=mdp.zmp_stability_reward,
      weight=0.3,
      params={"asset_cfg": SceneEntityCfg("robot"), "support_polygon_margin": 0.05},
    ),
    "base_height": RewardTermCfg(
      func=mdp.base_height_penalty,
      weight=-100.0,
      params={"target_height": 0.98, "asset_cfg": SceneEntityCfg("robot")},
    ),
    "survival_bonus": RewardTermCfg(
      func=mdp.survival_bonus,
      weight=0.5,
      params={"bonus_per_step": 1.0},
    ),
    "air_time": RewardTermCfg(
      func=mdp.feet_air_time,
      weight=0.5,
      params={
        "sensor_name": "feet_ground_contact",
        "threshold_min": 0.05,
        "threshold_max": 0.3,
        "command_name": "twist",
        "command_threshold": 0.5,
      },
    ),
    "foot_clearance": RewardTermCfg(
      func=mdp.feet_clearance_improved,
      weight=-0.5,
      params={
        "target_height": 0.08,
        "min_height": 0.03,
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
        "sensor_name": "feet_ground_contact",
      },
    ),
    "foot_swing_height": RewardTermCfg(
      func=mdp.feet_swing_height,
      weight=-0.5,
      params={
        "sensor_name": "feet_ground_contact",
        "target_height": 0.06,
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    "foot_slip": RewardTermCfg(
      func=mdp.feet_slip,
      weight=-0.5,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    "soft_landing": RewardTermCfg(
      func=mdp.soft_landing,
      weight=-1e-5,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.1,
      },
    ),
    "contact_force_penalty": RewardTermCfg(
      func=mdp.contact_force_penalty,
      weight=-1e-4,
      params={"sensor_name": "feet_ground_contact", "force_threshold": 800.0},
    ),
    "stance_contact": RewardTermCfg(
      func=mdp.stance_contact_reward,
      weight=0.2,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.1,
      },
    ),
    "single_foot_contact": RewardTermCfg(
      func=mdp.single_foot_contact,
      weight=0.5,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.1,
        "grace_period": 0.2,
      },
    ),
    "stable_upright_under_disturbance": RewardTermCfg(
      func=mdp.stable_upright_under_disturbance,
      weight=0.5,
      params={"phases": CURRICULUM_PHASES},
    ),
  }

  ##
  # Terminations
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(55.0)},
    ),
  }

  ##
  # Curriculum
  ##

  curriculum = {
    # gatekeeping MUST be first: it sets env._gk_phase_key which is read by
    # get_current_phase() in all subsequent curriculum / event terms.
    "gatekeeping": CurriculumTermCfg(
      func=mdp.gatekeeping_phase_control,
      params={"phases": CURRICULUM_PHASES},
    ),
    "terrain_levels": CurriculumTermCfg(
      func=mdp.terrain_levels_vel,
      params={"command_name": "twist"},
    ),
    # Velocity stages más conservadores para evitar inestabilidad al subir fases.
    "command_vel": CurriculumTermCfg(
      func=mdp.commands_vel,
      params={
        "command_name": "twist",
        "velocity_stages": [
          {
            "step": 0,
            "lin_vel_x": (-0.5, 0.5),
            "lin_vel_y": (-0.3, 0.3),
            "ang_vel_z": (-0.3, 0.3),
          },
          {
            "step": 240_000,
            "lin_vel_x": (-0.8, 0.8),
            "lin_vel_y": (-0.5, 0.5),
            "ang_vel_z": (-0.4, 0.4),
          },
          {
            "step": 480_000,
            "lin_vel_x": (-1.0, 1.0),
            "lin_vel_y": (-0.6, 0.6),
            "ang_vel_z": (-0.5, 0.5),
          },
          {
            "step": 720_000,
            "lin_vel_x": (-1.2, 1.2),
            "lin_vel_y": (-0.7, 0.7),
            "ang_vel_z": (-0.55, 0.55),
          },
          {
            "step": 960_000,
            "lin_vel_x": (-1.5, 1.5),
            "lin_vel_y": (-0.8, 0.8),
            "ang_vel_z": (-0.6, 0.6),
          },
          {
            "step": 1_200_000,
            "lin_vel_x": (-1.8, 2.0),
            "lin_vel_y": (-0.9, 0.9),
            "ang_vel_z": (-0.65, 0.65),
          },
          {
            "step": 1_440_000,
            "lin_vel_x": (-2.0, 2.5),
            "lin_vel_y": (-1.0, 1.0),
            "ang_vel_z": (-0.7, 0.7),
          },
          {
            "step": 1_920_000,
            "lin_vel_x": (-2.0, 3.0),
            "lin_vel_y": (-1.0, 1.0),
            "ang_vel_z": (-0.7, 0.7),
          },
        ],
      },
    ),
    "teleop_disturbances": CurriculumTermCfg(
      func=mdp.update_teleop_pushes,
      params={
        "phases": CURRICULUM_PHASES,
        "push_event_name": "push_robot",
      },
    ),
    "arm_teleop_continuous": CurriculumTermCfg(
      func=mdp.update_arm_teleop_continuous,
      params={
        "phases": CURRICULUM_PHASES,
        "event_name": "arm_pose_continuous_teleop",
      },
    ),
    "phase_info": CurriculumTermCfg(
      func=mdp.log_phase_curriculum,
      params={"phases": CURRICULUM_PHASES},
    ),
  }

  ##
  # Assemble and return
  ##

  terrain_scan = RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="", entity="robot"),  # Set per-robot.
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
    max_distance=5.0,
    exclude_parent_body=True,
    debug_vis=True,
    viz=RayCastSensorCfg.VizCfg(show_normals=True),
  )

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainImporterCfg(
        terrain_type="generator",
        terrain_generator=replace(ROUGH_TERRAINS_CFG),
        max_init_terrain_level=5,
      ),
      sensors=(terrain_scan,),
      num_envs=1024,  # Lower default for 6GB GPUs; override with --env.scene.num_envs 4096
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=1500,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=2,
    episode_length_s=10.0,
    reward_clip=10.0,
  )
