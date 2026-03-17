"""Unitree H1 v2 lower limb velocity environment configurations."""

import re

from mjlab.asset_zoo.robots.robot_hands.h1_2_with_hands_constants import (
  H1_2_ACTION_SCALE,
  get_h1_2_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.LOWER_LIMB_CONTROL_H1_2 import mdp
from mjlab.tasks.LOWER_LIMB_CONTROL_H1_2.mdp import (
  UniformVelocityCommandCfg,
)
from mjlab.tasks.LOWER_LIMB_CONTROL_H1_2.velocity_env_cfg import (
  CONTROLLED_JOINTS,
  make_velocity_env_cfg,
)
from mjlab.terrains.config import ROUGH_TERRAINS_H1_2_CFG

# Arm joints (not controlled, but moved as perturbation via teleop).
ARM_JOINTS = (
  ".*_shoulder_pitch_joint",
  ".*_shoulder_roll_joint",
  ".*_shoulder_yaw_joint",
  ".*_elbow_joint",
  ".*_wrist_roll_joint",
  ".*_wrist_pitch_joint",
  ".*_wrist_yaw_joint",
)

# Step multiplier: 1 iteration = num_steps_per_env * num_envs env steps.
_STEP_MUL = 24 * 1024  # 24 steps/iter * 1024 envs


def unitree_h1_2_rough_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree H1 v2 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 200

  # Coarser heightfield for H1_2's 14 foot geoms.
  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator = ROUGH_TERRAINS_H1_2_CFG

  cfg.scene.entities = {"robot": get_h1_2_robot_cfg()}

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "pelvis"

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  controlled_action_scale: dict[str, float] = {}
  for pattern, scale in H1_2_ACTION_SCALE.items():
    for ctrl_pattern in CONTROLLED_JOINTS:
      if re.match(ctrl_pattern, pattern.replace(".*", "left")) or re.match(
        ctrl_pattern, pattern.replace(".*", "right")
      ):
        controlled_action_scale[pattern] = scale
        break

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = controlled_action_scale

  cfg.viewer.body_name = "torso_link"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15

  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    r".*hip_pitch.*": 0.3,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.35,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
    r"torso_joint": 0.2,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.2,
    r".*hip_yaw.*": 0.2,
    r".*knee.*": 0.6,
    r".*ankle_pitch.*": 0.35,
    r".*ankle_roll.*": 0.15,
    r"torso_joint": 0.3,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)

  for reward_name in [
    "feet_lift",
    "foot_clearance",
    "foot_swing_height",
    "foot_slip",
  ]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  cfg.rewards["body_ang_vel"].weight = -0.02
  cfg.rewards["angular_momentum"].weight = -0.005
  cfg.rewards["air_time"].weight = 1.0

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.25,
    params={"sensor_name": self_collision_cfg.name},
  )

  # Arm continuous teleop: starts disabled (max_delta=0), curriculum ramps it.
  cfg.events["arm_pose_continuous_teleop"] = EventTermCfg(
    func=mdp.arm_pose_continuous_teleop,
    mode="interval",
    interval_range_s=(0.02, 0.02),
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS),
      "max_delta": 0.0,  # Disabled initially; curriculum overrides.
      "main_arm_scale": 1.0,
    },
  )

  # Curriculum: gradually increase arm teleop intensity.
  # 0-10k iters: no arm movement (policy learns to walk first)
  # 10k-20k: very slow (max_delta=0.003)
  # 20k-30k: slow (max_delta=0.006)
  # 30k-40k: medium (max_delta=0.010)
  # 40k-50k: faster (max_delta=0.015)
  # 50k-60k: full speed (max_delta=0.020)
  cfg.curriculum["arm_teleop"] = CurriculumTermCfg(
    func=mdp.arm_teleop_vel_stages,
    params={
      "event_name": "arm_pose_continuous_teleop",
      "stages": [
        {
          "step": 10_000 * _STEP_MUL,
          "max_delta": 0.003,
          "main_arm_scale": 1.5,
        },
        {
          "step": 20_000 * _STEP_MUL,
          "max_delta": 0.006,
          "main_arm_scale": 2.0,
        },
        {
          "step": 30_000 * _STEP_MUL,
          "max_delta": 0.010,
          "main_arm_scale": 2.5,
        },
        {
          "step": 40_000 * _STEP_MUL,
          "max_delta": 0.015,
          "main_arm_scale": 3.0,
        },
        {
          "step": 50_000 * _STEP_MUL,
          "max_delta": 0.020,
          "main_arm_scale": 3.5,
        },
      ],
    },
  )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
    # Enable arm teleop at full speed for evaluation.
    cfg.events["arm_pose_continuous_teleop"].params["max_delta"] = 0.020
    cfg.events["arm_pose_continuous_teleop"].params["main_arm_scale"] = 3.5
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )
    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_h1_2_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree H1 v2 flat terrain velocity configuration."""
  cfg = unitree_h1_2_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg
