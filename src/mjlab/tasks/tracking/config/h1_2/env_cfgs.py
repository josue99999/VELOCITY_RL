"""Unitree H1 v2 flat tracking environment configurations."""

from mjlab.asset_zoo.robots.robot_hands.h1_2_with_hands_constants import (
  H1_2_ACTION_SCALE,
  get_h1_2_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking import mdp
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import (
  VELOCITY_RANGE,
  make_tracking_env_cfg,
)

_POSE_RANGE_FULL = {
  "x": (-0.05, 0.05),
  "y": (-0.05, 0.05),
  "z": (-0.01, 0.01),
  "roll": (-0.1, 0.1),
  "pitch": (-0.1, 0.1),
  "yaw": (-0.2, 0.2),
}
_POSE_RANGE_HALF = {k: (v[0] * 0.5, v[1] * 0.5) for k, v in _POSE_RANGE_FULL.items()}
_VEL_RANGE_HALF = {k: (v[0] * 0.5, v[1] * 0.5) for k, v in VELOCITY_RANGE.items()}


def unitree_h1_2_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree H1 v2 flat terrain tracking configuration."""
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": get_h1_2_robot_cfg()}

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = H1_2_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = "torso_link"
  motion_cmd.body_names = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
  )

  # Start with zero RSI noise so the untrained policy can survive.
  motion_cmd.pose_range = {}
  motion_cmd.velocity_range = {}
  motion_cmd.joint_position_range = (0.0, 0.0)

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_foot[1-7]_collision$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  )

  cfg.viewer.body_name = "torso_link"

  cfg.rewards["action_rate_l2"].weight = -0.01
  cfg.rewards["self_collisions"].weight = -1.0
  cfg.rewards["joint_vel"] = RewardTermCfg(
    func=mdp.joint_vel_l2,
    weight=-0.001,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
  )

  cfg.curriculum = {
    "rsi_noise": CurriculumTermCfg(
      func=mdp.motion_rsi_curriculum,
      params={
        "command_name": "motion",
        "stages": [
          {"step": 0, "pose_range": {}, "velocity_range": {}},
          {
            "step": 48_000,
            "pose_range": _POSE_RANGE_HALF,
            "velocity_range": _VEL_RANGE_HALF,
            "joint_position_range": (-0.05, 0.05),
          },
          {
            "step": 120_000,
            "pose_range": _POSE_RANGE_FULL,
            "velocity_range": VELOCITY_RANGE,
            "joint_position_range": (-0.1, 0.1),
          },
        ],
      },
    ),
    "ee_threshold": CurriculumTermCfg(
      func=mdp.termination_threshold_curriculum,
      params={
        "termination_name": "ee_body_pos",
        "stages": [
          {"step": 0, "threshold": 0.5},
          {"step": 48_000, "threshold": 0.35},
          {"step": 120_000, "threshold": 0.25},
        ],
      },
    ),
    "anchor_pos_threshold": CurriculumTermCfg(
      func=mdp.termination_threshold_curriculum,
      params={
        "termination_name": "anchor_pos",
        "stages": [
          {"step": 0, "threshold": 0.5},
          {"step": 48_000, "threshold": 0.35},
          {"step": 120_000, "threshold": 0.25},
        ],
      },
    ),
    "anchor_ori_threshold": CurriculumTermCfg(
      func=mdp.termination_threshold_curriculum,
      params={
        "termination_name": "anchor_ori",
        "stages": [
          {"step": 0, "threshold": 1.5},
          {"step": 48_000, "threshold": 1.2},
          {"step": 120_000, "threshold": 0.8},
        ],
      },
    ),
    "action_rate_weight": CurriculumTermCfg(
      func=mdp.reward_weight_curriculum,
      params={
        "reward_name": "action_rate_l2",
        "stages": [
          {"step": 0, "weight": -0.01},
          {"step": 48_000, "weight": -0.05},
          {"step": 120_000, "weight": -0.1},
        ],
      },
    ),
    "self_collision_weight": CurriculumTermCfg(
      func=mdp.reward_weight_curriculum,
      params={
        "reward_name": "self_collisions",
        "stages": [
          {"step": 0, "weight": -1.0},
          {"step": 48_000, "weight": -5.0},
          {"step": 120_000, "weight": -10.0},
        ],
      },
    ),
  }

  if not has_state_estimation:
    new_actor_terms = {
      k: v
      for k, v in cfg.observations["actor"].terms.items()
      if k not in ["motion_anchor_pos_b", "base_lin_vel"]
    }
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=new_actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.sampling_mode = "start"
    cfg.curriculum = {}

  return cfg
