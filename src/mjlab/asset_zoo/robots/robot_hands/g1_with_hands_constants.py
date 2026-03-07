"""Unitree G1 + Inspire Hands constants.

Respecto a g1_constants.py, hay 4 cambios:
  1. G1_XML apunta al nuevo XML con manos integradas
  2. get_spec() inyecta los <equality> mimic de los dedos (no están en el XML)
  3. Se añaden HAND_ACTUATOR_THUMB y HAND_ACTUATOR_FINGERS
  4. Los keyframes incluyen pose inicial de los dedos (manos abiertas)
"""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import (
  ElectricActuator,
  reflected_inertia_from_two_stage_planetary,
)
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

# ── CAMBIO 1: apunta al XML con manos ────────────────────────────────────────
G1_XML: Path = Path(__file__).resolve().parent / "xmls" / "g1_with_hands.xml"

assert G1_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, G1_XML.parent / "assets", meshdir)
  return assets


# ── CAMBIO 2: inyectar equalities mimic de los dedos ─────────────────────────
# El XML no tiene <equality>. Se añaden aquí para que los joints
# intermediate/distal sigan a los proximales automáticamente.
_FINGER_MIMICS = [
  # (nombre,               esclavo,                        maestro,                           ratio)
  (
    "L_thumb_int_mimic",
    "L_thumb_intermediate_joint",
    "L_thumb_proximal_pitch_joint",
    1.6,
  ),
  ("L_thumb_dis_mimic", "L_thumb_distal_joint", "L_thumb_proximal_pitch_joint", 2.4),
  ("L_index_int_mimic", "L_index_intermediate_joint", "L_index_proximal_joint", 1.0),
  ("L_middle_int_mimic", "L_middle_intermediate_joint", "L_middle_proximal_joint", 1.0),
  ("L_ring_int_mimic", "L_ring_intermediate_joint", "L_ring_proximal_joint", 1.0),
  ("L_pinky_int_mimic", "L_pinky_intermediate_joint", "L_pinky_proximal_joint", 1.0),
  (
    "R_thumb_int_mimic",
    "R_thumb_intermediate_joint",
    "R_thumb_proximal_pitch_joint",
    1.6,
  ),
  ("R_thumb_dis_mimic", "R_thumb_distal_joint", "R_thumb_proximal_pitch_joint", 2.4),
  ("R_index_int_mimic", "R_index_intermediate_joint", "R_index_proximal_joint", 1.0),
  ("R_middle_int_mimic", "R_middle_intermediate_joint", "R_middle_proximal_joint", 1.0),
  ("R_ring_int_mimic", "R_ring_intermediate_joint", "R_ring_proximal_joint", 1.0),
  ("R_pinky_int_mimic", "R_pinky_intermediate_joint", "R_pinky_proximal_joint", 1.0),
]

import numpy as np


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(G1_XML))
  spec.assets = get_assets(spec.meshdir)

  # Inyectar equalities mimic si no existen ya
  existing = {eq.name for eq in spec.equalities}
  for name, slave, master, ratio in _FINGER_MIMICS:
    if name not in existing:
      eq = spec.add_equality()
      eq.name = name
      eq.type = mujoco.mjtEq.mjEQ_JOINT
      eq.name1 = slave
      eq.name2 = master
      eq.data = np.array([0.0, ratio, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

  return spec


##
# Actuator config — cuerpo G1 (idéntico a g1_constants.py original).
##

ROTOR_INERTIAS_5020 = (0.139e-4, 0.017e-4, 0.169e-4)
GEARS_5020 = (1, 1 + (46 / 18), 1 + (56 / 16))
ARMATURE_5020 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_5020, GEARS_5020
)

ROTOR_INERTIAS_7520_14 = (0.489e-4, 0.098e-4, 0.533e-4)
GEARS_7520_14 = (1, 4.5, 1 + (48 / 22))
ARMATURE_7520_14 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_7520_14, GEARS_7520_14
)

ROTOR_INERTIAS_7520_22 = (0.489e-4, 0.109e-4, 0.738e-4)
GEARS_7520_22 = (1, 4.5, 5)
ARMATURE_7520_22 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_7520_22, GEARS_7520_22
)

ROTOR_INERTIAS_4010 = (0.068e-4, 0.0, 0.0)
GEARS_4010 = (1, 5, 5)
ARMATURE_4010 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_4010, GEARS_4010
)

ACTUATOR_5020 = ElectricActuator(
  reflected_inertia=ARMATURE_5020, velocity_limit=37.0, effort_limit=25.0
)
ACTUATOR_7520_14 = ElectricActuator(
  reflected_inertia=ARMATURE_7520_14, velocity_limit=32.0, effort_limit=88.0
)
ACTUATOR_7520_22 = ElectricActuator(
  reflected_inertia=ARMATURE_7520_22, velocity_limit=20.0, effort_limit=139.0
)
ACTUATOR_4010 = ElectricActuator(
  reflected_inertia=ARMATURE_4010, velocity_limit=22.0, effort_limit=5.0
)

NATURAL_FREQ = 10 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

G1_ACTUATOR_5020 = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_elbow_joint",
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_wrist_roll_joint",
  ),
  stiffness=STIFFNESS_5020,
  damping=DAMPING_5020,
  effort_limit=ACTUATOR_5020.effort_limit,
  armature=ACTUATOR_5020.reflected_inertia,
)
G1_ACTUATOR_7520_14 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch_joint", ".*_hip_yaw_joint", "waist_yaw_joint"),
  stiffness=STIFFNESS_7520_14,
  damping=DAMPING_7520_14,
  effort_limit=ACTUATOR_7520_14.effort_limit,
  armature=ACTUATOR_7520_14.reflected_inertia,
)
G1_ACTUATOR_7520_22 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_roll_joint", ".*_knee_joint"),
  stiffness=STIFFNESS_7520_22,
  damping=DAMPING_7520_22,
  effort_limit=ACTUATOR_7520_22.effort_limit,
  armature=ACTUATOR_7520_22.reflected_inertia,
)
G1_ACTUATOR_4010 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_yaw_joint"),
  stiffness=STIFFNESS_4010,
  damping=DAMPING_4010,
  effort_limit=ACTUATOR_4010.effort_limit,
  armature=ACTUATOR_4010.reflected_inertia,
)
G1_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_pitch_joint", "waist_roll_joint"),
  stiffness=STIFFNESS_5020 * 2,
  damping=DAMPING_5020 * 2,
  effort_limit=ACTUATOR_5020.effort_limit * 2,
  armature=ACTUATOR_5020.reflected_inertia * 2,
)
G1_ACTUATOR_ANKLE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
  stiffness=STIFFNESS_5020 * 2,
  damping=DAMPING_5020 * 2,
  effort_limit=ACTUATOR_5020.effort_limit * 2,
  armature=ACTUATOR_5020.reflected_inertia * 2,
)

# ── CAMBIO 3: actuadores de las manos Inspire ─────────────────────────────────
# Parámetros del XML original de la Inspire Hand:
#   kp=2 Nm/rad, damping=0.05, effort_limit=1 Nm
# Solo se actúan los joints PROXIMALES (6 por mano).
# Los intermediate/distal son mimics (equality) → no se actúan.
HAND_ACTUATOR_THUMB = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "[LR]_thumb_proximal_yaw_joint",  # abducción pulgar
    "[LR]_thumb_proximal_pitch_joint",  # flexión proximal pulgar
  ),
  stiffness=2.0,
  damping=0.05,
  effort_limit=1.0,
  armature=0.0,
)
HAND_ACTUATOR_FINGERS = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "[LR]_index_proximal_joint",
    "[LR]_middle_proximal_joint",
    "[LR]_ring_proximal_joint",
    "[LR]_pinky_proximal_joint",
  ),
  stiffness=2.0,
  damping=0.05,
  effort_limit=1.0,
  armature=0.0,
)

##
# Keyframe config.
##

# ── CAMBIO 4: añadir pose inicial de los dedos (manos abiertas) ───────────────
HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.783675),
  joint_pos={
    ".*_hip_pitch_joint": -0.1,
    ".*_knee_joint": 0.3,
    ".*_ankle_pitch_joint": -0.2,
    ".*_shoulder_pitch_joint": 0.2,
    ".*_elbow_joint": 1.28,
    "left_shoulder_roll_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    # manos abiertas
    "[LR]_thumb_proximal_yaw_joint": 0.0,
    "[LR]_thumb_proximal_pitch_joint": 0.0,
    "[LR]_index_proximal_joint": 0.0,
    "[LR]_middle_proximal_joint": 0.0,
    "[LR]_ring_proximal_joint": 0.0,
    "[LR]_pinky_proximal_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.76),
  joint_pos={
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
    # manos abiertas
    "[LR]_thumb_proximal_yaw_joint": 0.0,
    "[LR]_thumb_proximal_pitch_joint": 0.0,
    "[LR]_index_proximal_joint": 0.0,
    "[LR]_middle_proximal_joint": 0.0,
    "[LR]_ring_proximal_joint": 0.0,
    "[LR]_pinky_proximal_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config (sin cambios respecto a g1_constants.py).
# Las colisiones de la mano usan L/R_hand_base_col que ya son capsules en el XML.
##

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision", ".*_col"),
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision", ".*_col"),
  contype=0,
  conaffinity=1,
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot[1-7]_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

G1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    G1_ACTUATOR_5020,
    G1_ACTUATOR_7520_14,
    G1_ACTUATOR_7520_22,
    G1_ACTUATOR_4010,
    G1_ACTUATOR_WAIST,
    G1_ACTUATOR_ANKLE,
    HAND_ACTUATOR_THUMB,  # ← nuevo
    HAND_ACTUATOR_FINGERS,  # ← nuevo
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_g1_robot_cfg() -> EntityCfg:
  """G1 con manos Inspire — para standing RL y teleoperación."""
  return EntityCfg(
    init_state=KNEES_BENT_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=G1_ARTICULATION,
  )


G1_ACTION_SCALE: dict[str, float] = {}
for a in G1_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    G1_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_g1_robot_cfg())
  viewer.launch(robot.spec.compile())
