"""Unitree G1 with Inspire Hands configuration."""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import (
  ElectricActuator,
  reflected_inertia_from_two_stage_planetary,
)
from mjlab.utils.os import update_assets

##
# MJCF and assets
##

# ✅ FIX 1: Usar Path(__file__).parent en lugar de MJLAB_SRC_PATH
# Así siempre apunta a la carpeta donde vive este propio archivo.
G1_HANDS_XML: Path = Path(__file__).parent / "g1_hands.xml"
assert G1_HANDS_XML.exists(), f"XML not found: {G1_HANDS_XML}"


def get_assets(meshdir: str) -> dict[str, bytes]:
  """Carga todos los meshes (STL) que usa el XML."""
  assets: dict[str, bytes] = {}
  update_assets(assets, Path(__file__).parent / "meshes", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  """Carga la especificación de MuJoCo desde el XML."""
  spec = mujoco.MjSpec.from_file(str(G1_HANDS_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator configs
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

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
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

##
# Inspire Hand actuators
##

ACTUATOR_HAND_MOTOR = ElectricActuator(
  reflected_inertia=0.001,
  velocity_limit=10.0,
  effort_limit=1.0,
)

STIFFNESS_HAND = 10.0
DAMPING_HAND = 2.0

G1_ACTUATOR_HANDS = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "L_thumb_proximal_yaw_joint",
    "L_thumb_proximal_pitch_joint",
    "L_index_proximal_joint",
    "L_middle_proximal_joint",
    "L_ring_proximal_joint",
    "L_pinky_proximal_joint",
    "R_thumb_proximal_yaw_joint",
    "R_thumb_proximal_pitch_joint",
    "R_index_proximal_joint",
    "R_middle_proximal_joint",
    "R_ring_proximal_joint",
    "R_pinky_proximal_joint",
  ),
  stiffness=STIFFNESS_HAND,
  damping=DAMPING_HAND,
  effort_limit=ACTUATOR_HAND_MOTOR.effort_limit,
  armature=ACTUATOR_HAND_MOTOR.reflected_inertia,
)


##
# Keyframes
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.793),
  joint_pos={
    ".*_hip_pitch_joint": -0.1,
    ".*_knee_joint": 0.3,
    ".*_ankle_pitch_joint": -0.2,
    ".*_shoulder_pitch_joint": 0.2,
    ".*_elbow_joint": 1.28,
    "left_shoulder_roll_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "L_thumb_proximal_yaw_joint": 0.0,
    "L_thumb_proximal_pitch_joint": 0.0,
    "L_index_proximal_joint": 0.0,
    "L_middle_proximal_joint": 0.0,
    "L_ring_proximal_joint": 0.0,
    "L_pinky_proximal_joint": 0.0,
    "R_thumb_proximal_yaw_joint": 0.0,
    "R_thumb_proximal_pitch_joint": 0.0,
    "R_index_proximal_joint": 0.0,
    "R_middle_proximal_joint": 0.0,
    "R_ring_proximal_joint": 0.0,
    "R_pinky_proximal_joint": 0.0,
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
    "L_thumb_proximal_yaw_joint": 0.0,
    "L_thumb_proximal_pitch_joint": 0.0,
    "L_index_proximal_joint": 0.0,
    "L_middle_proximal_joint": 0.0,
    "L_ring_proximal_joint": 0.0,
    "L_pinky_proximal_joint": 0.0,
    "R_thumb_proximal_yaw_joint": 0.0,
    "R_thumb_proximal_pitch_joint": 0.0,
    "R_index_proximal_joint": 0.0,
    "R_middle_proximal_joint": 0.0,
    "R_ring_proximal_joint": 0.0,
    "R_pinky_proximal_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)


##
# Articulation config
##

G1_HANDS_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    G1_ACTUATOR_5020,
    G1_ACTUATOR_7520_14,
    G1_ACTUATOR_7520_22,
    G1_ACTUATOR_4010,
    G1_ACTUATOR_WAIST,
    G1_ACTUATOR_ANKLE,
    G1_ACTUATOR_HANDS,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_g1_hands_robot_cfg() -> EntityCfg:
  """Factory function: devuelve configuración fresca del G1 con manos."""
  return EntityCfg(
    init_state=KNEES_BENT_KEYFRAME,
    collisions=(),  # ✅ FIX 2: El XML no tiene geoms con sufijo _collision.
    # Se añadirán colisiones propias en el siguiente paso.
    spec_fn=get_spec,
    articulation=G1_HANDS_ARTICULATION,
  )


##
# Action scale
##

G1_HANDS_ACTION_SCALE: dict[str, float] = {}

for a in G1_HANDS_ARTICULATION.actuators[:-1]:
  if isinstance(a, BuiltinPositionActuatorCfg):
    e = a.effort_limit
    s = a.stiffness
    if e is not None and s is not None:
      for n in a.target_names_expr:
        G1_HANDS_ACTION_SCALE[n] = 0.25 * e / s

hand_actuator = G1_HANDS_ARTICULATION.actuators[-1]
if isinstance(hand_actuator, BuiltinPositionActuatorCfg):
  e = hand_actuator.effort_limit
  s = hand_actuator.stiffness
  if e is not None and s is not None:
    for n in hand_actuator.target_names_expr:
      G1_HANDS_ACTION_SCALE[n] = 0.5 * e / s


##
# Quick visual test
##

if __name__ == "__main__":
  import mujoco.viewer as viewer
  from mjlab.entity.entity import Entity

  robot = Entity(get_g1_hands_robot_cfg())
  compiled = robot.spec.compile()
  print(f"✅ Robot loaded with {compiled.nq} DOFs")
  hand_joints = [
    compiled.joint(i).name
    for i in range(compiled.njnt)
    if any(
      k in compiled.joint(i).name for k in ("thumb", "index", "middle", "ring", "pinky")
    )
  ]
  print(f"Hand joints ({len(hand_joints)}): {hand_joints}")
  viewer.launch(compiled)
