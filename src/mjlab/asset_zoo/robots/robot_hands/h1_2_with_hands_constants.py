"""Unitree H1 v2 + Inspire Hands constants.

Estructura análoga a g1_with_hands_constants.py:
  1. H1_2_XML apunta al XML con manos integradas (sin <actuator> ni sensores de joints)
  2. get_spec() inyecta los <equality> mimic de los dedos
  3. Actuadores del cuerpo H1_2 por grupos (effort_limit del XML original)
  4. HAND_ACTUATOR_THUMB y HAND_ACTUATOR_FINGERS (Inspire)
  5. Keyframes con pose inicial y manos abiertas
"""

from pathlib import Path

import mujoco
import numpy as np

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

H1_2_XML: Path = (
  Path(__file__).resolve().parent.parent / "H1_2_with_hands" / "h1_2_hand.xml"
)
assert H1_2_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  # Meshes pueden estar en H1_2_with_hands/ o H1_2_with_hands/meshes/
  mesh_path = H1_2_XML.parent / "meshes"
  if mesh_path.exists():
    update_assets(assets, mesh_path, meshdir)
  else:
    update_assets(assets, H1_2_XML.parent, meshdir)
  return assets


# Mimic equalities para dedos (misma mano Inspire que G1).
_FINGER_MIMICS = [
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


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(H1_2_XML))
  spec.assets = get_assets(spec.meshdir)

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
# Actuator config — cuerpo H1 v2 (effort_limit del XML original).
##

# Constantes de control (posición) para agrupar por esfuerzo máximo.
NATURAL_FREQ = 10 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0
ARMATURE_DEFAULT = 0.1  # del <default> del XML


def _stiffness(effort: float) -> float:
  return ARMATURE_DEFAULT * NATURAL_FREQ**2


def _damping(effort: float) -> float:
  return 2.0 * DAMPING_RATIO * ARMATURE_DEFAULT * NATURAL_FREQ


# Grupos por ctrlrange (Nm) del XML original.
H1_2_ACTUATOR_HIP_LEG = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_hip_yaw_joint",
    ".*_hip_pitch_joint",
    ".*_hip_roll_joint",
  ),
  stiffness=_stiffness(200),
  damping=_damping(200),
  effort_limit=200.0,
  armature=ARMATURE_DEFAULT,
)
H1_2_ACTUATOR_KNEE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_knee_joint",),
  stiffness=_stiffness(300),
  damping=_damping(300),
  effort_limit=300.0,
  armature=ARMATURE_DEFAULT,
)
# ankle_pitch ±60 Nm, ankle_roll ±40 Nm en el XML original.
H1_2_ACTUATOR_ANKLE_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_roll_joint",),
  stiffness=_stiffness(40),
  damping=_damping(40),
  effort_limit=40.0,
  armature=ARMATURE_DEFAULT,
)
H1_2_ACTUATOR_TORSO = BuiltinPositionActuatorCfg(
  target_names_expr=("torso_joint",),
  stiffness=_stiffness(200),
  damping=_damping(200),
  effort_limit=200.0,
  armature=ARMATURE_DEFAULT,
)
H1_2_ACTUATOR_SHOULDER_ELBOW = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_elbow_joint",
  ),
  stiffness=_stiffness(40),
  damping=_damping(40),
  effort_limit=40.0,
  armature=ARMATURE_DEFAULT,
)
# shoulder_yaw y elbow tenían ±18; agrupamos con shoulder para simplificar (40 domina).
H1_2_ACTUATOR_WRIST = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_wrist_roll_joint",
    ".*_wrist_pitch_joint",
    ".*_wrist_yaw_joint",
  ),
  stiffness=_stiffness(19),
  damping=_damping(19),
  effort_limit=19.0,
  armature=ARMATURE_DEFAULT,
)

# Ajuste: ankle en un solo grupo con effort 60 para pitch y 40 para roll (promedio o el mayor).
# Mejor dos grupos: ya tenemos H1_2_ACTUATOR_ANKLE (pitch 60) y H1_2_ACTUATOR_ANKLE_ROLL (40).
# Quitamos el effort_limit 60 del grupo ankle y dejamos solo pitch en H1_2_ACTUATOR_ANKLE.
# H1_2_ACTUATOR_ANKLE ya tiene ambos joints; el effort_limit 60 es para pitch. Para no
# sobrecargar ankle_roll, usamos dos configs: uno para pitch (60), otro para roll (40).
H1_2_ACTUATOR_ANKLE_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch_joint",),
  stiffness=_stiffness(60),
  damping=_damping(60),
  effort_limit=60.0,
  armature=ARMATURE_DEFAULT,
)

# Manos Inspire (igual que G1).
HAND_ACTUATOR_THUMB = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "[LR]_thumb_proximal_yaw_joint",
    "[LR]_thumb_proximal_pitch_joint",
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

# Altura inicial H1_2 (pelvis en 1.03 en el XML).
HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 1.03),
  joint_pos={
    ".*_hip_pitch_joint": -0.1,
    ".*_knee_joint": 0.3,
    ".*_ankle_pitch_joint": -0.2,
    ".*_ankle_roll_joint": 0.0,
    "torso_joint": 0.0,
    ".*_shoulder_pitch_joint": 0.2,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
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
  pos=(0, 0, 1.0),
  joint_pos={
    ".*_hip_pitch_joint": -0.3,
    ".*_knee_joint": 0.6,
    ".*_ankle_pitch_joint": -0.3,
    ".*_ankle_roll_joint": 0.0,
    "torso_joint": 0.0,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    ".*_shoulder_pitch_joint": 0.2,
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
# Collision config (geoms mano *_col; H1_2 no tiene foot_collision como G1).
##

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_col", ".*_collision"),
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1, ".*": 0},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,), ".*": (0.5,)},
)

##
# Final config.
##

H1_2_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    H1_2_ACTUATOR_HIP_LEG,
    H1_2_ACTUATOR_KNEE,
    H1_2_ACTUATOR_ANKLE_PITCH,
    H1_2_ACTUATOR_ANKLE_ROLL,
    H1_2_ACTUATOR_TORSO,
    H1_2_ACTUATOR_SHOULDER_ELBOW,
    H1_2_ACTUATOR_WRIST,
    HAND_ACTUATOR_THUMB,
    HAND_ACTUATOR_FINGERS,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_h1_2_robot_cfg() -> EntityCfg:
  """H1 v2 con manos Inspire — para standing RL y teleoperación."""
  return EntityCfg(
    init_state=KNEES_BENT_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=H1_2_ARTICULATION,
  )


H1_2_ACTION_SCALE: dict[str, float] = {}
for a in H1_2_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  assert s is not None
  for n in names:
    H1_2_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_h1_2_robot_cfg())
  viewer.launch(robot.spec.compile())
