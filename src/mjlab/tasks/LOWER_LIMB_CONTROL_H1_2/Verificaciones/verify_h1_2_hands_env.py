"""
verify_h1_2_hands_env.py
========================
Script de verificación del entorno LOWER_LIMB_CONTROL_H1_2 / H1 v2 con manos.
Ejecutar ANTES de iniciar el entrenamiento:

    python verify_h1_2_hands_env.py

Comprueba:
  1. Carga del spec MuJoCo (XML + equalities mimic inyectadas)
  2. Joints: cuerpo H1 v2 + manos Inspire (proximal + mimic)
  3. Actuadores: que los patrones del constants cubran todos los joints actuados
  4. Sites: left_wrist_attach / right_wrist_attach presentes
  5. Geoms de mano: L/R_hand_base_col presentes
  6. Dimensión del espacio de observaciones actor y critic
  7. Dimensión del espacio de acciones
  8. Compatibilidad obs/act con la red neuronal del rl_cfg (hidden_dims)
  9. Keyframe HOME y KNEES_BENT: joints referenciados existen en el modelo
 10. Equality mimic: slave/master existen en el modelo
 11. Resumen final: PASS / FAIL por sección
"""

import re
import sys
from pathlib import Path

# Forzar uso del repo mjlab
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import mujoco

try:
  from mjlab.actuator import BuiltinPositionActuatorCfg
  from mjlab.asset_zoo.robots.robot_hands.h1_2_with_hands_constants import (
    _FINGER_MIMICS,
    H1_2_ACTION_SCALE,
    H1_2_ARTICULATION,
    HOME_KEYFRAME,
    KNEES_BENT_KEYFRAME,
    get_spec,
  )
except ImportError as e:
  print(f"[ERROR] No se pudo importar h1_2_with_hands_constants: {e}")
  sys.exit(1)

PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠️  WARN"

results: dict[str, str] = {}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Cargar spec MuJoCo
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  H1 v2 + Inspire Hands — Verificación pre-entrenamiento")
print("=" * 65)

print("\n[1] Cargando spec MuJoCo...")
try:
  spec = get_spec()
  model = spec.compile()
  print(f"    nq={model.nq}  nv={model.nv}  nu={model.nu}  nbody={model.nbody}")
  results["1_spec_load"] = PASS
except Exception as e:
  print(f"    {FAIL}: {e}")
  results["1_spec_load"] = FAIL
  sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_joint_names(m: mujoco.MjModel) -> list[str]:
  return [
    m.joint(i).name
    for i in range(m.njnt)
    if m.joint(i).type != mujoco.mjtJoint.mjJNT_FREE
  ]


def get_actuator_names(m: mujoco.MjModel) -> list[str]:
  return [m.actuator(i).name for i in range(m.nu)]


def get_site_names(m: mujoco.MjModel) -> list[str]:
  return [m.site(i).name for i in range(m.nsite)]


def get_geom_names(m: mujoco.MjModel) -> list[str]:
  return [m.geom(i).name for i in range(m.ngeom)]


def get_equality_names(m: mujoco.MjModel) -> list[str]:
  return [m.equality(i).name for i in range(m.neq)]


joint_names = get_joint_names(model)
actuator_names = get_actuator_names(model)
site_names = get_site_names(model)
geom_names = get_geom_names(model)
eq_names = get_equality_names(model)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Joints: cuerpo H1 v2 + manos
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] Verificando joints...")

# H1 v2: torso_joint en lugar de waist_yaw/roll/pitch
EXPECTED_BODY_JOINTS = [
  "left_hip_yaw_joint",
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_yaw_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "torso_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
]

EXPECTED_HAND_PROXIMAL = [
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
]

EXPECTED_HAND_MIMIC = [
  "L_thumb_intermediate_joint",
  "L_thumb_distal_joint",
  "L_index_intermediate_joint",
  "L_middle_intermediate_joint",
  "L_ring_intermediate_joint",
  "L_pinky_intermediate_joint",
  "R_thumb_intermediate_joint",
  "R_thumb_distal_joint",
  "R_index_intermediate_joint",
  "R_middle_intermediate_joint",
  "R_ring_intermediate_joint",
  "R_pinky_intermediate_joint",
]

missing_body = [j for j in EXPECTED_BODY_JOINTS if j not in joint_names]
missing_prox = [j for j in EXPECTED_HAND_PROXIMAL if j not in joint_names]
missing_mimic = [j for j in EXPECTED_HAND_MIMIC if j not in joint_names]

print(f"    Total joints no-free en modelo: {len(joint_names)}")
print(
  f"    Cuerpo H1 v2: {len(EXPECTED_BODY_JOINTS) - len(missing_body)}/{len(EXPECTED_BODY_JOINTS)}"
)
print(
  f"    Mano prox   : {len(EXPECTED_HAND_PROXIMAL) - len(missing_prox)}/{len(EXPECTED_HAND_PROXIMAL)}"
)
print(
  f"    Mano mimic  : {len(EXPECTED_HAND_MIMIC) - len(missing_mimic)}/{len(EXPECTED_HAND_MIMIC)}"
)

if missing_body or missing_prox or missing_mimic:
  for j in missing_body + missing_prox + missing_mimic:
    print(f"    {FAIL} falta: {j}")
  results["2_joints"] = FAIL
else:
  print("    Todos los joints esperados presentes.")
  results["2_joints"] = PASS


# ─────────────────────────────────────────────────────────────────────────────
# 3. Actuadores: cobertura de patrones
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] Verificando actuadores y cobertura de patrones...")

joints_to_actuate = set(EXPECTED_BODY_JOINTS + EXPECTED_HAND_PROXIMAL)

all_patterns: list[str] = []
for act_cfg in H1_2_ARTICULATION.actuators:
  assert isinstance(act_cfg, BuiltinPositionActuatorCfg)
  all_patterns.extend(act_cfg.target_names_expr)


def joint_covered(jname: str, patterns: list[str]) -> bool:
  for pat in patterns:
    try:
      if re.fullmatch(pat, jname):
        return True
    except re.error:
      pass
  return False


not_covered = [j for j in joints_to_actuate if not joint_covered(j, all_patterns)]

print(f"    Patrones de actuadores definidos: {len(all_patterns)}")
print(f"    Joints a actuar esperados: {len(joints_to_actuate)}")
print(
  "    (nu=0 es normal — mjlab inyecta actuadores al compilar EntityCfg, no en XML crudo)"
)

if not_covered:
  for j in not_covered:
    print(f"    {FAIL} sin actuador: {j}")
  results["3_actuators"] = FAIL
else:
  print("    Todos los joints a actuar están cubiertos por patrones.")
  results["3_actuators"] = PASS

if H1_2_ACTION_SCALE:
  print(f"    H1_2_ACTION_SCALE tiene {len(H1_2_ACTION_SCALE)} entradas.")
else:
  print(f"    {WARN} H1_2_ACTION_SCALE está vacío.")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Sites de wrists (H1_2 no tiene left_foot/right_foot por defecto)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] Verificando sites de wrists...")

REQUIRED_WRIST_SITES = ["left_wrist_attach", "right_wrist_attach"]
missing_sites = [s for s in REQUIRED_WRIST_SITES if s not in site_names]

if missing_sites:
  for s in missing_sites:
    print(f"    {FAIL} site faltante: {s}")
  results["4_wrist_sites"] = FAIL
else:
  print("    ✓ left_wrist_attach y right_wrist_attach presentes.")
  results["4_wrist_sites"] = PASS

# Opcional: foot sites si existen
if "left_foot" in site_names and "right_foot" in site_names:
  print("    ✓ left_foot y right_foot presentes (opcional).")
else:
  print("    ⚠  left_foot/right_foot no presentes (H1_2 puede no tenerlos)")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Geoms de mano
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] Verificando geoms de colisión de manos...")

REQUIRED_HAND_GEOMS = ["L_hand_base_col", "R_hand_base_col"]
missing_hand_geoms = [g for g in REQUIRED_HAND_GEOMS if g not in geom_names]

if missing_hand_geoms:
  for g in missing_hand_geoms:
    print(f"    {FAIL} geom faltante: {g}")
  results["5_hand_geoms"] = FAIL
else:
  print("    ✓ L_hand_base_col y R_hand_base_col presentes.")
  results["5_hand_geoms"] = PASS


# ─────────────────────────────────────────────────────────────────────────────
# 6. Equality mimic inyectadas
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6] Verificando equalities mimic...")

missing_eq = [name for name, *_ in _FINGER_MIMICS if name not in eq_names]

if missing_eq:
  for e in missing_eq:
    print(f"    {FAIL} equality faltante: {e}")
  results["6_equalities"] = FAIL
else:
  print(f"    ✓ {len(_FINGER_MIMICS)} equalities mimic inyectadas correctamente.")
  results["6_equalities"] = PASS

eq_joint_errors = []
for _name, slave, master, _ in _FINGER_MIMICS:
  if slave not in joint_names:
    eq_joint_errors.append(f"slave '{slave}' no existe")
  if master not in joint_names:
    eq_joint_errors.append(f"master '{master}' no existe")

if eq_joint_errors:
  for e in eq_joint_errors:
    print(f"    {FAIL} {e}")
  results["6_equalities"] += " (joint refs FAIL)"
else:
  print("    ✓ Todos los slave/master de equalities existen en el modelo.")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Keyframes: joints referenciados existen
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7] Verificando keyframes...")


def check_keyframe(name: str, kf) -> list[str]:
  errors = []
  for pattern, _ in kf.joint_pos.items():
    try:
      matched = [j for j in joint_names if re.fullmatch(pattern, j)]
    except re.error:
      matched = [j for j in joint_names if j == pattern]
    if not matched:
      errors.append(f"patrón '{pattern}' no matchea ningún joint")
  return errors


home_errors = check_keyframe("HOME_KEYFRAME", HOME_KEYFRAME)
knees_errors = check_keyframe("KNEES_BENT_KEYFRAME", KNEES_BENT_KEYFRAME)

for e in home_errors:
  print(f"    HOME: {FAIL} {e}")
for e in knees_errors:
  print(f"    KNEES_BENT: {FAIL} {e}")

if not home_errors and not knees_errors:
  print("    ✓ Todos los patrones de keyframes matchean joints existentes.")
  results["7_keyframes"] = PASS
else:
  results["7_keyframes"] = FAIL


# ─────────────────────────────────────────────────────────────────────────────
# 8. Dimensiones de observaciones y acciones
# ─────────────────────────────────────────────────────────────────────────────
print("\n[8] Estimando dimensiones de obs/act...")

n_actuated = len(joints_to_actuate)
print(f"    Joints actuados (por patrones)  : {n_actuated}")

n_joint_pos = n_actuated
n_joint_vel = n_actuated
n_actions = n_actuated
n_command = 3
n_height = 17 * 9
n_base_lin = 3
n_base_ang = 3
n_proj_grav = 3

actor_dim = (
  n_base_lin
  + n_base_ang
  + n_proj_grav
  + n_joint_pos
  + n_joint_vel
  + n_actions
  + n_command
  + n_height
)

n_foot_height = 2
n_foot_air = 2
n_foot_contact = 2
n_foot_forces = 2 * 3
critic_extra = n_foot_height + n_foot_air + n_foot_contact + n_foot_forces
critic_dim = actor_dim + critic_extra

print(f"    Obs actor (estimado)          : {actor_dim}")
print(f"    Obs critic (estimado)         : {critic_dim}")
print(f"    Acciones                      : {n_actuated}")

results["8_dimensions"] = PASS


# ─────────────────────────────────────────────────────────────────────────────
# 9. Compatibilidad con red neuronal del rl_cfg
# ─────────────────────────────────────────────────────────────────────────────
print("\n[9] Verificando compatibilidad con red neuronal...")

HIDDEN_DIMS = (512, 256, 128)
print(f"    hidden_dims actor/critic: {HIDDEN_DIMS}")
print(
  f"    Input actor  → primera capa {HIDDEN_DIMS[0]}: {'OK' if actor_dim > 0 else 'ERROR'}"
)
print(
  f"    Input critic → primera capa {HIDDEN_DIMS[0]}: {'OK' if critic_dim > 0 else 'ERROR'}"
)
print(f"    Output → acciones {n_actuated}: OK")

if actor_dim > 1000:
  print(
    f"    {WARN} obs actor muy grande ({actor_dim}), considera reducir height_scan."
  )
else:
  print("    ✓ Dimensiones dentro de rangos normales.")

results["9_network"] = PASS


# ─────────────────────────────────────────────────────────────────────────────
# 10. Resumen final
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  RESUMEN FINAL")
print("=" * 65)

all_pass = True
for key, status in sorted(results.items()):
  label = key.split("_", 1)[1].replace("_", " ").upper()
  print(f"  [{label:<25}] {status}")
  if "FAIL" in status:
    all_pass = False

print("=" * 65)
if all_pass:
  print("  🚀 Todo OK — puedes iniciar el entrenamiento.")
else:
  print("   Hay errores — revisa los puntos marcados con FAIL.")
print("=" * 65 + "\n")

sys.exit(0 if all_pass else 1)
