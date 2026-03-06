"""
verify_h1_2_constants.py
========================
Verifica que h1_2_with_hands_constants compila correctamente y que:
  - Los joints actuados existen en el modelo
  - Las equalities mimic se inyectaron
  - La pose inicial (keyframe) es válida
  - El modelo no explota al hacer mj_forward

Uso:
  python verify_h1_2_constants.py
"""

import re
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(
  0,
  str(Path(__file__).resolve().parents[3] / "src"),
)

from mjlab.asset_zoo.robots.robot_hands.h1_2_with_hands_constants import (
  _FINGER_MIMICS,
  H1_2_ARTICULATION,
  KNEES_BENT_KEYFRAME,
  get_spec,
)


def section(title: str):
  print(f"\n{'─' * 60}")
  print(f"  {title}")
  print(f"{'─' * 60}")


def check_mark(ok: bool) -> str:
  return "✓" if ok else "✗ FALLO"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Compilar el spec
# ─────────────────────────────────────────────────────────────────────────────
section("1. Compilando modelo H1 v2 + Inspire Hands")
try:
  spec = get_spec()
  model = spec.compile()
  data = mujoco.MjData(model)
  print(f"  {check_mark(True)}  Modelo compilado")
  print(f"      nq={model.nq}  nv={model.nv}  nu={model.nu}  nbody={model.nbody}")
  print(f"      njnt={model.njnt}  neq={model.neq}")
except Exception as e:
  print(f"  ✗ ERROR al compilar: {e}")
  raise SystemExit(1) from e


# ─────────────────────────────────────────────────────────────────────────────
# 2. Verificar actuadores
# ─────────────────────────────────────────────────────────────────────────────
section("2. Actuadores")

all_joint_names = [
  mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)
]

for actuator_cfg in H1_2_ARTICULATION.actuators:
  for pattern in actuator_cfg.target_names_expr:
    matches = [j for j in all_joint_names if re.fullmatch(pattern, j)]
    ok = len(matches) > 0
    print(f"  {check_mark(ok)}  pattern '{pattern}' → {len(matches)} joints")
    if not ok:
      print("       ⚠ No se encontró ningún joint con este patrón")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Verificar equalities mimic
# ─────────────────────────────────────────────────────────────────────────────
section("3. Equalities mimic (dedos)")

eq_names = [
  mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, i) for i in range(model.neq)
]
print(f"  Equalities en modelo: {model.neq}")

for name, _slave, _master, _ratio in _FINGER_MIMICS:
  ok = name in eq_names
  print(f"  {check_mark(ok)}  {name}")

missing = [n for n, *_ in _FINGER_MIMICS if n not in eq_names]
if missing:
  print(f"\n  ⚠ Faltan {len(missing)} equalities — revisa get_spec()")
else:
  print("\n  Todos los mimics inyectados correctamente")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Verificar keyframe / pose inicial
# ─────────────────────────────────────────────────────────────────────────────
section("4. Pose inicial (KNEES_BENT_KEYFRAME)")

joint_pos_cfg = KNEES_BENT_KEYFRAME.joint_pos
for pattern, value in joint_pos_cfg.items():
  matches = [j for j in all_joint_names if re.fullmatch(pattern, j)]
  ok = len(matches) > 0
  print(f"  {check_mark(ok)}  '{pattern}' = {value} → {len(matches)} joints")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Aplicar pose inicial y hacer mj_forward (¿explota?)
# ─────────────────────────────────────────────────────────────────────────────
section("5. mj_forward con pose inicial")

# Posición del pelvis
data.qpos[0:3] = KNEES_BENT_KEYFRAME.pos  # x, y, z
data.qpos[3] = 1.0  # quat w (orientación neutra)

# Aplicar joint_pos del keyframe
for pattern, value in joint_pos_cfg.items():
  for i in range(model.njnt):
    jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    if jname and re.fullmatch(pattern, jname):
      jtype = model.jnt_type[i]
      if jtype != mujoco.mjtJoint.mjJNT_FREE:
        qadr = model.jnt_qposadr[i]
        data.qpos[qadr] = value

mujoco.mj_forward(model, data)

# Chequear NaN/Inf en posiciones y velocidades
nan_pos = np.any(np.isnan(data.qpos)) or np.any(np.isinf(data.qpos))
nan_vel = np.any(np.isnan(data.qvel)) or np.any(np.isinf(data.qvel))
print(f"  {check_mark(not nan_pos)}  qpos sin NaN/Inf")
print(f"  {check_mark(not nan_vel)}  qvel sin NaN/Inf")

# Altura del pelvis — H1_2 ~1.0–1.03
pelvis_z = data.qpos[2]
ok_height = 0.5 < pelvis_z < 1.5
print(f"  {check_mark(ok_height)}  Altura pelvis = {pelvis_z:.4f} m  (esperado ~1.0)")

# Sites de pies — H1_2 no tiene left_foot/right_foot por defecto; opcional
for site_name in ["left_foot", "right_foot"]:
  sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
  if sid >= 0:
    foot_z = data.site_xpos[sid][2]
    ok_foot = foot_z > -0.1
    print(f"  {check_mark(ok_foot)}  {site_name} z = {foot_z:.4f} m")
  else:
    print(f"  ⚠  {site_name} no existe en el modelo (opcional para H1_2)")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Resumen final
# ─────────────────────────────────────────────────────────────────────────────
section("Resumen H1 v2 + Inspire Hands")
print(f"  Joints totales  : {model.njnt}")
print(f"  Actuadores (nu) : {model.nu}")
print(f"  Equalities (neq): {model.neq}")
print(f"  Bodies         : {model.nbody}")
print(f"  Sites          : {model.nsite}")
print()

# Desglose actuadores cuerpo vs manos
body_joints = [
  j
  for j in all_joint_names
  if re.search(r"hip|knee|ankle|torso|shoulder|elbow|wrist", j or "")
]
finger_joints = [
  j for j in all_joint_names if re.search(r"thumb|index|middle|ring|pinky", j or "")
]
print(f"  Joints cuerpo   : {len(body_joints)}")
print(f"  Joints dedos   : {len(finger_joints)}")
print()
print("  Si todo muestra ✓ → el modelo está listo para RL")
print()
