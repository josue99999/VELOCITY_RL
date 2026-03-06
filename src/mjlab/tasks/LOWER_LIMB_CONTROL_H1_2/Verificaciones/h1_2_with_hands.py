"""
h1_2_with_hands.py
==================
Carga h1_2_hand.xml (vía get_spec con mimic equalities) y:
  1. Imprime info completa (joints, sites, sensores)
  2. Visualiza el robot en pose HOME sin gravedad
  3. Barre joints del H1 v2 uno por uno
  4. Barre joints de mano izquierda uno por uno
  5. Barre joints de mano derecha uno por uno
  6. Muestra frames de todos los bodies
  7. Imprime valores de sensores al terminar

Uso:
  cd tasks/LOWER_LIMB_CONTROL_H1_2/Verificaciones
  python h1_2_with_hands.py
"""

import re

# ── Usar get_spec para cargar con assets y mimic equalities ───────────────────
import sys
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from mjlab.asset_zoo.robots.robot_hands.h1_2_with_hands_constants import get_spec

# ── Lock para evitar race condition ───────────────────────────────────────────
sim_lock = threading.Lock()

SKIP_JOINTS = {"floating_base_joint"}

HAND_JOINTS_L = {
  "L_thumb_proximal_yaw_joint",
  "L_thumb_proximal_pitch_joint",
  "L_index_proximal_joint",
  "L_middle_proximal_joint",
  "L_ring_proximal_joint",
  "L_pinky_proximal_joint",
}
HAND_JOINTS_R = {
  "R_thumb_proximal_yaw_joint",
  "R_thumb_proximal_pitch_joint",
  "R_index_proximal_joint",
  "R_middle_proximal_joint",
  "R_ring_proximal_joint",
  "R_pinky_proximal_joint",
}
MIMIC_JOINTS = {
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
}
HAND_JOINTS_ALL = HAND_JOINTS_L | HAND_JOINTS_R

SWEEP_FRAC = 0.35
SWEEP_TIME = 1.2


def print_info(model):
  print("\n══ MODELO H1 v2 + Inspire Hands ═══════════════════════════════════")
  print(f"  nq={model.nq}  nv={model.nv}  nu={model.nu}")
  print(f"  nbody={model.nbody}  njnt={model.njnt}  nsensor={model.nsensor}")

  print("\n── Joints H1 v2 (cuerpo) ───────────────────────────────────────────")
  for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    if name in SKIP_JOINTS or name in HAND_JOINTS_ALL or name in MIMIC_JOINTS:
      continue
    if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
      continue
    lo, hi = model.jnt_range[i]
    print(f"  [{i:2d}] {name:<44s} [{lo:6.3f}, {hi:6.3f}]")

  print("\n── Joints mano IZQUIERDA ─────────────────────────────────────")
  for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    if not (name and name.startswith("L_")):
      continue
    lo, hi = model.jnt_range[i]
    tag = " ← mimic" if name in MIMIC_JOINTS else ""
    print(f"  [{i:2d}] {name:<44s} [{lo:6.3f}, {hi:6.3f}]{tag}")

  print("\n── Joints mano DERECHA ───────────────────────────────────────")
  for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    if not (name and name.startswith("R_")):
      continue
    lo, hi = model.jnt_range[i]
    tag = " ← mimic" if name in MIMIC_JOINTS else ""
    print(f"  [{i:2d}] {name:<44s} [{lo:6.3f}, {hi:6.3f}]{tag}")

  print("\n── Sites ─────────────────────────────────────────────────────")
  for i in range(model.nsite):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
    print(f"  [{i}] {name}")

  print("\n── Sensores ──────────────────────────────────────────────────")
  for i in range(model.nsensor):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
    print(f"  [{i}] {name}  dim={model.sensor_dim[i]}")


def print_sensors(model, data):
  print("\n── Valores sensores ──────────────────────────────────────────")
  idx = 0
  for i in range(model.nsensor):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
    dim = model.sensor_dim[i]
    with sim_lock:
      vals = data.sensordata[idx : idx + dim].copy()
    print(f"  {name:<25s}: {np.round(vals, 4)}")
    idx += dim


def set_home_pose(model, data):
  """Pose HOME para H1 v2 (torso_joint en lugar de waist)."""
  home = {
    r".*_hip_pitch_joint": -0.1,
    r".*_knee_joint": 0.3,
    r".*_ankle_pitch_joint": -0.2,
    r".*_ankle_roll_joint": 0.0,
    r"torso_joint": 0.0,
    r".*_shoulder_pitch_joint": 0.2,
    r".*_elbow_joint": 0.6,
    r"left_shoulder_roll_joint": 0.2,
    r"right_shoulder_roll_joint": -0.2,
  }
  for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    if not name:
      continue
    for pattern, val in home.items():
      if re.fullmatch(pattern, name):
        data.qpos[model.jnt_qposadr[i]] = val
        break

  data.qpos[2] = 1.03
  data.qpos[3] = 1.0
  mujoco.mj_forward(model, data)


def sweep_joints(model, data, v):
  time.sleep(1.5)
  dt = model.opt.timestep

  def interp(qadr, start, end, secs):
    steps = max(1, int(secs / dt))
    for s in range(steps):
      if not v.is_running():
        return False
      with sim_lock:
        data.qpos[qadr] = start + (s + 1) / steps * (end - start)
        mujoco.mj_forward(model, data)
      v.sync()
      time.sleep(dt)
    return True

  def sweep_body(i):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    lo, hi = model.jnt_range[i]
    center = (lo + hi) / 2.0
    amp = (hi - lo) * SWEEP_FRAC
    qadr = model.jnt_qposadr[i]
    with sim_lock:
      orig = float(data.qpos[qadr])
    print(f"  ▶  {name}")
    if not interp(qadr, orig, center - amp, SWEEP_TIME / 2):
      return False
    if not interp(qadr, center - amp, center + amp, SWEEP_TIME):
      return False
    if not interp(qadr, center + amp, orig, SWEEP_TIME / 2):
      return False
    return True

  def sweep_hand(i):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    lo, hi = model.jnt_range[i]
    qadr = model.jnt_qposadr[i]
    print(f"  ▶  {name}")
    if not interp(qadr, 0.0, hi, SWEEP_TIME):
      return False
    if not interp(qadr, hi, 0.0, SWEEP_TIME):
      return False
    return True

  print("\n══ Barrido joints H1 v2 ════════════════════════════════════════")
  for i in range(model.njnt):
    if not v.is_running():
      break
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    if not name:
      continue
    if name in SKIP_JOINTS or name in HAND_JOINTS_ALL or name in MIMIC_JOINTS:
      continue
    if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
      continue
    if not sweep_body(i):
      break

  print("\n══ Barrido mano IZQUIERDA ═══════════════════════════════════")
  for i in range(model.njnt):
    if not v.is_running():
      break
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    if name not in HAND_JOINTS_L:
      continue
    if not sweep_hand(i):
      break

  print("\n══ Barrido mano DERECHA ═════════════════════════════════════")
  for i in range(model.njnt):
    if not v.is_running():
      break
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    if name not in HAND_JOINTS_R:
      continue
    if not sweep_hand(i):
      break

  print("\n✅ Barrido completo.")
  print_sensors(model, data)
  print("\n  → Usa los sliders del panel para control manual.")


if __name__ == "__main__":
  print("Cargando H1 v2 + Inspire Hands (get_spec con mimic equalities)...\n")
  spec = get_spec()
  model = spec.compile()
  data = mujoco.MjData(model)

  print_info(model)

  model.opt.gravity[:] = [0, 0, 0]
  set_home_pose(model, data)

  print("\nAbriendo visor...")
  print("  • Gravedad desactivada para el test")
  print("  • Secuencia: joints H1 v2 → mano izq → mano der")
  print("  • mjFRAME_BODY activo: verás ejes RGB en cada body\n")

  with mujoco.viewer.launch_passive(model, data) as v:
    v.opt.frame = mujoco.mjtFrame.mjFRAME_BODY.value

    t = threading.Thread(target=sweep_joints, args=(model, data, v), daemon=True)
    t.start()

    while v.is_running():
      with sim_lock:
        mujoco.mj_step(model, data)
      v.sync()
