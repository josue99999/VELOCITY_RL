"""
g1_inspect.py
Visualiza el G1 (sin manos) con piso, barre todos los joints uno a uno
y muestra los sensores en tiempo real por terminal.
Ahora incluye visualización de sistemas de referencia en los wrists.

Uso:
  python g1_inspect.py
"""

import os
import time
import threading
import numpy as np
import mujoco
import mujoco.viewer

# ── Rutas ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
G1_XML = os.path.join(
  BASE_DIR, "..", "..", "asset_zoo", "robots", "robot_hands", "xmls", "g1.xml"
)

# ── Joints a barrer (ignoramos floating_base_joint) ───────────────────────────
SKIP_JOINTS = {"floating_base_joint"}

SWEEP_FRAC = 0.4
SWEEP_TIME = 1.5

# ── Sites de referencia a monitorear ──────────────────────────────────────────
WRIST_SITES = ["left_wrist_attach", "right_wrist_attach"]


def print_info(model):
  print("\n── Modelo ────────────────────────────────────────────────────")
  print(f"  nq  : {model.nq}   nv  : {model.nv}")
  print(f"  nu  : {model.nu}   nbody: {model.nbody}")
  print(f"  nsensor: {model.nsensor}")

  print("\n── Joints ────────────────────────────────────────────────────")
  for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    lo, hi = model.jnt_range[i]
    print(f"  [{i:2d}] {name:<40s} range=[{lo:6.3f}, {hi:6.3f}]")

  print("\n── Sensores ──────────────────────────────────────────────────")
  for i in range(model.nsensor):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
    print(f"  [{i}] {name}  (dim={model.sensor_dim[i]})")

  # ── NUEVO: info de sites de wrist ────────────────────────────────────────
  print("\n── Sites de referencia (wrists) ──────────────────────────────")
  for site_name in WRIST_SITES:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id == -1:
      print(f"  [!] '{site_name}' NO encontrado en el modelo")
    else:
      print(f"  [✓] '{site_name}' → site_id={site_id}")


def print_sensors(model, data):
  idx = 0
  print("\n── Sensores ──────────────────────────────────────────────────")
  for i in range(model.nsensor):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
    dim = model.sensor_dim[i]
    vals = data.sensordata[idx : idx + dim]
    print(f"  {name:<25s}: {np.round(vals, 4)}")
    idx += dim


# ── NUEVO: imprime pose de los sistemas de referencia ─────────────────────────
def print_wrist_frames(model, data):
  print("\n── Sistemas de referencia (wrists) ───────────────────────────")
  for site_name in WRIST_SITES:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id == -1:
      print(f"  [!] '{site_name}' no encontrado")
      continue
    pos = data.site_xpos[site_id]
    rot = data.site_xmat[site_id].reshape(3, 3)
    print(f"\n  {site_name}:")
    print(f"    Posición (world): [{pos[0]:7.4f}  {pos[1]:7.4f}  {pos[2]:7.4f}]")
    print(
      f"    Eje X (+rojo  ): [{rot[0, 0]:6.3f}  {rot[1, 0]:6.3f}  {rot[2, 0]:6.3f}]"
    )
    print(
      f"    Eje Y (+verde ): [{rot[0, 1]:6.3f}  {rot[1, 1]:6.3f}  {rot[2, 1]:6.3f}]"
    )
    print(
      f"    Eje Z (+azul  ): [{rot[0, 2]:6.3f}  {rot[1, 2]:6.3f}  {rot[2, 2]:6.3f}]"
    )


def sweep_joints(model, data, v):
  time.sleep(1.5)

  data.qpos[2] = 1.0
  data.qpos[3] = 1.0
  data.qpos[4] = 0.0
  data.qpos[5] = 0.0
  data.qpos[6] = 0.0
  mujoco.mj_forward(model, data)

  # ── NUEVO: imprimir frames en pose neutra ─────────────────────────────────
  print_wrist_frames(model, data)

  print("\n── Barrido de joints ─────────────────────────────────────────")
  dt = model.opt.timestep

  for i in range(model.njnt):
    if not v.is_running():
      break
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    if name in SKIP_JOINTS:
      continue

    jtype = model.jnt_type[i]
    if jtype == mujoco.mjtJoint.mjJNT_FREE:
      continue

    lo, hi = model.jnt_range[i]
    center = (lo + hi) / 2.0
    amp = (hi - lo) * SWEEP_FRAC
    qadr = model.jnt_qposadr[i]

    print(f"  ▶  {name}")

    target = center - amp
    steps = int(SWEEP_TIME / 2 / dt)
    for s in range(steps):
      if not v.is_running():
        return
      alpha = s / steps
      data.qpos[qadr] = data.qpos[qadr] + alpha * (target - data.qpos[qadr]) * 0.1
      mujoco.mj_forward(model, data)
      v.sync()

    target = center + amp
    for s in range(steps):
      if not v.is_running():
        return
      alpha = s / steps
      data.qpos[qadr] = data.qpos[qadr] + alpha * (target - data.qpos[qadr]) * 0.1
      mujoco.mj_forward(model, data)
      v.sync()

    data.qpos[qadr] = center
    mujoco.mj_forward(model, data)

  print("\n✅ Barrido completado.")
  print_sensors(model, data)

  # ── NUEVO: frames en pose final ───────────────────────────────────────────
  print_wrist_frames(model, data)


if __name__ == "__main__":
  print(f"Cargando G1:\n  {G1_XML}\n")
  model = mujoco.MjModel.from_xml_path(G1_XML)
  data = mujoco.MjData(model)

  print_info(model)

  model.opt.gravity[:] = [0, 0, 0]
  data.qpos[2] = 1.0
  data.qpos[3] = 1.0
  mujoco.mj_forward(model, data)

  print("\nAbriendo visor...")
  print("  • Sin gravedad para el test de joints")
  print("  • La demo barre cada joint automáticamente")
  print("  • Al final se imprimen los valores de los sensores")
  print("  • Sistemas de referencia de wrists visibles como ejes RGB\n")

  with mujoco.viewer.launch_passive(model, data) as v:
    # ── NUEVO: activar visualización de frames de sites ───────────────────
    v.opt.frame = mujoco.mjtFrame.mjFRAME_BODY  # ejes XYZ en cada site
    v.opt.label = mujoco.mjtLabel.mjLABEL_SITE  # nombre del site
    v.opt.geomgroup[5] = True  # grupo 5 visible (frames geom)
    # Ajuste de cámara para ver mejor los brazos
    v.cam.distance = 2.8
    v.cam.elevation = -15
    v.cam.azimuth = 120

    t = threading.Thread(target=sweep_joints, args=(model, data, v), daemon=True)
    t.start()

    while v.is_running():
      mujoco.mj_step(model, data)
      v.sync()
