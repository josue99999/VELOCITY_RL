"""
g1_hands_inspect.py
Carga la mano Inspire izquierda y permite mover cada dedo
con sliders interactivos. También muestra los sistemas de
referencia principales de cada body.

Uso:
  python g1_hands_inspect.py
"""

import os
import time
import threading
import mujoco
import mujoco.viewer

# ── Ruta al XML de la mano ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HAND_XML = os.path.join(
  BASE_DIR, "..", "..", "asset_zoo", "robots", "robot_hands", "inspire_hand_right.xml"
)

# ── Secuencia demo: mueve cada dedo uno por uno ───────────────────────────────
# (descripción, índice_actuador_o_None, valor_objetivo, duración_seg)
DEMO_SEQUENCE = [
  ("Pulgar YAW   → cierra", 0, 1.3, 1.5),
  ("Pulgar YAW   → abre", 0, 0.0, 1.0),
  ("Pulgar PITCH → cierra", 1, 0.5, 1.5),
  ("Pulgar PITCH → abre", 1, 0.0, 1.0),
  ("Índice       → cierra", 2, 1.7, 1.5),
  ("Índice       → abre", 2, 0.0, 1.0),
  ("Corazón      → cierra", 3, 1.7, 1.5),
  ("Corazón      → abre", 3, 0.0, 1.0),
  ("Anular       → cierra", 4, 1.7, 1.5),
  ("Anular       → abre", 4, 0.0, 1.0),
  ("Meñique      → cierra", 5, 1.7, 1.5),
  ("Meñique      → abre", 5, 0.0, 1.0),
  ("TODOS        → cierran", None, 1.5, 2.0),
  ("TODOS        → abren", None, 0.0, 2.0),
]


def print_info(model):
  print("\n── Modelo ────────────────────────────────────────────────────")
  print(f"  nq (DOF pos)   : {model.nq}")
  print(f"  nv (DOF vel)   : {model.nv}")
  print(f"  nu (actuadores): {model.nu}")
  print(f"  nbody          : {model.nbody}")

  print("\n── Joints ────────────────────────────────────────────────────")
  for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    lo, hi = model.jnt_range[i]
    print(f"  [{i:2d}] {name:<42s}  range=[{lo:.2f}, {hi:.2f}]")

  print("\n── Actuadores ────────────────────────────────────────────────")
  for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    lo, hi = model.actuator_ctrlrange[i]
    print(f"  [{i}] {name:<42s}  ctrl=[{lo:.2f}, {hi:.2f}]")

  print("\n── Sites (fingertips) ────────────────────────────────────────")
  for i in range(model.nsite):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
    print(f"  [{i}] {name}")


def run_demo(model, data, v):
  """Hilo que ejecuta la secuencia automática de movimientos."""
  dt = model.opt.timestep
  time.sleep(1.0)  # espera a que el visor esté listo

  print("\n── Demo automática ───────────────────────────────────────────")
  for desc, act_idx, target, duration in DEMO_SEQUENCE:
    if not v.is_running():
      break
    print(f"  ▶  {desc}")
    n_steps = int(duration / dt)
    for _ in range(n_steps):
      if not v.is_running():
        return
      if act_idx is None:
        for i in range(1, model.nu):  # todos excepto thumb_yaw
          data.ctrl[i] = target
      else:
        data.ctrl[act_idx] = target
      mujoco.mj_step(model, data)
      v.sync()

  print("\n✅ Demo completada. Usa los sliders del panel para control manual.")


if __name__ == "__main__":
  print(f"Cargando modelo:\n  {HAND_XML}\n")
  model = mujoco.MjModel.from_xml_path(HAND_XML)
  data = mujoco.MjData(model)

  print_info(model)

  # Mano abierta al inicio
  data.ctrl[:] = 0.0
  mujoco.mj_forward(model, data)

  print("\nAbriendo visor...")
  print("  • Tecla F  → activa/desactiva ejes de cada body")
  print("  • Tecla S  → muestra/oculta sites (fingertips)")
  print("  • Panel derecho → sliders de control manual\n")

  with mujoco.viewer.launch_passive(model, data) as v:
    # Mostrar sistemas de referencia de cada body
    v.opt.frame = mujoco.mjtFrame.mjFRAME_BODY.value
    # Mostrar joints y sites
    v.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT.value] = True

    t = threading.Thread(target=run_demo, args=(model, data, v), daemon=True)
    t.start()

    while v.is_running():
      mujoco.mj_step(model, data)
      v.sync()
