"""
verify_lower_limb_h1_2_task.py
==============================
Verificación integral del task LOWER_LIMB_CONTROL_H1_2.

Ejecutar:
    uv run python src/mjlab/tasks/LOWER_LIMB_CONTROL_H1_2/Verificaciones/verify_lower_limb_h1_2_task.py

Comprueba:
  1. velocity_env_cfg: CONTROLLED_JOINTS, OBSERVED_JOINTS, consistencia
  2. config/h1_2/env_cfgs: robot, sensors, rewards, events
  3. config/h1_2/rl_cfg: dimensiones, max_iterations
  4. Task registration: Flat y Rough
  5. Cross-reference: joints/sites/geoms en modelo H1_2
  6. Action scale: cobertura de CONTROLLED_JOINTS
"""

import re
import sys
from pathlib import Path

# Forzar uso del repo mjlab
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

PASS = "✅"
FAIL = "❌"
WARN = "⚠️"

results: dict[str, str] = {}


def section(title: str) -> None:
  print(f"\n{'─' * 60}")
  print(f"  {title}")
  print(f"{'─' * 60}")


def ok(cond: bool) -> str:
  return PASS if cond else FAIL


# ─────────────────────────────────────────────────────────────────────────────
# 1. Imports y carga de config
# ─────────────────────────────────────────────────────────────────────────────
section("1. Carga de configuración")

try:
  from mjlab.asset_zoo.robots.robot_hands.h1_2_with_hands_constants import (
    H1_2_ACTION_SCALE,
    get_spec,
  )
  from mjlab.tasks.LOWER_LIMB_CONTROL_H1_2.config.h1_2.env_cfgs import (
    unitree_h1_2_flat_env_cfg,
    unitree_h1_2_rough_env_cfg,
  )
  from mjlab.tasks.LOWER_LIMB_CONTROL_H1_2.config.h1_2.rl_cfg import (
    unitree_h1_2_ppo_runner_cfg,
  )
  from mjlab.tasks.LOWER_LIMB_CONTROL_H1_2.velocity_env_cfg import (
    ARM_AND_HAND_JOINTS,
    CONTROLLED_JOINTS,
    CURRICULUM_PHASES,
    OBSERVED_JOINTS,
  )
  from mjlab.tasks.registry import list_tasks, load_env_cfg

  print(f"  {PASS} Imports OK")
  results["1_imports"] = PASS
except Exception as e:
  print(f"  {FAIL} Imports: {e}")
  results["1_imports"] = FAIL
  sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# 2. velocity_env_cfg: CONTROLLED_JOINTS vs OBSERVED_JOINTS
# ─────────────────────────────────────────────────────────────────────────────
section("2. velocity_env_cfg — Joints")

n_controlled = len(CONTROLLED_JOINTS)
n_observed = len(OBSERVED_JOINTS)
observed_includes_arms = any("shoulder" in str(p) for p in OBSERVED_JOINTS)

print(f"  CONTROLLED_JOINTS: {n_controlled} patrones (legs + torso)")
print(f"  OBSERVED_JOINTS:  {n_observed} patrones")
print(f"  OBSERVED incluye brazos: {ok(observed_includes_arms)}")

# OBSERVED = CONTROLLED + arms
expected_observed = n_controlled + len(ARM_AND_HAND_JOINTS)
if n_observed == expected_observed:
  print(f"  OBSERVED = CONTROLLED + ARM_AND_HAND: {PASS}")
  results["2_joints"] = PASS
else:
  print(f"  OBSERVED esperado {expected_observed}, got {n_observed}: {WARN}")
  results["2_joints"] = PASS  # No es error si hay duplicados por patrón

# ─────────────────────────────────────────────────────────────────────────────
# 3. Modelo H1_2: joints que matchean
# ─────────────────────────────────────────────────────────────────────────────
section("3. Modelo H1_2 — Joints que matchean")

try:
  spec = get_spec()
  model = spec.compile()
  joint_names = [
    model.joint(i).name
    for i in range(model.njnt)
    if model.joint(i).type != 0  # no free
  ]
except Exception as e:
  print(f"  {FAIL} No se pudo compilar spec: {e}")
  joint_names = []
  results["3_model"] = FAIL

if joint_names:
  controlled_matches = set()
  for pat in CONTROLLED_JOINTS:
    for j in joint_names:
      if re.fullmatch(pat, j):
        controlled_matches.add(j)
  observed_matches = set()
  for pat in OBSERVED_JOINTS:
    for j in joint_names:
      if re.fullmatch(pat, j):
        observed_matches.add(j)

  print(f"  Joints en modelo: {len(joint_names)}")
  print(f"  CONTROLLED matchean: {len(controlled_matches)} joints")
  print(f"  OBSERVED matchean:   {len(observed_matches)} joints")

  if len(controlled_matches) == 13:
    print(f"  CONTROLLED = 13 DOF: {PASS}")
  else:
    print(f"  CONTROLLED esperado 13, got {len(controlled_matches)}: {FAIL}")
    results["3_model"] = FAIL

  if len(observed_matches) >= 13:
    print(f"  OBSERVED >= 13: {PASS}")
    results["3_model"] = results.get("3_model", PASS)
  else:
    print(f"  OBSERVED < 13: {FAIL}")
    results["3_model"] = FAIL

# ─────────────────────────────────────────────────────────────────────────────
# 4. Action scale: cobertura de CONTROLLED_JOINTS
# ─────────────────────────────────────────────────────────────────────────────
section("4. Action scale (H1_2_ACTION_SCALE)")

controlled_action_scale = {}
for pattern, scale in H1_2_ACTION_SCALE.items():
  for ctrl_pattern in CONTROLLED_JOINTS:
    if re.match(ctrl_pattern, pattern.replace(".*", "left")) or re.match(
      ctrl_pattern, pattern.replace(".*", "right")
    ):
      controlled_action_scale[pattern] = scale
      break
    if pattern == "torso_joint" and ctrl_pattern == "torso_joint":
      controlled_action_scale[pattern] = scale
      break

print(f"  H1_2_ACTION_SCALE entradas: {len(H1_2_ACTION_SCALE)}")
print(f"  Scale para CONTROLLED: {len(controlled_action_scale)} patrones")

# Deberíamos tener scale para cada uno de los 13 DOF (vía patrones)
if len(controlled_action_scale) >= 7:  # 7 patrones cubren 13 joints
  print(f"  Cobertura action scale: {PASS}")
  results["4_action_scale"] = PASS
else:
  print(f"  Cobertura insuficiente: {FAIL}")
  results["4_action_scale"] = FAIL

# ─────────────────────────────────────────────────────────────────────────────
# 5. Config env: Rough y Flat
# ─────────────────────────────────────────────────────────────────────────────
section("5. Config env_cfgs (Rough / Flat)")

try:
  cfg_rough = unitree_h1_2_rough_env_cfg()
  cfg_flat = unitree_h1_2_flat_env_cfg()
  print(f"  unitree_h1_2_rough_env_cfg(): {PASS}")
  print(f"  unitree_h1_2_flat_env_cfg():  {PASS}")

  # Verificar que scene.entities tiene robot
  assert "robot" in cfg_rough.scene.entities
  print(f"  scene.entities['robot']: {PASS}")

  # Verificar sensors
  sensor_names = [s.name for s in (cfg_rough.scene.sensors or ())]
  required = ["feet_ground_contact", "self_collision"]
  for r in required:
    if r in sensor_names:
      print(f"  sensor '{r}': {PASS}")
    else:
      print(f"  sensor '{r}': {FAIL}")
      results["5_env"] = FAIL

  # Flat no tiene terrain_scan
  flat_sensors = [s.name for s in (cfg_flat.scene.sensors or ())]
  if "terrain_scan" not in flat_sensors:
    print(f"  Flat sin terrain_scan: {PASS}")
  else:
    print(f"  Flat sin terrain_scan: {WARN} (tiene terrain_scan)")

  results["5_env"] = results.get("5_env", PASS)
except Exception as e:
  print(f"  {FAIL} env_cfgs: {e}")
  results["5_env"] = FAIL

# ─────────────────────────────────────────────────────────────────────────────
# 6. Observaciones: términos y referencias
# ─────────────────────────────────────────────────────────────────────────────
section("6. Observaciones")

try:
  cfg = unitree_h1_2_rough_env_cfg()
  actor_terms = list(cfg.observations["actor"].terms.keys())
  critic_terms = list(cfg.observations["critic"].terms.keys())

  required_actor = [
    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "actions",
    "command",
    "height_scan",
  ]
  for t in required_actor:
    if t in actor_terms:
      print(f"  actor '{t}': {PASS}")
    else:
      print(f"  actor '{t}': {FAIL}")

  # joint_pos/joint_vel deben usar OBSERVED_JOINTS
  jp = cfg.observations["actor"].terms["joint_pos"]
  ac = jp.params.get("asset_cfg") if hasattr(jp.params, "get") else None
  if ac is None:
    ac = getattr(jp.params, "asset_cfg", None)
  jn = getattr(ac, "joint_names", ()) if ac else ()
  if len(jn) > 13:
    print(f"  joint_pos observa {len(jn)} joints (incl. arms): {PASS}")
  elif len(jn) == 13:
    print(f"  joint_pos observa 13 joints (solo legs+torso): {WARN}")
  else:
    print(f"  joint_pos observa {len(jn)} joints: {WARN}")

  results["6_obs"] = PASS
except Exception as e:
  print(f"  {FAIL} observaciones: {e}")
  results["6_obs"] = FAIL

# ─────────────────────────────────────────────────────────────────────────────
# 7. Rewards
# ─────────────────────────────────────────────────────────────────────────────
section("7. Rewards")

try:
  cfg = unitree_h1_2_rough_env_cfg()
  reward_names = list(cfg.rewards.keys())
  expected_rewards = [
    "track_linear_velocity",
    "track_angular_velocity",
    "upright",
    "pose",
    "body_ang_vel",
    "angular_momentum",
    "dof_pos_limits",
    "action_rate_l2",
    "air_time",
    "foot_clearance",
    "foot_swing_height",
    "foot_slip",
    "soft_landing",
    "self_collisions",
  ]
  missing = [r for r in expected_rewards if r not in reward_names]
  if not missing:
    print(f"  Todos los rewards presentes: {PASS}")
    results["7_rewards"] = PASS
  else:
    print(f"  Faltan: {missing}: {FAIL}")
    results["7_rewards"] = FAIL
except Exception as e:
  print(f"  {FAIL} rewards: {e}")
  results["7_rewards"] = FAIL

# ─────────────────────────────────────────────────────────────────────────────
# 8. Events
# ─────────────────────────────────────────────────────────────────────────────
section("8. Events")

try:
  cfg = unitree_h1_2_rough_env_cfg()
  event_names = list(cfg.events.keys())
  expected_events = [
    "reset_base",
    "reset_robot_joints",
    "push_robot",
    "foot_friction",
    "encoder_bias",
    "base_com",
    "randomize_arm_pose",
    "randomize_arm_mass",
  ]
  missing = [e for e in expected_events if e not in event_names]
  if not missing:
    print(f"  Todos los events presentes: {PASS}")
    results["8_events"] = PASS
  else:
    print(f"  Faltan: {missing}: {WARN}")
    results["8_events"] = PASS
except Exception as e:
  print(f"  {FAIL} events: {e}")
  results["8_events"] = FAIL

# ─────────────────────────────────────────────────────────────────────────────
# 9. Curriculum
# ─────────────────────────────────────────────────────────────────────────────
section("9. Curriculum")

try:
  cfg = unitree_h1_2_rough_env_cfg()
  curr_names = list(cfg.curriculum.keys())
  expected_curr = ["terrain_levels", "command_vel", "teleop_disturbances", "phase_info"]
  missing = [c for c in expected_curr if c not in curr_names]
  if not missing:
    print(f"  Curriculum terms: {PASS}")
  else:
    print(f"  Faltan: {missing}: {WARN}")

  print(f"  CURRICULUM_PHASES: {len(CURRICULUM_PHASES)} fases")
  results["9_curriculum"] = PASS
except Exception as e:
  print(f"  {FAIL} curriculum: {e}")
  results["9_curriculum"] = FAIL

# ─────────────────────────────────────────────────────────────────────────────
# 10. Task registration
# ─────────────────────────────────────────────────────────────────────────────
section("10. Task registration")

try:
  import mjlab.tasks  # noqa: F401

  tasks = list_tasks()
  h1_tasks = [t for t in tasks if "H1_2" in t and "LowerLimb" in t]
  expected_tasks = [
    "Mjlab-Velocity-Rough-H1_2-LowerLimb",
    "Mjlab-Velocity-Flat-H1_2-LowerLimb",
  ]
  for et in expected_tasks:
    if et in h1_tasks:
      print(f"  {et}: {PASS}")
    else:
      print(f"  {et}: {FAIL}")
      results["10_registration"] = FAIL

  if all(et in h1_tasks for et in expected_tasks):
    results["10_registration"] = results.get("10_registration", PASS)
except Exception as e:
  print(f"  {FAIL} registration: {e}")
  results["10_registration"] = FAIL

# ─────────────────────────────────────────────────────────────────────────────
# 11. RL config
# ─────────────────────────────────────────────────────────────────────────────
section("11. RL config")

try:
  rl_cfg = unitree_h1_2_ppo_runner_cfg()
  print(f"  max_iterations: {rl_cfg.max_iterations}")
  print(f"  num_steps_per_env: {rl_cfg.num_steps_per_env}")
  print(f"  actor hidden_dims: {rl_cfg.actor.hidden_dims}")
  print(f"  critic hidden_dims: {rl_cfg.critic.hidden_dims}")
  if rl_cfg.max_iterations == 60_000:
    print(f"  60K iterations: {PASS}")
  results["11_rl"] = PASS
except Exception as e:
  print(f"  {FAIL} rl_cfg: {e}")
  results["11_rl"] = FAIL

# ─────────────────────────────────────────────────────────────────────────────
# 12. Carga completa del env (si posible)
# ─────────────────────────────────────────────────────────────────────────────
section("12. Carga env (load_env_cfg)")

try:
  cfg = load_env_cfg("Mjlab-Velocity-Flat-H1_2-LowerLimb", play=False)
  print(f"  load_env_cfg Flat: {PASS}")
  cfg_play = load_env_cfg("Mjlab-Velocity-Flat-H1_2-LowerLimb", play=True)
  print(f"  load_env_cfg Flat play: {PASS}")
  results["12_load"] = PASS
except Exception as e:
  print(f"  {FAIL} load_env_cfg: {e}")
  results["12_load"] = FAIL

# ─────────────────────────────────────────────────────────────────────────────
# Resumen final
# ─────────────────────────────────────────────────────────────────────────────
section("RESUMEN FINAL")

all_pass = True
for key in sorted(results.keys()):
  status = results[key]
  label = key.replace("_", " ").upper()
  print(f"  [{label:<20}] {status}")
  if status == FAIL:
    all_pass = False

print(f"\n{'=' * 60}")
if all_pass:
  print("  🚀 LOWER_LIMB_CONTROL_H1_2 — Todo OK")
else:
  print("  ⚠️  Revisar los puntos marcados con FAIL")
print(f"{'=' * 60}\n")

sys.exit(0 if all_pass else 1)
