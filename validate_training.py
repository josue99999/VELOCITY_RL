#!/usr/bin/env python3
"""Validación completa antes de lanzar entrenamiento (LOWER_LIMB_CONTROL_H1_2)."""

import sys
import torch


def test_imports():
  """Test 1: Imports."""
  try:
    from mjlab.tasks.LOWER_LIMB_CONTROL_H1_2 import mdp
    from mjlab.tasks.LOWER_LIMB_CONTROL_H1_2.velocity_env_cfg import (
      make_velocity_env_cfg,
    )
    from mjlab.tasks.LOWER_LIMB_CONTROL_H1_2.config.h1_2.env_cfgs import (
      unitree_h1_2_flat_env_cfg,
    )

    print("✅ Test 1: Imports OK")
    return True
  except Exception as e:
    print(f"❌ Test 1 FAILED: {e}")
    return False


def test_config():
  """Test 2: Configuración."""
  try:
    from mjlab.tasks.LOWER_LIMB_CONTROL_H1_2.config.h1_2.env_cfgs import (
      unitree_h1_2_flat_env_cfg,
    )

    cfg = unitree_h1_2_flat_env_cfg()

    required_curriculum = [
      "gatekeeping",
      "teleop_disturbances",
      "arm_teleop_continuous",
      "phase_info",
    ]
    for term in required_curriculum:
      assert term in cfg.curriculum, f"Missing curriculum: {term}"

    required_events = [
      "randomize_arm_pose",
      "randomize_arm_mass",
      "arm_pose_continuous_teleop",
    ]
    for event in required_events:
      assert event in cfg.events, f"Missing event: {event}"

    print("✅ Test 2: Config OK")
    return True
  except Exception as e:
    print(f"❌ Test 2 FAILED: {e}")
    return False


def test_env_creation():
  """Test 3: Crear entorno y hacer steps."""
  try:
    from mjlab.tasks.LOWER_LIMB_CONTROL_H1_2.config.h1_2.env_cfgs import (
      unitree_h1_2_flat_env_cfg,
    )
    from mjlab.envs import ManagerBasedRlEnv

    cfg = unitree_h1_2_flat_env_cfg()
    cfg.scene.num_envs = 2
    cfg.episode_length_s = 4.0

    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")

    # Reset
    obs, _ = env.reset()
    batch = list(obs.values())[0].shape[0] if isinstance(obs, dict) else obs.shape[0]
    assert batch == 2, "Wrong obs batch size"

    # Steps
    actions = torch.zeros(
      env.num_envs, env.action_manager.total_action_dim, device=env.device
    )
    for _ in range(20):
      obs, reward, terminated, truncated, info = env.step(actions)
      assert not torch.isnan(reward).any(), "NaN in rewards"
      assert not torch.isinf(reward).any(), "Inf in rewards"

    env.close()
    print("✅ Test 3: Environment OK")
    return True
  except Exception as e:
    print(f"❌ Test 3 FAILED: {e}")
    import traceback

    traceback.print_exc()
    return False


def test_curriculum():
  """Test 4: Curriculum y gatekeeping."""
  try:
    from mjlab.tasks.LOWER_LIMB_CONTROL_H1_2.config.h1_2.env_cfgs import (
      unitree_h1_2_flat_env_cfg,
    )
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.LOWER_LIMB_CONTROL_H1_2.mdp.curriculums import (
      gatekeeping_phase_control,
      update_teleop_pushes,
    )

    cfg = unitree_h1_2_flat_env_cfg()
    cfg.scene.num_envs = 4
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")

    # Test fase con perturbaciones (light_disturbances: episode_range (750, 2000))
    # After gatekeeping, offset can be 10 → need effective_episodes >= 750, so steps >= (750+10)*1000
    env.common_step_counter = 760_000

    # Test gatekeeping con métricas malas
    env._curriculum_metrics = {
      "mean_reward": 20.0,  # Muy bajo
      "value_loss": 500.0,
      "fell_over_rate": 0.3,
    }

    result = gatekeeping_phase_control(
      env,
      torch.arange(env.num_envs),
      cfg.curriculum["gatekeeping"].params["phases"],
    )

    assert result["phase_frozen"].item() == 1.0, "Gatekeeping should freeze"
    assert getattr(env, "_episode_offset", 0) > 0, "Offset should increase"

    # Test pushes en fase 4
    push_result = update_teleop_pushes(
      env,
      torch.arange(env.num_envs),
      cfg.curriculum["teleop_disturbances"].params["phases"],
    )
    assert push_result["push_velocity"].item() > 0, "Pushes should be active"

    env.close()
    print("✅ Test 4: Curriculum OK")
    return True
  except Exception as e:
    print(f"❌ Test 4 FAILED: {e}")
    import traceback

    traceback.print_exc()
    return False


def main():
  """Run all tests."""
  print("=" * 50)
  print("VALIDACIÓN PRE-ENTRENAMIENTO (LOWER_LIMB_CONTROL_H1_2)")
  print("=" * 50)

  tests = [test_imports, test_config, test_env_creation, test_curriculum]
  results = []

  for test in tests:
    results.append(test())
    print()

  print("=" * 50)
  if all(results):
    print("🎉 TODOS LOS TESTS PASARON - Listo para entrenar!")
    print("Comando sugerido:")
    print(
      "  python -m mjlab.train --task Mjlab-Velocity-Rough-H1_2-LowerLimb --headless"
    )
    return 0
  else:
    print("❌ ALGUNOS TESTS FALLARON - Corrige antes de entrenar")
    return 1


if __name__ == "__main__":
  sys.exit(main())
