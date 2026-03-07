# Análisis LOWER_LIMB_CONTROL_H1_2

## Estructura del task

```
LOWER_LIMB_CONTROL_H1_2/
├── velocity_env_cfg.py      # Config base: obs, actions, rewards, events, curriculum
├── mdp/
│   ├── __init__.py          # Re-exporta envs.mdp + curriculums, events, rewards, etc.
│   ├── curriculums.py       # terrain_levels_vel, commands_vel, update_teleop_pushes, log_phase_curriculum
│   ├── events.py            # randomize_arm_pose_phase_based, randomize_arm_mass_phase_based
│   ├── rewards.py           # track_linear/angular_velocity, feet_*, etc.
│   ├── observations.py  # foot_height, foot_air_time, foot_contact
│   ├── terminations.py   # illegal_contact
│   └── velocity_command.py  # UniformVelocityCommand
├── config/
│   └── h1_2/
│       ├── __init__.py      # Registra Mjlab-Velocity-{Rough,Flat}-H1_2-LowerLimb
│       ├── env_cfgs.py      # unitree_h1_2_rough_env_cfg, unitree_h1_2_flat_env_cfg
│       └── rl_cfg.py        # unitree_h1_2_ppo_runner_cfg
└── rl/
    └── runner.py            # VelocityOnPolicyRunner (exporta ONNX)
```

## Dependencias

- **velocity_env_cfg** importa `mdp` de `LOWER_LIMB_CONTROL_H1_2` (propio de la carpeta).
- **config/h1_2/env_cfgs** importa `mdp` de `LOWER_LIMB_CONTROL_H1_2` (self_collision_cost, etc.).
- **mdp/events** en LOWER_LIMB_CONTROL_H1_2 tiene `randomize_arm_pose_phase_based` y `randomize_arm_mass_phase_based`; velocity_env_cfg usa `mdp.events.*` del mdp local.

## Constantes clave

| Constante | Valor | Descripción |
|-----------|-------|-------------|
| CONTROLLED_JOINTS | 7 patrones | 13 DOF: piernas (12) + torso (1) |
| OBSERVED_JOINTS | 15 patrones | CONTROLLED + brazos/manos (39 joints en modelo) |
| ARM_AND_HAND_JOINTS | 8 patrones | Para curriculum (arm pose/mass) |
| CURRICULUM_PHASES | 5 fases | phase_0_baseline → phase_4_teleop_robust |

## Flujo de datos

1. **Observaciones (actor)**: base_lin_vel, base_ang_vel, projected_gravity, joint_pos (OBSERVED), joint_vel (OBSERVED), actions, command, height_scan
2. **Acciones**: 13 DOF (JointPositionActionCfg con CONTROLLED_JOINTS)
3. **Comando**: twist (vx, vy, ωz) con resampling 3–8 s
4. **Rewards**: track_linear/angular_velocity, upright, pose, body_ang_vel, angular_momentum, dof_pos_limits, action_rate_l2, air_time, foot_*, soft_landing, self_collisions
5. **Terminations**: time_out (20 s), fell_over (70°)

## Eventos (orden)

- **reset**: reset_base, reset_robot_joints (CONTROLLED_JOINTS), randomize_arm_pose, randomize_arm_mass
- **interval**: push_robot (parámetros por curriculum)
- **startup**: foot_friction, encoder_bias, base_com (torso_link)

## Config H1_2 específica (env_cfgs.py)

- Robot: `get_h1_2_robot_cfg()`
- Sites: left_foot, right_foot
- Geoms fricción: left_foot1_collision … left_foot7_collision, right_foot1 … right_foot7
- Contact sensors: feet_ground_contact (ankle_roll_link vs terrain), self_collision (pelvis vs pelvis)
- Action scale: filtrado de H1_2_ACTION_SCALE por CONTROLLED_JOINTS
- Pose reward: std_standing/walking/running por joint (hip, knee, ankle, torso)
- Body: torso_link para upright, body_ang_vel

## Verificación

Ejecutar: `uv run python src/mjlab/tasks/LOWER_LIMB_CONTROL_H1_2/Verificaciones/verify_lower_limb_h1_2_task.py`

Comprueba: imports, joints, modelo, action scale, env config, observaciones, rewards, events, curriculum, task registration, rl_cfg, load_env_cfg.
