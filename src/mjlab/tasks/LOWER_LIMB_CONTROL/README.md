# balance_manipulation

Tarea de **seguimiento de velocidad** y **balance** para el humanoide **Unitree G1 con manos** (G1-Hands). El robot aprende a seguir comandos de velocidad lineal/angular mientras mantiene el equilibrio.

**Task IDs registrados:**
- `Mjlab-Velocity-Flat-G1-Hands` (terreno plano)
- `Mjlab-Velocity-Rough-G1-Hands` (terreno irregular)

---

## Estructura de la carpeta

```
balance_manipulation/
├── config/g1/
│   ├── __init__.py      # Registra las tareas (Flat y Rough)
│   ├── env_cfgs.py      # Config del entorno (escena, robot, sensores, terreno)
│   └── rl_cfg.py        # Config PPO (experiment_name, hiperparámetros)
├── mdp/
│   ├── observations.py  # Qué observa el agente (vel, joints, comandos, etc.)
│   ├── rewards.py       # Recompensas (track velocity, upright, foot clearance...)
│   ├── terminations.py  # Cuándo termina el episodio (timeout, fell_over)
│   ├── curriculums.py   # Curriculum de dificultad (terreno, velocidad)
│   └── velocity_command.py  # Comandos de velocidad (twist)
├── rl/
│   └── runner.py        # Runner específico (hereda de VelocityOnPolicyRunner)
├── velocity_env_cfg.py  # Ensambla el env completo (usa mdp + robot + terreno)
└── Verificaciones/      # Scripts para comprobar que todo funciona
    ├── verify_g1_constants.py
    ├── verify_g1_hands_env.py
    ├── g1_inspect.py
    ├── g1_with_hands.py
    └── g1_hands_inspect.py
```

---

## Qué hace cada parte

| Archivo / carpeta | Función |
|-------------------|---------|
| `config/g1/__init__.py` | Registra las tareas en el registry de mjlab. Al hacer `train Mjlab-Velocity-Flat-G1-Hands` se carga la config de aquí. |
| `config/g1/env_cfgs.py` | Define `unitree_g1_flat_env_cfg()` y `unitree_g1_rough_env_cfg()`: escena, robot G1-Hands, sensores (IMU, terrain scan, contactos), terreno (flat o rough). |
| `config/g1/rl_cfg.py` | Config de PPO: `experiment_name="g1_hands_velocity"`, seed, learning rate, etc. |
| `velocity_env_cfg.py` | Función `make_velocity_env_cfg()` que monta el MDP: observaciones, acciones, comandos, recompensas, terminaciones, curriculum, eventos. |
| `mdp/` | Implementación del MDP: funciones de reward, observaciones, comandos de velocidad, curriculum. |
| `rl/runner.py` | Runner que usa RSL-RL para entrenar; hereda de `VelocityOnPolicyRunner`. |
| `Verificaciones/` | Scripts para verificar constantes del robot, que el env carga bien y para inspeccionar el modelo visualmente. |

---

## Robot usado

La tarea usa el robot **G1 con manos** definido en:
- `src/mjlab/asset_zoo/robots/g1_hands/` (XML + cfg + meshes)
- `src/mjlab/asset_zoo/robots/robot_hands/` (constantes, URDF/XML de manos)
