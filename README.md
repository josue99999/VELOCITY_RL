# VELOCITY_RL – Balance Manipulation G1 Hands

Este directorio contiene tu tarea personalizada de **balance / seguimiento de velocidad** para el humanoide **Unitree G1 con manos** (`G1-Hands`) usando MuJoCo-Warp y RSL-RL.

Está pensado como recordatorio para ti mismo de TODO lo que tocaste y cómo volver a entrenar rápido tanto en tu laptop como en el servidor.

---



- **Robot G1 con manos:**
  - `src/mjlab/asset_zoo/robots/g1_hands/`
    - `g1_hands.xml`: modelo MuJoCo del G1 con manos.
    - `g1_hands_cfg.py`: configuración del robot (entidad, joints, sensores, etc.).
    - `meshes/*.STL`: geometría del cuerpo y manos (izquierda y derecha).
  - `src/mjlab/asset_zoo/robots/robot_hands/`
    - `g1_with_hands_constants.py`: constantes del robot/manos (DOF, nombres de joints, etc.).
    - `inspire_hand_*.urdf`, `inspire_hand_left/right.xml`: definición detallada de las manos.
    - `meshes/*.STL` y `xmls/*`: versiones alternativas de XMLs/meshes para inspección.

- **Tarea de balance / velocity para G1 Hands:**
  - `src/mjlab/tasks/balance_manipulation/`
    - `config/g1/__init__.py`  
      Registra las tareas:
      - `Mjlab-Velocity-Flat-G1-Hands`
      - `Mjlab-Velocity-Rough-G1-Hands`
    - `config/g1/env_cfgs.py`  
      Configuración del entorno (escena, robot, sensores, terrenos, episodios, etc.).
    - `config/g1/rl_cfg.py`  
      Configuración de PPO/RSL-RL (`experiment_name="g1_hands_velocity"`).
    - `velocity_env_cfg.py`  
      Ensambla el entorno de velocidad usando el MDP de `balance_manipulation`.
    - `mdp/observations.py`, `mdp/rewards.py`, `mdp/terminations.py`, `mdp/curriculums.py`, `mdp/velocity_command.py`  
      Definen observaciones, recompensas, condiciones de terminación, currículum y comandos de velocidad.
    - `rl/runner.py`  
      Runner específico para esta tarea (basado en `VelocityOnPolicyRunner`).
    - `Verificaciones/`  
      Scripts para comprobar que todo está bien:
      - `g1_inspect.py`, `g1_with_hands.py`, `g1_hands_inspect.py`
      - `verify_g1_constants.py`, `verify_g1_hands_env.py`

---

## 2. Task IDs que debes usar

Tareas registradas en `config/g1/__init__.py`:

- `Mjlab-Velocity-Flat-G1-Hands`  → terreno plano.
- `Mjlab-Velocity-Rough-G1-Hands` → terreno irregular.

Son los `task_id` que debes pasar al script `train`.

---

## 3. Cómo entrenar en tu laptop (`~/mjlab`)

Asumiendo que estás en el repo original `~/mjlab` y ya tienes el entorno `mjlab_nonhuman` configurado:

```bash
cd ~/mjlab
conda activate mjlab_nonhuman
export MUJOCO_GL=egl  # si usas GPU sin pantalla

# Terreno plano:
uv run train Mjlab-Velocity-Flat-G1-Hands --env.scene.num-envs 4096

# Terreno irregular:
# uv run train Mjlab-Velocity-Rough-G1-Hands --env.scene.num-envs 4096
```

Puedes ajustar `--env.scene.num-envs` según la memoria de tu GPU.

---

## 4. Cómo entrenar en el servidor (`VELOCITY_RL`)

En el servidor clonaste este mismo código como `VELOCITY_RL`.  
Normalmente estás trabajando en `/tmp/VELOCITY_RL` para evitar problemas de permisos.

### 4.1. Instalación rápida (solo primera vez)

Desde la raíz del repo en el servidor:

```bash
cd /tmp/VELOCITY_RL
bash install_server.sh
```

Esto crea/usa el entorno `mjlab_nonhuman` y deja todo instalado.

### 4.2. Entrenar

Cada vez que quieras entrenar:

```bash
cd /tmp/VELOCITY_RL
conda activate mjlab_nonhuman
export MUJOCO_GL=egl

python -m mjlab.scripts.train Mjlab-Velocity-Flat-G1-Hands --env.scene.num-envs 4096
```

Terreno irregular:

```bash
python -m mjlab.scripts.train Mjlab-Velocity-Rough-G1-Hands --env.scene.num-envs 4096
```

### 4.3. Dejar entrenando aunque cierres SSH

Con `nohup`:

```bash
cd /tmp/VELOCITY_RL
conda activate mjlab_nonhuman
export MUJOCO_GL=egl

nohup python -m mjlab.scripts.train \
  Mjlab-Velocity-Flat-G1-Hands \
  --env.scene.num-envs 4096 > train.log 2>&1 &
```

Ver progreso:

```bash
tail -f train.log
```

Con `tmux`:

```bash
tmux new -s g1_hands_train
conda activate mjlab_nonhuman
export MUJOCO_GL=egl
cd /tmp/VELOCITY_RL
python -m mjlab.scripts.train Mjlab-Velocity-Flat-G1-Hands --env.scene.num-envs 4096
# Detach: Ctrl+B, luego D
# Re-attach: tmux attach -t g1_hands_train
```

---

## 5. Reanudar entrenamientos (resume)

Los logs y checkpoints se guardan en:

- `logs/rsl_rl/g1_hands_velocity/AAAA-MM-DD_HH-MM-SS_*`

### 5.1. Ver las runs disponibles

```bash
ls logs/rsl_rl/g1_hands_velocity/
```

### 5.2. Reanudar desde la última run (automático)

Laptop (`uv`):

```bash
cd ~/mjlab
conda activate mjlab_nonhuman
export MUJOCO_GL=egl

uv run train Mjlab-Velocity-Flat-G1-Hands \
  --env.scene.num-envs 4096 \
  --agent.resume True
```

Servidor:

```bash
cd /tmp/VELOCITY_RL
conda activate mjlab_nonhuman
export MUJOCO_GL=egl

python -m mjlab.scripts.train \
  Mjlab-Velocity-Flat-G1-Hands \
  --env.scene.num-envs 4096 \
  --agent.resume True
```

### 5.3. Reanudar una run concreta

```bash
ls logs/rsl_rl/g1_hands_velocity/
# Ejemplo: 2026-03-01_15-18-27

python -m mjlab.scripts.train \
  Mjlab-Velocity-Flat-G1-Hands \
  --env.scene.num-envs 4096 \
  --agent.resume True \
  --agent.load-run "2026-03-01_15-18-27"
```

---

## 6. Scripts de verificación

Desde la raíz del repo:

```bash
cd src/mjlab/tasks/balance_manipulation/Verificaciones
conda activate mjlab_nonhuman
```

- Verificar constantes del robot G1 + manos:

```bash
python verify_g1_constants.py
```

- Verificar entorno `balance_manipulation` con G1 Hands:

```bash
python verify_g1_hands_env.py
```

- Inspeccionar visualmente el modelo:

```bash
python g1_with_hands.py
python g1_hands_inspect.py
python g1_inspect.py
```

---

## 7. WandB (Weights & Biases)

### 7.1. Activar logging

Una vez por máquina (laptop y servidor):

```bash
conda activate mjlab_nonhuman
wandb login
```

O usando variable de entorno:

```bash
export WANDB_API_KEY=TU_API_KEY
```

### 7.2. Desactivar WandB (por ejemplo en el servidor)

```bash
export WANDB_MODE=disabled
```

---


