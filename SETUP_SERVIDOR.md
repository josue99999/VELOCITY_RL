# Setup en servidor – VELOCITY_RL (balance_manipulation G1 Hands)

Instrucciones para clonar el repo en un servidor Linux con GPU, instalar el mismo entorno que en local (conda `mjlab_nonhuman`) y ejecutar el entrenamiento.

## Requisitos del servidor

- **SO**: Linux (x86_64)
- **Python**: 3.10, 3.11 o 3.12 (el script usa 3.11)
- **GPU**: NVIDIA con **CUDA 12.4+** (recomendado 12.8 para `cu128`)
- **Conda**: Miniconda o Anaconda instalado
- **Git**: para clonar el repositorio

Comprobar CUDA (opcional):

```bash
nvidia-smi
```

## Instalación en un solo comando (recomendado)

Desde la raíz del repositorio:

```bash
git clone https://github.com/josue99999/VELOCITY_RL.git
cd VELOCITY_RL
bash install_server.sh
```

El script:

1. Crea el entorno conda `mjlab_nonhuman` (Python 3.11) si no existe.
2. Instala PyTorch con CUDA 12.8.
3. Instala `warp-lang` (índice NVIDIA) y `mujoco`.
4. Instala `mujoco-warp` desde el repo de GitHub (revisión fija del proyecto).
5. Instala este proyecto en modo editable y el resto de dependencias.

Al finalizar, imprime los comandos para activar el entorno y lanzar el entrenamiento.

## Instalación manual (paso a paso)

Si prefieres no usar el script o necesitas adaptar algo:

```bash
# 1. Clonar
git clone https://github.com/josue99999/VELOCITY_RL.git
cd VELOCITY_RL

# 2. Crear entorno conda
conda env create -f environment.yml
conda activate mjlab_nonhuman

# 3. PyTorch con CUDA 12.8 (ajusta cu118/cu121/cu128 según tu CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 4. warp-lang (NVIDIA)
pip install warp-lang --extra-index-url https://pypi.nvidia.com

# 5. MuJoCo (índice oficial)
pip install "mujoco>=3.4.0" --extra-index-url https://py.mujoco.org

# 6. mujoco-warp desde git (revisión usada en el proyecto)
pip install "mujoco-warp @ git+https://github.com/google-deepmind/mujoco_warp@0828fb0b57d7baf734dd71fa164d092cb17e635b"

# 7. Proyecto y dependencias restantes
pip install -e .
```

## Entrenar después de instalar

En servidor sin pantalla es necesario usar EGL:

```bash
conda activate mjlab_nonhuman
export MUJOCO_GL=egl
```

**Terreno plano (flat):**

```bash
python -m mjlab.scripts.train Mjlab-Velocity-Flat-G1-Hands --env.scene.num-envs 4096
```

**Terreno irregular (rough):**

```bash
python -m mjlab.scripts.train Mjlab-Velocity-Rough-G1-Hands --env.scene.num-envs 4096
```

Puedes ajustar `--env.scene.num-envs` según la RAM y la GPU del servidor (por ejemplo 2048 o 8192).

**Varios GPUs:**

```bash
python -m mjlab.scripts.train Mjlab-Velocity-Flat-G1-Hands --env.scene.num-envs 4096 --gpu-ids 0 1
```

**Ejecutar en segundo plano (nohup):**

```bash
nohup python -m mjlab.scripts.train Mjlab-Velocity-Flat-G1-Hands --env.scene.num-envs 4096 > train.log 2>&1 &
```

Los logs y checkpoints se guardan en `logs/rsl_rl/g1_hands_velocity/`.

## Resumen de lo que instala (mapeo con tu entorno local)

- **Entorno conda**: `mjlab_nonhuman`, Python 3.11 (como en local).
- **PyTorch**: con CUDA 12.8 (mismo índice que en `pyproject.toml`).
- **warp-lang**: desde PyPI NVIDIA (requerido por mujoco-warp).
- **mujoco**: desde `py.mujoco.org` (versión ≥3.4.0).
- **mujoco-warp**: desde el repo de Google DeepMind, revisión fija `0828fb0b57d7baf734dd71fa164d092cb17e635b` (la del `pyproject.toml`).
- **mjlab**: instalación editable (`pip install -e .`) con todas las dependencias del `pyproject.toml` (tyro, tensorboard, wandb, rsl-rl-lib, etc.).

Con esto el servidor queda alineado con lo que usas en local para solo tener que ejecutar el comando de entrenamiento anterior.
