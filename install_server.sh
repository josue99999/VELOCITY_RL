#!/usr/bin/env bash
# Instalación en servidor para VELOCITY_RL (balance_manipulation G1 Hands).
# Ejecutar desde la raíz del repo: bash install_server.sh
# Requisitos: conda (miniconda/anaconda) instalado, NVIDIA GPU con CUDA 12.x, git.

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

ENV_NAME="mjlab_nonhuman"
MUJOCO_WARP_REV="0828fb0b57d7baf734dd71fa164d092cb17e635b"

echo "[1/6] Comprobando conda..."
if ! command -v conda &>/dev/null; then
  echo "Error: conda no encontrado. Instala Miniconda o Anaconda primero."
  exit 1
fi

echo "[2/6] Creando entorno conda '$ENV_NAME' (python=3.11)..."
if conda env list | grep -q "^${ENV_NAME} "; then
  echo "Entorno '$ENV_NAME' ya existe. Saltando creación."
else
  conda env create -f environment.yml
fi

echo "[3/6] Instalando PyTorch con CUDA 12.8..."
conda run -n "$ENV_NAME" pip install --quiet torch --index-url https://download.pytorch.org/whl/cu128

echo "[4/6] Instalando warp-lang (NVIDIA) y mujoco..."
conda run -n "$ENV_NAME" pip install --quiet warp-lang --extra-index-url https://pypi.nvidia.com
conda run -n "$ENV_NAME" pip install --quiet "mujoco>=3.4.0" --extra-index-url https://py.mujoco.org

echo "[5/6] Instalando mujoco-warp desde git (revisión fija)..."
conda run -n "$ENV_NAME" pip install --quiet "mujoco-warp @ git+https://github.com/google-deepmind/mujoco_warp@${MUJOCO_WARP_REV}"

echo "[6/6] Instalando mjlab y resto de dependencias (editable)..."
conda run -n "$ENV_NAME" pip install --quiet -e .

echo ""
echo "=== Instalación completada ==="
echo ""
echo "Para entrenar (en servidor sin pantalla usa EGL):"
echo "  conda activate $ENV_NAME"
echo "  export MUJOCO_GL=egl"
echo "  python -m mjlab.scripts.train Mjlab-Velocity-Flat-G1-Hands --env.scene.num-envs 4096"
echo ""
echo "O terreno rough:"
echo "  python -m mjlab.scripts.train Mjlab-Velocity-Rough-G1-Hands --env.scene.num-envs 4096"
echo ""
