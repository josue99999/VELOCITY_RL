# MjLab — Velocity RL

**Isaac Lab–style API for legged locomotion, powered by [MuJoCo-Warp](https://github.com/google-deepmind/mujoco_warp).**

Train velocity-tracking policies for humanoid robots at GPU-scale — without Isaac Sim.

---

## Demos

| G1 · Flat terrain | H1-2 · Rough terrain (curriculum) |
|:-----------------:|:---------------------------------:|
| ![G1](G1.gif)     | ![H1_2](H1_2_Curriculum.gif)      |

---

## Overview

MjLab reimplements the Isaac Lab training API on top of MuJoCo + Warp, enabling massively parallel RL directly in MuJoCo without proprietary NVIDIA simulators. Key features:

- **Parallel rollouts** via MuJoCo-Warp GPU backend
- **RSL-RL** training loop (PPO) out of the box
- **Curriculum learning** for rough-terrain locomotion (stairs, height fields)
- **Export to ONNX** for deployment
- **Live visualization** via `viser`
- **W&B + TensorBoard** logging

---

## Requirements

| Component | Version |
|-----------|---------|
| Python    | 3.10 – 3.13 |
| PyTorch   | ≥ 2.7.0 (CUDA 12.8) |
| MuJoCo    | ≥ 3.4.0 |
| CUDA      | 12.8 (Linux x86\_64) |
| Platform  | Linux x86\_64 · macOS arm64 (CPU only) |

---

## Installation

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/josue99999/VELOCITY_RL.git
cd VELOCITY_RL
uv sync --extra cu128          # GPU (Linux)
# uv sync                      # CPU (macOS)
```

> **Always use `uv run` instead of `python` directly.**

---

## Quickstart

```bash
# List available environments
uv run list_envs

# Train a velocity-tracking policy
uv run train --env UnitreeG1Velocity --num_envs 4096

# Play back a trained checkpoint
uv run play --env UnitreeG1Velocity --checkpoint logs/UnitreeG1Velocity/latest/model.pt

# Interactive demo with viser visualizer
uv run demo --env UnitreeH1_2Velocity
```

---

## Project Structure

```
src/mjlab/
├── envs/          # Task definitions (rewards, observations, terrain)
├── robots/        # MJCF assets and actuator configs
├── scripts/       # train · play · demo · list_envs entry points
├── runners/       # RSL-RL wrappers
└── utils/         # Math helpers, logging, export
scripts/           # Standalone utility scripts
notebooks/         # Analysis and visualization notebooks
docs/              # Sphinx documentation
tests/             # Unit and integration tests
```

---

## Development

```bash
make format     # ruff format + lint
make type       # ty (fast) + pyright (thorough)
make test-fast  # Exclude slow tests
make test       # Full test suite
make docs       # Build Sphinx docs
```

Run `make check` (format + type) before opening a PR.

Style: 88-column line limit, no local imports unless unavoidable, pytest fixtures over test classes.

---

## Docker

```bash
docker build -t mjlab .
docker run --gpus all mjlab uv run train --env UnitreeG1Velocity
```

---

## Citation

```bibtex
@software{mjlab2025,
  title   = {MjLab: Isaac Lab API powered by MuJoCo-Warp},
  author  = {The MjLab Developers},
  year    = {2025},
  url     = {https://github.com/josue99999/VELOCITY_RL}
}
```

---

## License

[Apache 2.0](LICENSE)
