# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Architecture Overview

mjlab is a GPU-accelerated robot RL framework that combines [Isaac Lab](https://github.com/isaac-sim/IsaacLab)'s manager-based API with [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp). It uses PPO via `rsl-rl-lib` for training.

## Manager-Based Environment

The core abstraction is `ManagerBasedRlEnvCfg` (`src/mjlab/envs/manager_based_rl_env.py`). An environment is fully described by composing **manager term configs**:

| Manager | Config class | Purpose |
|---|---|---|
| `ObservationManager` | `ObservationTermCfg` | Sensor readings fed to the policy |
| `ActionManager` | `ActionTermCfg` | Maps policy outputs to actuator commands |
| `RewardManager` | `RewardTermCfg` | Weighted reward terms (scaled by `dt` by default) |
| `EventManager` | `EventTermCfg` | Domain randomization, resets |
| `TerminationManager` | `TerminationTermCfg` | Episode termination conditions |
| `CommandManager` | `CommandTermCfg` | Goal/velocity commands injected into observations |
| `CurriculumManager` | `CurriculumTermCfg` | Curriculum scheduling |

Each manager term config has a `func` callable and optional `params`, `weight`, `noise`, etc.

## Task Registration

Tasks are registered with `register_mjlab_task()` (`src/mjlab/tasks/registry.py`). Each task needs:
- `env_cfg`: training environment config (`ManagerBasedRlEnvCfg`)
- `play_env_cfg`: evaluation config (usually same factory with `play=True`)
- `rl_cfg`: PPO runner config (`RslRlOnPolicyRunnerCfg`)
- `runner_cls`: optional custom runner subclassing `MjlabOnPolicyRunner`

Registration happens in each task's `config/<robot>/__init__.py`, which is imported at package init time.

## Task Structure

Each task lives under `src/mjlab/tasks/<TASK_NAME>/` and follows this layout:

```
<TASK_NAME>/
  velocity_env_cfg.py      # Factory: make_velocity_env_cfg() → ManagerBasedRlEnvCfg
  mdp/                     # Task-specific MDP terms
    observations.py
    rewards.py
    actions.py
    events.py
    terminations.py
    curriculums.py
    velocity_command.py
  config/
    <robot>/
      env_cfgs.py          # Calls make_*_env_cfg(), adds robot-specific overrides
      rl_cfg.py            # Returns RslRlOnPolicyRunnerCfg
      __init__.py          # Calls register_mjlab_task(...)
  rl/
    runner.py              # Optional custom MjlabOnPolicyRunner subclass
```

The `velocity` task is the canonical reference implementation. The `LOWER_LIMB_CONTROL_H1_2` task is the custom lower-limb-only variant for the Unitree H1 v2 robot (`Mjlab-Velocity-Rough-H1_2-LowerLimb`, `Mjlab-Velocity-Flat-H1_2-LowerLimb`).

## Running Tasks

```sh
# Train
uv run train Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs 4096

# Evaluate (fetches latest checkpoint from W&B)
uv run play Mjlab-Velocity-Flat-Unitree-G1 --wandb-run-path org/mjlab/run-id

# Sanity-check with dummy agents (no training needed)
uv run play Mjlab-Velocity-Flat-Unitree-G1 --agent zero
uv run play Mjlab-Velocity-Flat-Unitree-G1 --agent random

# List all registered task IDs
uv run list_envs
```

## Code Style

- **Indentation: 2 spaces** (configured in `ruff` — not 4).
- `src/mjlab/utils/lab_api/` is forked from Isaac Lab (BSD-3-Clause) and excluded from ruff/pyright.

---

# Development Workflow

**Always use `uv run`, not python**.

```sh

# 1. Make changes.

# 2. Type check.
uv run ty check  # Fast
uv run pyright  # More thorough, but slower

# 3. Run tests.
uv run pytest tests/  # Single suite
uv run pytest tests/<test_file>.py  # Specific file

# 4. Format and lint before committing.
uv run ruff format
uv run ruff check --fix
```

We've bundled common commands into a Makefile for convenience.

```sh
make format     # Format and lint
make type       # Type-check
make check      # make format && make type
make test-fast  # Run tests excluding slow ones
make test       # Run the full test suite
make docs       # Build documentation
```

Before creating a PR, ensure all checks pass with `make test`.

Some style guidelines to follow:
- Line length limit is 88 columns. This applies to code, comments, and docstrings.
- Avoid local imports unless they are strictly necessary (e.g. circular imports).
- Tests should follow these principles:
  - Use functions and fixtures; do not use test classes.
  - Favor targeted, efficient tests over exhaustive edge-case coverage.
  - Prefer running individual tests rather than the full test suite to improve iteration speed.
