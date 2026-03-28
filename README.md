# Motion Tracking Task

This branch (`motion_tracking`) contains a motion tracking task that teaches humanoid robots (G1 and H1_2) to replicate human dance movements captured by [video2robot](https://github.com/google-deepmind/video2robot).

## Overview

The motion tracking task uses **PPO (Proximal Policy Optimization)** with a **progressive curriculum** that gradually increases difficulty during training:

1. **Phase 1 (0-2k iterations)**: Zero noise, loose constraints → policy learns basics
2. **Phase 2 (2k-5k iterations)**: 50% noise, medium constraints → policy generalizes
3. **Phase 3 (5k-60k iterations)**: Full noise, tight constraints → policy converges

This prevents early-training collapse where untrained policies die in ~3 steps.

---

## Datasets

Two dance datasets are supported:

- **Huayno** (Peruvian dance): `abadjosue25-abba/csv_to_npz/huayno-g1:v0`, `huayno-h1-2:v0`
- **Caporal** (Mexican dance): `abadjosue25-abba/csv_to_npz/caporal-g1:v0`, `caporal-h1-2:v0`

---

## Training

### G1 (4096 envs, ~8 hours to 60k iterations)

**Huayno:**
```bash
WANDB_CACHE_DIR=/tmp/wandb_cache uv run train Mjlab-Tracking-Flat-Unitree-G1 \
  --env.scene.num-envs 4096 \
  --registry-name abadjosue25-abba/csv_to_npz/huayno-g1:v0
```

**Caporal:**
```bash
WANDB_CACHE_DIR=/tmp/wandb_cache uv run train Mjlab-Tracking-Flat-Unitree-G1 \
  --env.scene.num-envs 4096 \
  --registry-name abadjosue25-abba/csv_to_npz/caporal-g1:v0
```

### H1_2 (2048 envs, ~12 hours to 60k iterations)

**Huayno:**
```bash
WANDB_CACHE_DIR=/tmp/wandb_cache uv run train Mjlab-Tracking-Flat-H1_2 \
  --env.scene.num-envs 2048 \
  --registry-name abadjosue25-abba/csv_to_npz/huayno-h1-2:v0
```

**Caporal:**
```bash
WANDB_CACHE_DIR=/tmp/wandb_cache uv run train Mjlab-Tracking-Flat-H1_2 \
  --env.scene.num-envs 2048 \
  --registry-name abadjosue25-abba/csv_to_npz/caporal-h1-2:v0
```

### With tmux (background):

```bash
tmux new-session -s training -d 'cd /path/to/VELOCITY_RL && \
  export UV_CACHE_DIR=/tmp/uv_cache && \
  export WANDB_CACHE_DIR=/tmp/wandb_cache && \
  uv run train Mjlab-Tracking-Flat-Unitree-G1 --env.scene.num-envs 4096 --registry-name abadjosue25-abba/csv_to_npz/caporal-g1:v0'

# Attach to see logs
tmux attach -t training
# Detach: Ctrl+b d
```

---

## Evaluation

Test a trained checkpoint with visualization:

### G1 Caporal (59,999 iterations, checkpoint included)

```bash
uv run play Mjlab-Tracking-Flat-Unitree-G1 \
  --checkpoint-file src/mjlab/output/G1_MOTION_TRACKING/model_caporal_59999.pt \
  --motion-file artifacts/caporal-g1:v0/motion.npz
```

### Record Video (500 frames)

```bash
uv run play Mjlab-Tracking-Flat-Unitree-G1 \
  --checkpoint-file src/mjlab/output/G1_MOTION_TRACKING/model_caporal_59999.pt \
  --motion-file artifacts/caporal-g1:v0/motion.npz \
  --video True --video-length 500 --num-envs 1
```

Video saved to: `src/mjlab/output/G1_MOTION_TRACKING/videos/play/rl-video-step-0.mp4`

---

## Curriculum Details

The curriculum modifies 6 terms during training:

| Term | Phase 1 | Phase 2 | Phase 3 |
|------|---------|---------|---------|
| RSI Noise (pose) | 0% | 50% | 100% |
| EE Body Position Threshold | 0.5m | 0.35m | 0.25m |
| Anchor Orientation Threshold | 1.5rad | 1.2rad | 0.8rad |
| Action Rate Penalty | -0.01 | -0.05 | -0.1 |
| Self-Collision Penalty | -1.0 | -5.0 | -10.0 |
| Joint Velocity Penalty | -0.001 | -0.001 | -0.001 |

**RSI = Reference State Initialization**: initial pose/velocity perturbation when resetting.

---

## Monitoring Training

View real-time metrics on Weights & Biases:
- **Episode Length**: Should grow from ~4 to ~150+ steps
- **Reward**: Should increase from ~0 to ~+4
- **Entropy**: Should decrease from ~1.0 to ~0.1-0.3 (policy becomes confident)

---

## GIFs

See motion tracking results:
- `src/mjlab/output/H1_2_MOTION_TRACKING/giff/G1_HUAYNO.gif`
- `src/mjlab/output/H1_2_MOTION_TRACKING/giff/G1_CAPORAL.gif`

---

## Key Files

- **Curriculum logic**: `src/mjlab/tasks/tracking/mdp/curriculums.py`
- **G1 config**: `src/mjlab/tasks/tracking/config/g1/env_cfgs.py`, `rl_cfg.py`
- **H1_2 config**: `src/mjlab/tasks/tracking/config/h1_2/env_cfgs.py`, `rl_cfg.py`
- **Motion commands**: `src/mjlab/tasks/tracking/mdp/commands.py`

---

## Troubleshooting

**Episode length stuck low (~3-5 steps)?**
- Check that curriculum is active (should see phase transitions in logs)
- Verify RSI noise is 0 at step 0

**Entropy stuck at ~1.0?**
- Train with more environments: `--env.scene.num-envs 4096`
- Increase `max_iterations` in RL config

**Motion file not found?**
- Set `WANDB_CACHE_DIR=/tmp/wandb_cache` to avoid permission issues
- Or pre-download: `python -c "import wandb; api=wandb.Api(); a=api.artifact('abadjosue25-abba/csv_to_npz/caporal-g1:v0'); a.download(root='artifacts/caporal-g1:v0')"`
