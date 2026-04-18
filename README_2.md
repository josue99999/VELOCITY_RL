

## Demos

| G1 · Flat terrain | H1-2 · Rough terrain (curriculum) |
|:-----------------:|:---------------------------------:|
| ![G1](G1.gif)     | ![H1_2](H1_2_Curriculum.gif)      |

---

## 📌 What This Project Does

This project builds and trains **robust lower-limb locomotion policies** for two humanoid platforms:

- **Unitree H1_2** — 13 DOF control (leg joints + torso joint)
- **Unitree G1** — 15 DOF control (leg joints + waist joints)

The core idea: the **lower body handles all locomotion and balance**, while the upper body stays free for teleoperation. The policy must remain stable even when upper-body disturbances are applied — simulating real teleop conditions.

MjLab reimplements the Isaac Lab training API on top of **MuJoCo + Warp**, enabling massively parallel RL directly in MuJoCo without proprietary NVIDIA simulators.

**Key features:**
- Parallel rollouts via MuJoCo-Warp GPU backend
- RSL-RL training loop (PPO) out of the box
- Curriculum learning for rough-terrain locomotion (stairs, heightfields)
- Export to ONNX for deployment
- Live visualization via `viser`
- W&B + TensorBoard logging

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

## 🏗️ Environment Design

The environment is **manager-based** — observations, actions, rewards, events, terminations, commands, and curriculum live in **separate modular components**. This makes the system easy to debug and extend to new robots or tasks.

### Registered Tasks

```
Mjlab-Velocity-Flat-H1_2-LowerLimb
Mjlab-Velocity-Rough-H1_2-LowerLimb
Mjlab-Velocity-Flat-G1-LowerLimb
Mjlab-Velocity-Rough-G1-LowerLimb
```

### Observation Space

**Actor (what the deployed policy sees at runtime):**

| Signal | Purpose |
|---|---|
| Base linear velocity | Track commanded speed |
| Base angular velocity | Detect rotation/yaw drift |
| Projected gravity | Encode body tilt and balance |
| Joint positions | Current robot configuration |
| Joint velocities | Motion dynamics |
| Previous actions | Enforce smoothness and continuity |
| Velocity command | Target the robot must follow |
| Gait phase clock | Rhythmic signal for step timing |
| Terrain height scan | Local ground awareness (rough mode only) |

**Critic (privileged info — training only):**  
Foot height, foot contact state, foot air time, foot contact forces.

This is **asymmetric actor-critic training**: the critic sees richer information during training to improve value estimates, but the deployed actor uses only signals available on real hardware.

### Action Space

The policy outputs **joint position targets**. Scaling is configured per joint using robot-specific actuator constants so command amplitudes stay physically reasonable.

---

## 🎯 Reward Function

The reward is not just speed tracking. It rewards **physically clean, stable locomotion**.

| Term | What it encourages |
|---|---|
| Linear velocity tracking | Follow commanded forward/lateral speed |
| Angular velocity tracking | Follow commanded yaw rate |
| Upright posture | Penalize body tilt |
| Posture consistency | Discourage erratic trunk motion |
| Action smoothness | Reduce jerky joint commands |
| Joint limit safety | Stay away from mechanical limits |
| Foot air time | Proper gait with clear lift phases |
| Foot clearance | Swing foot clears the ground |
| Swing height | Consistent step height |
| Slip reduction | Minimize foot sliding on contact |
| Soft landing | Smooth foot contact, not stomping |
| Self-collision penalty | Prevent the robot from hitting itself |

---

## 📈 Curriculum Learning

Training difficulty increases across **three axes**:

1. **Command curriculum** — velocity ranges start small, then expand
2. **Terrain curriculum** — flat → progressively rougher heightfields
3. **Disturbance curriculum** (H1_2) — arm teleoperation disturbance intensity increases gradually

The robot first learns **clean gait structure**, then learns **robustness under disturbance**.  
Introducing disturbances too early breaks learning; too late, the policy never adapts.

---

## 🌉 Sim-to-Real Techniques

| Technique | Detail |
|---|---|
| Friction randomization | Foot-ground friction varied per episode |
| CoM offset randomization | Base center-of-mass shifted randomly |
| Encoder bias randomization | Simulated sensor noise on joint readings |
| External pushes | Random perturbation forces during training |
| Arm disturbances (H1_2) | Teleop-like upper-body forces applied to torso |
| Action delay modeling | Simulates actuator latency |

These techniques reduce **overfitting to a perfect simulation** and improve transfer to real hardware.

---

## 🧠 Policy Architecture & Training

### Neural Network

| Robot | Hidden layers | Activation |
|---|---|---|
| H1_2 | (512, 256, 128) | ELU |
| G1 | (256, 128, 64) | ELU |

Both networks use **observation normalization** at the input.

### PPO Configuration

- Clipped surrogate objective with adaptive learning rate schedule
- KL divergence target for stable updates
- Gradient clipping
- Training runs up to **60,000 iterations**
- **NaN/Inf protection**: log-std clamping, sanitized distribution inputs, skip optimizer step on invalid gradients

---

## 🔧 Top 3 Hardest Problems Solved

### Problem 1 — Contact Instability on Rough Terrain (H1_2)

The H1_2 has many foot collision geometries. On rough MuJoCo terrains this caused unstable or saturated contact behavior that broke physics and produced nonsense rollouts. I solved it by creating a robot-specific terrain config with tuned heightfield scale and simulation parameters (`ccd_iterations`, `contact_sensor_maxmatch`, `nconmax`). The fix was validated under randomized terrain across long rollouts. The result was stable rough-terrain simulation that enabled reliable large-scale PPO training.

### Problem 2 — PPO Training Collapse from NaN/Inf Gradients

In parallel GPU training, a single corrupted update could propagate NaN/Inf values through the policy distribution and destroy learning across thousands of environments. I implemented runner-level safeguards: clamping log-std/std ranges, sanitizing distribution inputs, and skipping optimizer steps when gradients were invalid. These protections were kept minimal to avoid changing PPO's core learning logic. The fix eliminated catastrophic run failures and significantly improved training continuity.

### Problem 3 — Curriculum Design for Disturbance Robustness

Training a policy robust to teleoperation-like upper-body disturbances requires careful staging — too much disturbance too early and the robot never learns to walk; too little and it never adapts. I designed a staged curriculum: first clean locomotion, then expanding command ranges and terrain difficulty, then progressively harder arm disturbances. Combined with the multi-term reward (tracking + posture + foot quality), this prevented shortcut behaviors. The result was a policy that maintains balance under significantly harder disturbance conditions than naive single-stage training.

---

## 📦 Deployment

Policies are exported to **ONNX format** via the custom velocity runners, enabling downstream integration with teleoperation frameworks and onboard inference pipelines.

---

## Project Structure

```
src/mjlab/
├── envs/          # Task definitions (rewards, observations, terrain)
├── robots/        # MJCF assets and actuator configs
├── scripts/       # train · play · demo · list_envs entry points
├── runners/       # RSL-RL wrappers + NaN protection + ONNX export
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
