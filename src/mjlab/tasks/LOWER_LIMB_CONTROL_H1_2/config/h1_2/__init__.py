from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.LOWER_LIMB_CONTROL_H1_2.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_h1_2_flat_env_cfg,
  unitree_h1_2_rough_env_cfg,
)
from .rl_cfg import unitree_h1_2_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-H1_2-LowerLimb",
  env_cfg=unitree_h1_2_rough_env_cfg(),
  play_env_cfg=unitree_h1_2_rough_env_cfg(play=True),
  rl_cfg=unitree_h1_2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-H1_2-LowerLimb",
  env_cfg=unitree_h1_2_flat_env_cfg(),
  play_env_cfg=unitree_h1_2_flat_env_cfg(play=True),
  rl_cfg=unitree_h1_2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
