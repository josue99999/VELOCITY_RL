import os
import statistics
import time

import torch
import wandb

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner


class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def _set_curriculum_metrics_after_update(self, loss_dict: dict) -> None:
    """Set env._curriculum_metrics so gatekeeping_phase_control can use them."""
    env = getattr(self.env, "unwrapped", self.env)
    mean_reward = (
      statistics.mean(self.logger.rewbuffer)
      if getattr(self.logger, "rewbuffer", None) and len(self.logger.rewbuffer) > 0
      else None
    )
    if mean_reward is not None and (
      mean_reward != mean_reward or mean_reward == float("inf")
    ):
      mean_reward = None
    value_loss = loss_dict.get("value")
    if value_loss is not None:
      if isinstance(value_loss, torch.Tensor):
        if torch.isnan(value_loss).any() or torch.isinf(value_loss).any():
          value_loss = float("inf")
      elif value_loss != value_loss or value_loss == float("inf"):
        value_loss = float("inf")
    # fell_over_rate: env can log it in extras["log"]; otherwise leave None
    fell_over_rate = None
    if getattr(self.logger, "ep_extras", None) and len(self.logger.ep_extras) > 0:
      for ep in self.logger.ep_extras:
        if isinstance(ep, dict) and "Termination/fell_over" in ep:
          # Sum over batch and compute rate if available
          v = ep["Termination/fell_over"]
          if isinstance(v, torch.Tensor):
            fell_over_rate = float(v.mean().item())
          else:
            fell_over_rate = float(v)
          break
    setattr(
      env,
      "_curriculum_metrics",
      {
        "mean_reward": mean_reward,
        "value_loss": value_loss,
        "fell_over_rate": fell_over_rate,
      },
    )

  def learn(
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False
  ) -> None:
    if init_at_random_ep_len:
      self.env.episode_length_buf = torch.randint_like(
        self.env.episode_length_buf, high=int(self.env.max_episode_length)
      )

    obs = self.env.get_observations().to(self.device)
    self.alg.train_mode()

    if self.is_distributed:
      print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
      self.alg.broadcast_parameters()

    self.logger.init_logging_writer()

    start_it = self.current_learning_iteration
    total_it = start_it + num_learning_iterations
    for it in range(start_it, total_it):
      start = time.time()
      with torch.inference_mode():
        for _ in range(self.cfg["num_steps_per_env"]):
          actions = self.alg.act(obs)
          obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
          obs, rewards, dones = (
            obs.to(self.device),
            rewards.to(self.device),
            dones.to(self.device),
          )
          self.alg.process_env_step(obs, rewards, dones, extras)
          intrinsic_rewards = (
            self.alg.intrinsic_rewards
            if self.cfg["algorithm"].get("rnd_cfg") and self.alg.rnd
            else None
          )
          self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

        stop = time.time()
        collect_time = stop - start
        start = stop

        self.alg.compute_returns(obs)

      loss_dict = self.alg.update()
      has_nan = False
      for key, value in loss_dict.items():
        if isinstance(value, torch.Tensor):
          if torch.isnan(value).any() or torch.isinf(value).any():
            print(f"[WARNING] NaN/Inf in {key}: {value}")
            has_nan = True
        elif isinstance(value, (int, float)):
          if value != value or value == float("inf") or value == float("-inf"):
            print(f"[WARNING] NaN/Inf in {key}: {value}")
            has_nan = True
      if has_nan:
        raise RuntimeError(
          f"[CRITICAL] NaN/Inf detected at iteration {it}. "
          "Training stopped to prevent corruption. Restart from last checkpoint."
        )
      self._set_curriculum_metrics_after_update(loss_dict)

      stop = time.time()
      learn_time = stop - start
      self.current_learning_iteration = it

      self.logger.log(
        it=it,
        start_it=start_it,
        total_it=total_it,
        collect_time=collect_time,
        learn_time=learn_time,
        loss_dict=loss_dict,
        learning_rate=self.alg.learning_rate,
        action_std=self.alg.get_policy().output_std,
        rnd_weight=(
          self.alg.rnd.weight
          if self.cfg["algorithm"].get("rnd_cfg") and self.alg.rnd
          else None
        ),
      )

      if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
        self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))

    if self.logger.writer is not None:
      self.save(
        os.path.join(
          self.logger.log_dir,
          f"model_{self.current_learning_iteration}.pt",
        )
      )
      self.logger.stop_logging_writer()

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_path = path.split("model")[0]
    filename = os.path.basename(os.path.dirname(policy_path)) + ".onnx"
    self.export_policy_to_onnx(policy_path, filename)
    run_name: str = (
      wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
    )  # type: ignore[assignment]
    onnx_path = os.path.join(policy_path, filename)
    metadata = get_base_metadata(self.env.unwrapped, run_name)
    attach_metadata_to_onnx(onnx_path, metadata)
    if self.logger.logger_type in ["wandb"]:
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
