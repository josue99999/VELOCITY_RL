import json
import os
import types
from pathlib import Path

import torch
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner
from torch.distributions import Normal

from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper

ACTOR_MIN_STD = 1e-6
# When std is NaN we replace with this so the policy keeps some exploration (avoids collapse).
ACTOR_NAN_REPLACEMENT_STD = 0.1
# Clamp log_std so exp(log_std) does not underflow/overflow or become NaN.
ACTOR_LOG_STD_MIN = -20.0
ACTOR_LOG_STD_MAX = 2.0
# Debug log path (session 6bf6c6).
_DEBUG_LOG_PATH = str(
  Path(__file__).resolve().parents[3] / ".cursor" / "debug-6bf6c6.log"
)


def _debug_log(data: dict) -> None:
  import sys
  import time

  payload = {**data, "timestamp": data.get("timestamp", time.time())}
  line = json.dumps(payload) + "\n"
  try:
    dirpath = os.path.dirname(_DEBUG_LOG_PATH)
    if dirpath:
      os.makedirs(dirpath, exist_ok=True)
    with open(_DEBUG_LOG_PATH, "a") as f:
      f.write(line)
  except Exception:
    pass
  # Always echo to stderr so we get evidence even if file path is wrong.
  if payload.get("message") in (
    "Patch actor min std",
    "Patched _update_distribution invoked",
    "Std problematic before clamp",
  ):
    print(
      f"[DEBUG] {payload.get('message')} | {payload.get('data', {})}",
      file=sys.stderr,
      flush=True,
    )


def _actor_update_distribution_min_std(self, obs: torch.Tensor) -> None:
  """Drop-in for MLPModel._update_distribution that clamps std to avoid PPO crash."""
  # #region agent log
  if not getattr(self, "_debug_patch_invoked", False):
    self._debug_patch_invoked = True
    _debug_log(
      {
        "sessionId": "6bf6c6",
        "hypothesisId": "H1",
        "location": "runner.py:_actor_update_distribution_min_std",
        "message": "Patched _update_distribution invoked",
        "data": {"first_call": True},
      }
    )
  # #endregion
  if self.state_dependent_std:
    mean_and_std = self.mlp(obs)
    if self.noise_std_type == "scalar":
      mean, std = torch.unbind(mean_and_std, dim=-2)
    elif self.noise_std_type == "log":
      mean, log_std = torch.unbind(mean_and_std, dim=-2)
      log_std = log_std.clamp(ACTOR_LOG_STD_MIN, ACTOR_LOG_STD_MAX)
      log_std = torch.nan_to_num(
        log_std, nan=-2.3, posinf=ACTOR_LOG_STD_MAX, neginf=ACTOR_LOG_STD_MIN
      )
      std = torch.exp(log_std)
    else:
      raise ValueError(
        f"Unknown standard deviation type: {self.noise_std_type}. "
        "Should be 'scalar' or 'log'"
      )
  else:
    mean = self.mlp(obs)
    if self.noise_std_type == "scalar":
      std = self.std.expand_as(mean)
    elif self.noise_std_type == "log":
      log_std = self.log_std.clamp(ACTOR_LOG_STD_MIN, ACTOR_LOG_STD_MAX)
      log_std = torch.nan_to_num(
        log_std, nan=-2.3, posinf=ACTOR_LOG_STD_MAX, neginf=ACTOR_LOG_STD_MIN
      )
      std = torch.exp(log_std).expand_as(mean)
    else:
      raise ValueError(
        f"Unknown standard deviation type: {self.noise_std_type}. "
        "Should be 'scalar' or 'log'"
      )
  # #region agent log
  min_std_before = std.min().item()
  any_neg = (std <= 0).any().item()
  any_nan = torch.isnan(std).any().item()
  if any_neg or any_nan:
    _debug_log(
      {
        "sessionId": "6bf6c6",
        "hypothesisId": "H1",
        "location": "runner.py:std_before_clamp",
        "message": "Std problematic before clamp",
        "data": {
          "min_std_before": min_std_before,
          "any_neg": any_neg,
          "any_nan": any_nan,
        },
      }
    )
  # #endregion
  std = std.clamp(min=ACTOR_MIN_STD)
  # NaN/Inf break Normal(); clamp does not replace them. Use ACTOR_NAN_REPLACEMENT_STD so
  # the policy keeps exploration instead of collapsing to near-deterministic (std=1e-6).
  std = torch.nan_to_num(
    std,
    nan=ACTOR_NAN_REPLACEMENT_STD,
    posinf=1.0,
    neginf=ACTOR_MIN_STD,
  )
  # #region agent log
  if any_neg or any_nan:
    _debug_log(
      {
        "sessionId": "6bf6c6",
        "hypothesisId": "H1",
        "location": "runner.py:std_after_clamp",
        "message": "Std after clamp",
        "data": {"min_std_after": std.min().item()},
      }
    )
  # #endregion
  # Sanitize mean so Normal(mean, std) never receives NaN (e.g. after a corrupt update).
  mean = torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
  self.distribution = Normal(mean, std)


class MjlabOnPolicyRunner(OnPolicyRunner):
  """Base runner that persists environment state across checkpoints."""

  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    # Strip None-valued cnn_cfg so MLPModel doesn't receive it.
    for key in ("actor", "critic"):
      if key in train_cfg and train_cfg[key].get("cnn_cfg") is None:
        train_cfg[key].pop("cnn_cfg", None)
    super().__init__(env, train_cfg, log_dir, device)
    self._patch_actor_min_std()
    self._patch_optimizer_safe_step()

  def _patch_optimizer_safe_step(self) -> None:
    """Skip optimizer.step() when any gradient is NaN/Inf to avoid corrupting weights."""
    optimizer = getattr(self.alg, "optimizer", None)
    if optimizer is None:
      return
    original_step = optimizer.step

    def safe_step() -> None:
      for group in optimizer.param_groups:
        for p in group["params"]:
          if p.grad is not None and (
            torch.isnan(p.grad).any().item() or torch.isinf(p.grad).any().item()
          ):
            optimizer.zero_grad()
            return
      original_step()

    optimizer.step = safe_step

  def _patch_actor_min_std(self) -> None:
    """Patch actor so policy std is clamped to ACTOR_MIN_STD, avoiding PPO crash."""
    # #region agent log
    has_alg = hasattr(self, "alg")
    has_actor = hasattr(self.alg, "actor") if has_alg else False
    actor = getattr(self.alg, "actor", None) if has_alg else None
    has_update_dist = (
      hasattr(actor, "_update_distribution") if actor is not None else False
    )
    # #endregion
    if has_alg and has_actor:
      actor = self.alg.actor
      if hasattr(actor, "_update_distribution"):
        actor._update_distribution = types.MethodType(
          _actor_update_distribution_min_std, actor
        )
    # #region agent log
    _debug_log(
      {
        "sessionId": "6bf6c6",
        "hypothesisId": "H2",
        "location": "runner.py:_patch_actor_min_std",
        "message": "Patch actor min std",
        "data": {
          "has_alg": has_alg,
          "has_actor": has_actor,
          "has_update_distribution": has_update_dist,
          "patch_applied": has_alg and has_actor and has_update_dist,
        },
      }
    )
    # #endregion

  def export_policy_to_onnx(
    self, path: str, filename: str = "policy.onnx", verbose: bool = False
  ) -> None:
    """Export policy to ONNX format using legacy export path.

    Overrides the base implementation to set dynamo=False, avoiding warnings about
    dynamic_axes being deprecated with the new TorchDynamo export path
    (torch>=2.9 default).
    """
    onnx_model = self.alg.get_policy().as_onnx(verbose=verbose)
    onnx_model.to("cpu")
    onnx_model.eval()
    os.makedirs(path, exist_ok=True)
    torch.onnx.export(
      onnx_model,
      onnx_model.get_dummy_inputs(),  # type: ignore[operator]
      os.path.join(path, filename),
      export_params=True,
      opset_version=18,
      verbose=verbose,
      input_names=onnx_model.input_names,  # type: ignore[arg-type]
      output_names=onnx_model.output_names,  # type: ignore[arg-type]
      dynamic_axes={},
      dynamo=False,
    )

  def save(self, path: str, infos=None) -> None:
    """Save checkpoint.

    Extends the base implementation to persist the environment's common_step_counter.
    """
    env_state = {"common_step_counter": self.env.unwrapped.common_step_counter}
    infos = {**(infos or {}), "env_state": env_state}
    super().save(path, infos)

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    """Load checkpoint.

    Extends the base implementation to:
    1. Restore common_step_counter to preserve curricula state.
    2. Migrate legacy checkpoints (actor.* -> mlp.*, actor_obs_normalizer.*
      -> obs_normalizer.*) to the current format (rsl-rl>=4.0).
    """
    loaded_dict = torch.load(path, map_location=map_location, weights_only=False)

    if "model_state_dict" in loaded_dict:
      print(f"Detected legacy checkpoint at {path}. Migrating to new format...")
      model_state_dict = loaded_dict.pop("model_state_dict")
      actor_state_dict = {}
      critic_state_dict = {}

      for key, value in model_state_dict.items():
        # Migrate actor keys.
        if key.startswith("actor."):
          new_key = key.replace("actor.", "mlp.")
          actor_state_dict[new_key] = value
        elif key.startswith("actor_obs_normalizer."):
          new_key = key.replace("actor_obs_normalizer.", "obs_normalizer.")
          actor_state_dict[new_key] = value
        elif key in ["std", "log_std"]:
          actor_state_dict[key] = value

        # Migrate critic keys.
        if key.startswith("critic."):
          new_key = key.replace("critic.", "mlp.")
          critic_state_dict[new_key] = value
        elif key.startswith("critic_obs_normalizer."):
          new_key = key.replace("critic_obs_normalizer.", "obs_normalizer.")
          critic_state_dict[new_key] = value

      loaded_dict["actor_state_dict"] = actor_state_dict
      loaded_dict["critic_state_dict"] = critic_state_dict

    load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
    if load_iteration:
      self.current_learning_iteration = loaded_dict["iter"]

    infos = loaded_dict["infos"]
    if infos and "env_state" in infos:
      # Reset curriculum so it starts from phase 0; policy/optimizer stay from checkpoint.
      self.env.unwrapped.common_step_counter = 0
    return infos
