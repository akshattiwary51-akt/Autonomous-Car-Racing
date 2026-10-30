"""Typed, validated experiment configuration loaded from YAML/JSON.

Unknown keys are rejected so that typos in a config file fail loudly instead of
being silently ignored.
"""

from __future__ import annotations

import copy
import json
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

SUPPORTED_ALGOS = ("sac", "ppo")

# Hyperparameters forwarded to the SB3 constructors. Anything else is an error.
_ALLOWED_ALGO_KEYS = {
    "sac": {
        "learning_rate", "buffer_size", "learning_starts", "batch_size", "tau", "gamma",
        "train_freq", "gradient_steps", "ent_coef", "target_update_interval",
        "target_entropy", "use_sde", "sde_sample_freq", "optimize_memory_usage",
        "replay_buffer_kwargs",
    },
    "ppo": {
        "learning_rate", "n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda",
        "clip_range", "ent_coef", "vf_coef", "max_grad_norm", "use_sde",
        "sde_sample_freq", "target_kl",
    },
}


class ConfigError(ValueError):
    """Raised when a configuration file is missing, malformed, or invalid."""


@dataclass
class ShapingConfig:
    """Optional reward shaping (training env only; evaluation always uses native reward)."""

    enabled: bool = False
    steer_change_penalty: float = 0.0   # coef * |steer_t - steer_{t-1}|
    brake_gas_penalty: float = 0.0      # coef * min(gas, brake)
    stuck_patience: int = 0             # truncate after N consecutive non-positive native rewards (0=off)
    stuck_penalty: float = 0.0          # extra penalty applied on the stuck truncation step
    speed_bonus: float = 0.0            # + coef * min(speed/speed_ref, 1.5) while still making track progress (TASK-MODIFYING)
    speed_ref: float = 60.0             # speed (simulator units) that earns the full bonus coefficient
    progress_window: int = 8            # bonus only if a new tile was reached within the last N agent steps


@dataclass
class EnvConfig:
    id: str = "CarRacing-v3"
    continuous: bool = True
    lap_complete_percent: float = 0.95
    domain_randomize: bool = False
    grayscale: bool = True
    resize: Optional[int] = 64          # square side in pixels; None keeps native 96x96
    frame_stack: int = 4
    action_repeat: int = 2
    n_envs: int = 1
    vec_type: str = "dummy"             # "dummy" or "subproc"
    reward_scale: float = 1.0           # training env only: multiply reward (eval/Monitor keep the native reward)
    reward_shaping: ShapingConfig = field(default_factory=ShapingConfig)


@dataclass
class PolicyConfig:
    features_dim: int = 256
    log_std_init: Optional[float] = None  # PPO only: initial log std of the Gaussian policy (SB3 default 0)
    augment_pad: int = 0                # DrQ random-shift padding in px (0 = off; 4 is typical for 64 px)
    feature_norm: str = "none"          # "none" | "layernorm" (recommended: prevents feature collapse)
    cnn_channels: List[int] = field(default_factory=lambda: [32, 64, 64])
    cnn_kernels: List[int] = field(default_factory=lambda: [8, 4, 3])
    cnn_strides: List[int] = field(default_factory=lambda: [4, 2, 1])
    net_arch: List[int] = field(default_factory=lambda: [256, 256])
    share_features_extractor: bool = False


@dataclass
class AlgoConfig:
    name: str = "sac"
    hyperparams: Dict[str, Any] = field(default_factory=dict)
    policy: PolicyConfig = field(default_factory=PolicyConfig)


@dataclass
class TrainingConfig:
    total_timesteps: int = 500_000
    checkpoint_freq: int = 25_000       # in agent steps (summed over envs)
    save_replay_buffer: bool = False
    log_interval: int = 4
    progress_bar: bool = False
    resume_warmup_steps: int = 1000     # random-action refill of an empty replay buffer after resume


@dataclass
class EvaluationConfig:
    eval_freq: int = 10_000             # in agent steps (summed over envs)
    n_eval_episodes: int = 5
    deterministic: bool = True
    final_n_episodes: int = 10          # post-training evaluation on held-out seeds


@dataclass
class ExperimentConfig:
    name: str = "sac_carracing"
    seed: int = 0
    device: str = "auto"                # auto | cpu | cuda | cuda:0 ...
    track_memory_limit_fraction: float = 0.6  # max share of free RAM the replay buffer may use
    torch_threads: Optional[int] = None  # None = PyTorch default; try 1-2 on small CPU machines


@dataclass
class Config:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    algo: AlgoConfig = field(default_factory=AlgoConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Construction helpers
# --------------------------------------------------------------------------- #
def _build(cls: type, data: Dict[str, Any], path: str) -> Any:
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"Section '{path}' must be a mapping, got {type(data).__name__}.")
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ConfigError(
            f"Unknown key(s) in '{path}': {sorted(unknown)}. Valid keys: {sorted(known)}."
        )
    kwargs: Dict[str, Any] = {}
    for name, value in data.items():
        factory = known[name].default_factory
        template = factory() if factory is not MISSING else None
        if template is not None and is_dataclass(template):
            kwargs[name] = _build(type(template), value, f"{path}.{name}")
        else:
            kwargs[name] = value
    return cls(**kwargs)


def config_from_dict(data: Dict[str, Any]) -> Config:
    cfg = _build(Config, copy.deepcopy(data), "root")
    validate_config(cfg)
    return cfg


def load_config(path: Union[str, Path]) -> Config:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
    except (yaml.YAMLError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Could not parse {path}: {exc}") from exc
    return config_from_dict(data or {})


def save_config(cfg: Config, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")


def apply_overrides(cfg: Config, overrides: List[str]) -> Config:
    """Apply ``a.b.c=value`` overrides (value parsed as YAML) and re-validate."""
    data = cfg.to_dict()
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"Override '{item}' must look like section.key=value.")
        key, raw = item.split("=", 1)
        node = data
        parts = key.split(".")
        for part in parts[:-1]:
            if not isinstance(node, dict) or part not in node:
                raise ConfigError(f"Override path '{key}' does not exist in the config.")
            node = node[part]
        if not isinstance(node, dict):
            raise ConfigError(f"Override path '{key}' does not exist in the config.")
        if parts[-1] not in node and parts[-2:-1] != ["hyperparams"]:
            raise ConfigError(f"Override key '{key}' does not exist in the config.")
        node[parts[-1]] = yaml.safe_load(raw)
    return config_from_dict(data)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_config(cfg: Config) -> None:
    errors: List[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            errors.append(msg)

    e, a, t, v, x = cfg.env, cfg.algo, cfg.training, cfg.evaluation, cfg.experiment
    check(a.name in SUPPORTED_ALGOS, f"algo.name must be one of {SUPPORTED_ALGOS}, got '{a.name}'.")
    check(e.id == "CarRacing-v3", f"env.id must be 'CarRacing-v3' (v2 was removed in Gymnasium 1.x), got '{e.id}'.")
    check(e.continuous, "env.continuous must be true (SAC requires a continuous action space).")
    check(0.0 < e.lap_complete_percent <= 1.0, "env.lap_complete_percent must be in (0, 1].")
    check(e.resize is None or (isinstance(e.resize, int) and 16 <= e.resize <= 96),
          "env.resize must be null or an integer in [16, 96].")
    check(e.frame_stack >= 1, "env.frame_stack must be >= 1.")
    check(e.action_repeat >= 1, "env.action_repeat must be >= 1.")
    check(e.reward_scale > 0, "env.reward_scale must be > 0.")
    check(e.n_envs >= 1, "env.n_envs must be >= 1.")
    check(e.vec_type in ("dummy", "subproc"), "env.vec_type must be 'dummy' or 'subproc'.")
    s = e.reward_shaping
    check(s.steer_change_penalty >= 0 and s.brake_gas_penalty >= 0 and s.stuck_penalty >= 0,
          "reward_shaping penalties must be >= 0.")
    check(s.speed_bonus >= 0 and s.speed_ref > 0 and s.progress_window >= 1,
          "reward_shaping.speed_bonus >= 0, speed_ref > 0 and progress_window >= 1 are required.")
    check(s.stuck_patience >= 0, "reward_shaping.stuck_patience must be >= 0.")

    p = a.policy
    check(p.feature_norm in ("none", "layernorm"), "algo.policy.feature_norm must be 'none' or 'layernorm'.")
    check(p.augment_pad >= 0, "algo.policy.augment_pad must be >= 0.")
    check(p.features_dim >= 8, "algo.policy.features_dim must be >= 8.")
    check(len(p.cnn_channels) == len(p.cnn_kernels) == len(p.cnn_strides) and len(p.cnn_channels) > 0,
          "cnn_channels, cnn_kernels and cnn_strides must be non-empty and of equal length.")
    check(all(isinstance(n, int) and n > 0 for n in p.net_arch) and len(p.net_arch) > 0,
          "algo.policy.net_arch must be a non-empty list of positive integers.")

    if a.name in _ALLOWED_ALGO_KEYS:
        bad = set(a.hyperparams) - _ALLOWED_ALGO_KEYS[a.name]
        check(not bad, f"Unsupported {a.name} hyperparameter(s): {sorted(bad)}. "
                       f"Allowed: {sorted(_ALLOWED_ALGO_KEYS[a.name])}.")
        h = a.hyperparams
        check(0 < float(h.get("gamma", 0.99)) <= 1, "gamma must be in (0, 1].")
        check(int(h.get("batch_size", 64)) >= 1, "batch_size must be >= 1.")
        if a.name == "sac":
            check(0 < float(h.get("tau", 0.005)) <= 1, "tau must be in (0, 1].")
            check(int(h.get("buffer_size", 1)) >= 1, "buffer_size must be >= 1.")
            check(int(h.get("learning_starts", 0)) >= 0, "learning_starts must be >= 0.")
            check(int(h.get("gradient_steps", 1)) >= -1, "gradient_steps must be >= -1.")
        if a.name == "ppo":
            n_steps = int(h.get("n_steps", 2048))
            check(n_steps >= 1, "n_steps must be >= 1.")

    check(t.total_timesteps > 0, "training.total_timesteps must be > 0.")
    check(t.checkpoint_freq > 0, "training.checkpoint_freq must be > 0.")
    check(t.resume_warmup_steps >= 0, "training.resume_warmup_steps must be >= 0.")
    check(x.torch_threads is None or (isinstance(x.torch_threads, int) and x.torch_threads >= 1),
          "experiment.torch_threads must be null or an integer >= 1.")
    check(v.eval_freq > 0, "evaluation.eval_freq must be > 0.")
    check(v.n_eval_episodes >= 1, "evaluation.n_eval_episodes must be >= 1.")
    check(v.final_n_episodes >= 0, "evaluation.final_n_episodes must be >= 0.")
    check(isinstance(x.seed, int) and x.seed >= 0, "experiment.seed must be a non-negative integer.")
    check(bool(x.name) and all(c.isalnum() or c in "-_." for c in x.name),
          "experiment.name may only contain letters, digits, '-', '_' and '.'.")
    check(x.device == "auto" or x.device.split(":")[0] in ("cpu", "cuda", "mps"),
          "experiment.device must be 'auto', 'cpu', 'cuda', 'cuda:N' or 'mps'.")
    check(0 < x.track_memory_limit_fraction <= 1, "experiment.track_memory_limit_fraction must be in (0, 1].")

    errors = [m for m in errors if m]
    if errors:
        raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(errors))
