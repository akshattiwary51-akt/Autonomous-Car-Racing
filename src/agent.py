"""Model construction and loading for SAC / PPO with the shared CNN feature extractor."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv

from .config import Config
from .environment import observation_shape
from .feature_extractor import CustomCNN
from .utils import available_ram_gb, estimate_replay_buffer_gb, get_logger, resolve_device

ALGO_CLASSES = {"sac": SAC, "ppo": PPO}


def build_policy_kwargs(cfg: Config) -> Dict[str, Any]:
    """Policy kwargs shared by SAC and PPO (CNN extractor + MLP heads)."""
    p = cfg.algo.policy
    kwargs = {
        "features_extractor_class": CustomCNN,
        "features_extractor_kwargs": {
            "features_dim": p.features_dim,
            "channels": list(p.cnn_channels),
            "kernels": list(p.cnn_kernels),
            "strides": list(p.cnn_strides),
            "norm": p.feature_norm,
            "augment_pad": p.augment_pad,
        },
        "net_arch": list(p.net_arch),
        "share_features_extractor": p.share_features_extractor,
    }
    if cfg.algo.name == "ppo" and p.log_std_init is not None:
        kwargs["log_std_init"] = float(p.log_std_init)
    return kwargs


def _coerce_hyperparams(hp: Dict[str, Any]) -> Dict[str, Any]:
    """YAML 1.1 reads ``3e-4`` as a string and ``[1, step]`` as a list; normalise both."""
    out = dict(hp)
    for key in ("learning_rate", "tau", "gamma", "gae_lambda", "clip_range", "vf_coef",
                "max_grad_norm", "target_kl"):
        if isinstance(out.get(key), str):
            out[key] = float(out[key])
    if isinstance(out.get("ent_coef"), str) and not out["ent_coef"].startswith("auto"):
        out["ent_coef"] = float(out["ent_coef"])
    if isinstance(out.get("train_freq"), list):
        out["train_freq"] = tuple(out["train_freq"])
    return out


def check_replay_buffer_memory(cfg: Config) -> Optional[float]:
    """Fail early (with a fix suggestion) if the SAC replay buffer will not fit in RAM."""
    if cfg.algo.name != "sac":
        return None
    hp = cfg.algo.hyperparams
    shape = observation_shape(cfg)
    optimize = bool(hp.get("optimize_memory_usage", False))
    buf = int(hp.get("buffer_size", 1_000_000))
    need = estimate_replay_buffer_gb(shape, buf, cfg.env.n_envs, optimize_memory_usage=optimize)
    free = available_ram_gb()
    log = get_logger()
    log.info("Replay buffer: %d transitions of %s uint8 -> ~%.2f GiB%s", buf, shape, need,
             f" (free RAM {free:.1f} GiB)" if free is not None else "")
    if free is not None and need > cfg.experiment.track_memory_limit_fraction * free:
        per_item = need / buf
        suggested = int(cfg.experiment.track_memory_limit_fraction * free / per_item)
        raise RuntimeError(
            f"Replay buffer would need ~{need:.2f} GiB but only {free:.2f} GiB RAM is free "
            f"(limit {cfg.experiment.track_memory_limit_fraction:.0%}). Lower algo.hyperparams.buffer_size "
            f"to <= {suggested}, reduce env.resize / env.frame_stack, set optimize_memory_usage: true, "
            "or raise experiment.track_memory_limit_fraction."
        )
    return need


def build_model(
    cfg: Config, env: VecEnv, *, tensorboard_log: Optional[Union[str, Path]] = None, verbose: int = 0,
) -> BaseAlgorithm:
    """Create a fresh SAC/PPO agent for ``env`` from ``cfg``."""
    log = get_logger()
    device = resolve_device(cfg.experiment.device)
    algo_cls = ALGO_CLASSES[cfg.algo.name]
    hp = _coerce_hyperparams(cfg.algo.hyperparams)

    if cfg.algo.name == "sac":
        check_replay_buffer_memory(cfg)
        if hp.get("optimize_memory_usage"):
            # SB3 cannot combine memory-optimised storage with timeout handling.
            rbk = dict(hp.get("replay_buffer_kwargs") or {})
            rbk.setdefault("handle_timeout_termination", False)
            hp["replay_buffer_kwargs"] = rbk
            log.warning("optimize_memory_usage=true: time-limit truncations are treated as terminal "
                        "(handle_timeout_termination=False). This slightly changes the TD targets.")

    model = algo_cls(
        "CnnPolicy", env, policy_kwargs=build_policy_kwargs(cfg), seed=cfg.experiment.seed,
        device=device, tensorboard_log=str(tensorboard_log) if tensorboard_log else None,
        verbose=verbose, **hp,
    )
    n_params = sum(p.numel() for p in model.policy.parameters())
    log.info("Built %s on %s (%.2fM parameters)", cfg.algo.name.upper(), model.device, n_params / 1e6)
    return model


def load_model(
    path: Union[str, Path], cfg: Config, env: Optional[VecEnv] = None, device: Optional[str] = None,
) -> BaseAlgorithm:
    """Load a saved model. ``cfg`` must be the config the model was trained with."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model/checkpoint not found: {path}")
    algo_cls = ALGO_CLASSES[cfg.algo.name]
    dev = resolve_device(device or cfg.experiment.device)
    try:
        return algo_cls.load(str(path), env=env, device=dev)
    except (KeyError, ValueError, RuntimeError) as exc:
        raise RuntimeError(
            f"Could not load {path} as {cfg.algo.name.upper()} ({exc}). The config must match the one used "
            "for training (algorithm, env.resize, env.frame_stack, grayscale, CNN settings)."
        ) from exc
