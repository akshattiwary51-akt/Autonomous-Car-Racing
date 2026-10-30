"""Shared helpers: seeding, device selection, logging, memory estimates, metadata."""

from __future__ import annotations

import json
import logging
import platform
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

LOGGER_NAME = "carracing"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    return logging.getLogger(name)


def setup_logging(log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    """Configure console (and optional file) logging once; safe to call repeatedly."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in logger.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        if not any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_file.resolve()
                   for h in logger.handlers):
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    return logger


def set_global_seed(seed: int) -> None:
    """Seed python, numpy and torch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # torch is validated elsewhere with a clearer message
        pass


def resolve_device(requested: str = "auto") -> str:
    """Resolve 'auto' to cuda/mps/cpu and fail clearly if an explicit device is unavailable."""
    import torch

    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"  # mps is intentionally not auto-selected: small CNN + CPU env is usually faster on CPU
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Device '{requested}' requested but CUDA is not available "
            f"(torch {torch.__version__}, built for CUDA {torch.version.cuda}). "
            "Use --device cpu or install a CUDA-enabled PyTorch build."
        )
    if requested == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("Device 'mps' requested but is not available.")
    return requested


def estimate_replay_buffer_gb(
    obs_shape: Sequence[int], buffer_size: int, n_envs: int = 1,
    action_dim: int = 3, optimize_memory_usage: bool = False,
) -> float:
    """Approximate SB3 ReplayBuffer RAM use in GiB for uint8 image observations."""
    obs_bytes = int(np.prod(obs_shape))  # uint8
    n_obs_copies = 1 if optimize_memory_usage else 2  # obs + next_obs
    per_transition = n_obs_copies * obs_bytes + action_dim * 4 + 4 + 4 + 4  # action, reward, done, timeout
    return per_transition * (buffer_size // max(n_envs, 1)) * n_envs / 1024 ** 3


def available_ram_gb() -> Optional[float]:
    try:
        import psutil

        return psutil.virtual_memory().available / 1024 ** 3
    except ImportError:
        return None


def process_rss_gb() -> Optional[float]:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1024 ** 3
    except ImportError:
        return None


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def collect_system_metadata() -> Dict[str, Any]:
    """Versions and hardware info stored next to every run for reproducibility."""
    meta: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": __import__("os").cpu_count(),
        "git_commit": _git_commit(),
    }
    for mod in ("numpy", "torch", "gymnasium", "stable_baselines3", "Box2D", "pygame"):
        try:
            m = __import__(mod)
            meta[mod] = getattr(m, "__version__", "unknown")
        except ImportError:
            meta[mod] = None
    try:
        import torch

        meta["cuda_available"] = torch.cuda.is_available()
        meta["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    ram = available_ram_gb()
    meta["available_ram_gb"] = round(ram, 2) if ram is not None else None
    return meta


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(f"Not JSON serialisable: {type(o)}")

    path.write_text(json.dumps(obj, indent=2, default=default), encoding="utf-8")


def load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Corrupt JSON file {path}: {exc}") from exc


def run_dirs(run_name: str, root: Optional[Path] = None) -> Dict[str, Path]:
    """Per-run output directories (models/<run>, checkpoints/<run>, ...)."""
    root = root or PROJECT_ROOT
    return {k: root / k / run_name for k in ("models", "checkpoints", "logs", "results", "videos")}


def find_latest_checkpoint(directory: Path) -> Path:
    """Return the checkpoint .zip with the highest step count in ``directory``."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {directory}")
    candidates = []
    for p in directory.glob("*_steps.zip"):
        try:
            candidates.append((int(p.stem.split("_")[-2]), p))
        except (ValueError, IndexError):
            continue
    if not candidates:
        raise FileNotFoundError(f"No '*_steps.zip' checkpoints found in {directory}")
    return max(candidates)[1]


def resolve_model_path(path: str | Path) -> Path:
    """Accept a model .zip (with or without extension) and raise a helpful error if missing."""
    p = Path(path)
    if p.suffix != ".zip":
        p = p.with_suffix(".zip") if not p.is_dir() else p
    if p.is_dir():
        return find_latest_checkpoint(p)
    if not p.is_file():
        raise FileNotFoundError(
            f"Model file not found: {p}. Train one first (python -m src.train ...) "
            "or pass an existing .zip checkpoint."
        )
    return p


def find_config_for_model(model_path: Path, explicit: Optional[str | Path] = None) -> Path:
    """Locate the config.yaml saved alongside a model/checkpoint."""
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise FileNotFoundError(f"Config file not found: {p}")
        return p
    for folder in (model_path.parent, model_path.parent.parent):
        cand = folder / "config.yaml"
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"No config.yaml found next to {model_path}. Pass --config <path> explicitly "
        "(it must match the config the model was trained with)."
    )


def nested_get(d: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    node: Any = d
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def summarize_shape(shape: Tuple[int, ...]) -> str:
    return "x".join(str(s) for s in shape)
