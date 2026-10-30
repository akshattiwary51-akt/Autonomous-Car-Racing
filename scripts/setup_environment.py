#!/usr/bin/env python
"""Verify that the installed dependencies can run this project, with actionable error messages.

    python scripts/setup_environment.py            # check only
    python scripts/setup_environment.py --install  # pip install -r requirements.txt first
"""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# module, pip name, minimum version
REQUIRED = [
    ("numpy", "numpy", "1.24"), ("gymnasium", "gymnasium[box2d]", "1.0"), ("Box2D", "box2d", "2.3"),
    ("pygame", "pygame-ce or pygame", "2.1"), ("torch", "torch", "2.0"), ("stable_baselines3", "stable-baselines3", "2.3"),
    ("yaml", "pyyaml", "6.0"), ("pandas", "pandas", "1.5"), ("matplotlib", "matplotlib", "3.7"),
    ("cv2", "opencv-python-headless", "4.5"), ("tensorboard", "tensorboard", "2.10"), ("psutil", "psutil", "5.9"),
    ("imageio", "imageio", "2.20"), ("pytest", "pytest", "7.0"),
]
OPTIONAL = [("plotly", "plotly"), ("imageio_ffmpeg", "imageio-ffmpeg")]


def _ver(v: str) -> tuple:
    out = []
    for part in v.split(".")[:3]:
        digits = "".join(c for c in part if c.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--install", action="store_true", help="run 'pip install -r requirements.txt' first")
    args = ap.parse_args()
    if args.install:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")])

    problems: list[str] = []
    print(f"Python {platform.python_version()} on {platform.platform()}")
    if sys.version_info < (3, 10):
        problems.append("Python >= 3.10 is required (3.11-3.13 verified).")

    for mod, pip_name, minimum in REQUIRED:
        try:
            m = importlib.import_module(mod)
            version = getattr(m, "__version__", "?")
            ok = version == "?" or _ver(version) >= _ver(minimum)
            print(f"  {'OK ' if ok else 'OLD'} {mod:18s} {version}")
            if not ok:
                problems.append(f"{mod} {version} is older than {minimum}: pip install -U '{pip_name}'")
        except ImportError as exc:
            print(f"  MISSING {mod}")
            hint = f"pip install '{pip_name}'"
            if mod == "Box2D":
                hint += " (Windows/macOS: needs a wheel or MSVC/swig; try 'pip install box2d' or 'conda install -c conda-forge box2d-py')"
            problems.append(f"{mod} is not importable ({exc}). Fix: {hint}")
    for mod, pip_name in OPTIONAL:
        try:
            importlib.import_module(mod)
            print(f"  OK  {mod} (optional)")
        except ImportError:
            print(f"  --  {mod} not installed (optional): pip install {pip_name}")

    if not problems:
        try:
            import gymnasium as gym
            import torch

            env = gym.make("CarRacing-v3", continuous=True)
            obs, _ = env.reset(seed=0)
            obs, r, term, trunc, _ = env.step(env.action_space.sample())
            env.close()
            print(f"  CarRacing-v3: obs {obs.shape} {obs.dtype}, step OK")
            try:
                gym.make("CarRacing-v2")
            except Exception as exc:  # noqa: BLE001 - informational only
                print(f"  note: CarRacing-v2 is unavailable in this Gymnasium ({type(exc).__name__}); v3 is used.")
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"  PyTorch device: {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""))
            if dev == "cpu":
                print("  note: no GPU detected - training works on CPU but is slower (see README).")
            print(f"  display: {'yes' if os.environ.get('DISPLAY') or not sys.platform.startswith('linux') else 'no (use --headless / --video)'}")
        except Exception as exc:  # noqa: BLE001 - report any failure with context
            problems.append(f"CarRacing-v3 smoke test failed: {type(exc).__name__}: {exc}")

    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
