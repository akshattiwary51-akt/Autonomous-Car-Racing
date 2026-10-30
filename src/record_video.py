"""Video / frame recording for CarRacing rollouts (works headless, e.g. in Google Colab).

CLI (thin wrapper around :mod:`src.inference`)::

    python -m src.record_video --model models/sac_carracing/best_model.zip --episodes 2 --out videos/sac.mp4
    python -m src.record_video --random --episodes 1 --out videos/random.mp4

Python::

    from src.record_video import record_episodes
    record_episodes(cfg, model, "videos/demo.mp4", episodes=1)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

from .config import Config
from .utils import get_logger


def overlay_hud(frame: np.ndarray, episode: int, step: int, reward: float, info: Dict[str, Any],
                width: int = 600) -> np.ndarray:
    """Downscale ``frame`` to ``width`` px wide and draw episode / step / reward / coverage on top."""
    h, w = frame.shape[:2]
    scale = width / w
    img = cv2.resize(frame, (width, int(h * scale)), interpolation=cv2.INTER_AREA)
    text = f"ep {episode + 1}  step {step}  reward {reward:7.1f}  coverage {100 * info.get('coverage', 0.0):4.1f}%"
    cv2.rectangle(img, (0, 0), (width, 22), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.ascontiguousarray(img)


def save_video(frames: Sequence[np.ndarray], path: Path, fps: int = 25) -> Path:
    """Write frames to MP4 (H.264 via imageio-ffmpeg). Falls back to GIF with a visible warning."""
    import imageio.v2 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(frames) == 0:
        raise ValueError("No frames to write.")
    if path.suffix.lower() == ".gif":
        imageio.mimsave(path, list(frames), duration=1000 / fps, loop=0)
        return path
    try:
        with imageio.get_writer(str(path), fps=fps, codec="libx264", macro_block_size=1, quality=7) as w:
            for f in frames:
                w.append_data(f)
        return path
    except (ImportError, RuntimeError, ValueError, OSError) as exc:
        gif = path.with_suffix(".gif")
        get_logger().warning("MP4 encoding failed (%s); writing %s instead. Install imageio-ffmpeg for MP4.", exc, gif)
        imageio.mimsave(gif, list(frames), duration=1000 / fps, loop=0)
        return gif


class FrameRecorder:
    """Collects frames (with HUD) and writes them to a video file and/or PNG files on ``close()``."""

    def __init__(self, video_path: Optional[str] = None, frames_dir: Optional[str] = None,
                 frame_every: int = 1, fps: int = 25, hud: bool = True) -> None:
        self.video_path = Path(video_path) if video_path else None
        self.frames_dir = Path(frames_dir) if frames_dir else None
        self.frame_every, self.fps, self.hud = max(1, frame_every), fps, hud
        self.frames: List[np.ndarray] = []
        self._count = 0

    def add(self, frame: np.ndarray, episode: int, step: int, reward: float, info: Dict[str, Any]) -> None:
        if self.hud:
            frame = overlay_hud(frame, episode, step, reward, info)
        if self.video_path is not None:
            self.frames.append(frame)
        if self.frames_dir is not None and self._count % self.frame_every == 0:
            self.frames_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(self.frames_dir / f"ep{episode:03d}_step{step:05d}.png"), frame[..., ::-1])
        self._count += 1

    def close(self) -> List[Path]:
        written: List[Path] = []
        if self.video_path is not None and self.frames:
            written.append(save_video(self.frames, self.video_path, self.fps))
        if self.frames_dir is not None:
            written.append(self.frames_dir)
        return written


def record_episodes(cfg: Config, model, path: str, *, episodes: int = 1, deterministic: bool = True,
                    seeds: Optional[Sequence[int]] = None, max_steps: Optional[int] = None):
    """Record ``episodes`` rollouts of ``model`` (``None`` = random policy). Returns (video path, results)."""
    from .inference import run_inference

    rec = FrameRecorder(video_path=path, fps=max(1, int(round(50 / cfg.env.action_repeat))))
    results = run_inference(cfg, model, episodes=episodes, collect_frames=True, deterministic=deterministic,
                            seeds=seeds, max_steps=max_steps, frame_callback=rec.add)
    written = rec.close()
    return (written[0] if written else None), results


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Record a video of a CarRacing policy.")
    ap.add_argument("--out", required=True, help="output .mp4 or .gif path")
    rest = list(argv if argv is not None else sys.argv[1:])
    args, passthrough = ap.parse_known_args(rest)
    from .inference import main as inference_main

    return inference_main([*passthrough, "--video", args.out, "--headless"])


if __name__ == "__main__":
    sys.exit(main())
