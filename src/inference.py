"""Run a trained (or random) policy in CarRacing, with optional window, video frames or headless mode.

Examples::

    python -m src.inference --model models/sac_carracing/best_model.zip --episodes 3 --render     # window
    python -m src.inference --model models/sac_carracing/best_model.zip --episodes 2 --headless   # no display
    python -m src.inference --random --episodes 2 --video videos/random.mp4                       # random policy
    python -m src.inference --model models/sac_carracing/best_model.zip --video videos/sac.mp4

Training never renders; this module is the only place where rendering is enabled.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from .config import Config, load_config
from .environment import FINAL_SEED_OFFSET, close_env, make_env
from .evaluate import classify_end, model_policy, random_policy
from .utils import find_config_for_model, get_logger, resolve_model_path, setup_logging, write_json

# Called once per agent step when rendering to arrays: (frame, episode, step, episode_reward, info)
FrameCallback = Callable[[np.ndarray, int, int, float, Dict[str, Any]], None]


def has_display() -> bool:
    """True if a GUI window can plausibly be opened (always true on Windows/macOS desktops)."""
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


def run_inference(
    cfg: Config, model=None, *, episodes: int = 1, window: bool = False, collect_frames: bool = False,
    deterministic: bool = True, seeds: Optional[Sequence[int]] = None, max_steps: Optional[int] = None,
    frame_callback: Optional[FrameCallback] = None, policy_seed: int = 0,
) -> List[Dict[str, Any]]:
    """Drive ``episodes`` episodes and return one result dict per episode.

    Args:
        model: trained SB3 model, or ``None`` for the uniform-random policy.
        window: open the pygame window (``render_mode="human"``; needs a display).
        collect_frames: render RGB frames and pass them to ``frame_callback`` (works headless, e.g. Colab).
        seeds: track seeds, one per episode (default: held-out ``seed + 20000 + i``).
    """
    log = get_logger()
    if window and collect_frames:
        raise ValueError("Choose either a live window or frame collection, not both.")
    if window and not has_display():
        raise RuntimeError("No display found for --render (DISPLAY is not set). On a server or in Colab use "
                           "--headless together with --video or --frames-dir.")
    if not window:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # pygame without a screen
    render_mode = "human" if window else ("rgb_array" if collect_frames else None)
    seeds = list(seeds) if seeds is not None else [cfg.experiment.seed + FINAL_SEED_OFFSET + i for i in range(episodes)]
    if len(seeds) < episodes:
        raise ValueError(f"Need at least {episodes} seeds, got {len(seeds)}.")

    policy = random_policy(policy_seed) if model is None else model_policy(model, deterministic)
    env = make_env(cfg, "visualize", render_mode=render_mode, n_envs=1, eval_seeds=seeds[:episodes])
    out: List[Dict[str, Any]] = []
    try:
        env.env_method("restart_seed_cycle")
        obs = env.reset()
        for ep in range(episodes):
            ep_reward, step, lat, t0 = 0.0, 0, [], time.perf_counter()
            done, info = False, {}
            while not done:
                t = time.perf_counter()
                action = policy(obs)
                lat.append((time.perf_counter() - t) * 1000)
                obs, rewards, dones, infos = env.step(action)
                info, done = infos[0], bool(dones[0])
                ep_reward += float(rewards[0])
                step += 1
                # after a terminal step the VecEnv has already auto-reset, so its frame shows the next track
                if collect_frames and frame_callback is not None and not done:
                    frame_callback(env.get_images()[0], ep, step, ep_reward, info)
                if max_steps is not None and step >= max_steps and not done:
                    info, done = dict(info, truncated=True), True
                    obs = env.reset()
            res = {
                "episode": ep, "seed": seeds[ep], "reward": ep_reward, "steps": step,
                "coverage": float(info.get("coverage", 0.0)), "lap_finished": bool(info.get("lap_finished", False)),
                "end_reason": classify_end(info), "wall_time_s": time.perf_counter() - t0,
                "mean_latency_ms": float(np.mean(lat)),
            }
            out.append(res)
            log.info("episode %d/%d seed=%d | reward %.1f | steps %d | coverage %.1f%% | %s | %.2f ms/step",
                     ep + 1, episodes, res["seed"], res["reward"], res["steps"], 100 * res["coverage"],
                     res["end_reason"], res["mean_latency_ms"])
    finally:
        close_env(env)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Run a CarRacing policy (trained model or random) for visualization.")
    ap.add_argument("--model", help="model .zip or checkpoint directory (omit with --random)")
    ap.add_argument("--config", help="config.yaml used for training (default: saved next to the model)")
    ap.add_argument("--random", action="store_true", help="use a uniform random policy instead of a model")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--render", action="store_true", help="open a live window (needs a display)")
    ap.add_argument("--headless", action="store_true", help="never open a window (default unless --render)")
    ap.add_argument("--video", help="write an MP4 (or GIF) of the run, e.g. videos/run.mp4")
    ap.add_argument("--frames-dir", help="save PNG frames into this directory")
    ap.add_argument("--frame-every", type=int, default=1, help="save every N-th frame with --frames-dir")
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--seed-start", type=int, help="first track seed (default: experiment.seed + 20000)")
    ap.add_argument("--max-steps", type=int, help="cap steps per episode (for quick demos)")
    ap.add_argument("--device", help="override experiment.device")
    ap.add_argument("--out", help="write episode results to this JSON file")
    args = ap.parse_args(argv)

    setup_logging()
    try:
        if args.render and args.headless:
            raise ValueError("--render and --headless are mutually exclusive.")
        if not args.random and not args.model:
            raise ValueError("Pass --model <path> or --random.")
        model, cfg = None, None
        if args.model:
            model_path = resolve_model_path(args.model)
            cfg = load_config(find_config_for_model(model_path, args.config))
        else:
            cfg = load_config(args.config or Path(__file__).resolve().parents[1] / "configs" / "sac.yaml")
        if args.device:
            cfg.experiment.device = args.device
        if args.random and args.model:
            get_logger().warning("--random given: ignoring --model.")
        elif args.model:
            from .agent import load_model

            probe = make_env(cfg, "visualize", n_envs=1)
            try:
                model = load_model(model_path, cfg, env=probe)
            finally:
                close_env(probe)
        seeds = None if args.seed_start is None else [args.seed_start + i for i in range(args.episodes)]

        want_frames = bool(args.video or args.frames_dir)
        if args.render and want_frames:
            raise ValueError("--render cannot be combined with --video/--frames-dir; record headless instead.")
        sink = None
        if want_frames:
            from .record_video import FrameRecorder

            sink = FrameRecorder(video_path=args.video, frames_dir=args.frames_dir, frame_every=args.frame_every,
                                 fps=max(1, int(round(50 / cfg.env.action_repeat))))
        results = run_inference(cfg, model, episodes=args.episodes, window=args.render, collect_frames=want_frames,
                                deterministic=not args.stochastic, seeds=seeds, max_steps=args.max_steps,
                                frame_callback=sink.add if sink else None)
        if sink:
            for p in sink.close():
                get_logger().info("Wrote %s", p)
        if args.out:
            write_json({"policy": "random" if model is None else "model", "episodes": results}, Path(args.out))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        get_logger().error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
