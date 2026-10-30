"""Policy diagnostics: action statistics, input sensitivity and failure modes for any saved SB3 model.

Two environment modes:

* ``legacy``  - a plain ``CarRacing-v3`` (96x96 RGB, native action space steer[-1,1] / gas[0,1] / brake[0,1],
  optional frame stack), i.e. what a default Stable-Baselines3 notebook trains on. Use it to inspect models trained
  outside this repo (e.g. ``sac_final.zip``).
* ``project`` - this repo's pipeline (gray/resize/stack, rescaled actions). The config is read from the model folder.

Examples::

    python -m src.diagnostics --model sac_final.zip --env legacy --algo sac --episodes 5
    python -m src.diagnostics --model models/ppo_fast/best_stochastic.zip --env project --stochastic

All statistics are measured on rollouts; the printed verdicts are heuristics that point at likely causes, they do not
prove them. Evaluation never updates the model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecFrameStack

from .utils import get_logger, resolve_model_path, setup_logging, write_json

ACTION_NAMES = ("steer", "gas", "brake")


# --------------------------------------------------------------------------- #
# Environments
# --------------------------------------------------------------------------- #
def make_legacy_env(frame_stack: int = 1, lap_complete_percent: float = 0.95, render_mode: Optional[str] = None) -> VecEnv:
    """Plain CarRacing-v3 as a default SB3 setup builds it (no rescaling, RGB, 96x96)."""
    def thunk() -> gym.Env:
        return Monitor(gym.make("CarRacing-v3", continuous=True, lap_complete_percent=lap_complete_percent,
                               render_mode=render_mode))

    venv: VecEnv = DummyVecEnv([thunk])
    if frame_stack > 1:
        venv = VecFrameStack(venv, frame_stack, channels_order="last")
    return venv


def _base_env(venv: VecEnv) -> Optional[gym.Env]:
    node: Any = venv
    while node is not None and not hasattr(node, "envs"):
        node = getattr(node, "venv", None)
    return node.envs[0].unwrapped if node is not None else None


def _to_native_project(action: np.ndarray) -> np.ndarray:
    """Project pipeline action in [-1,1]^3 -> native (steer, gas, brake), see ActionRescaleWrapper."""
    a = np.clip(action, -1.0, 1.0)
    return np.array([a[0], (a[1] + 1.0) / 2.0, max(float(a[2]), 0.0)], dtype=np.float32)


def load_legacy_model(path: Any, algo: str, frame_stack: int = 1, device: str = "auto", render_mode: Optional[str] = None):
    """Load a model trained on a plain CarRacing-v3 setup. Returns (model, venv). Loading never changes the file."""
    venv = make_legacy_env(frame_stack, render_mode=render_mode)
    return {"sac": SAC, "ppo": PPO}[algo].load(str(path), env=venv, device=device), venv


def record_video(model, venv: VecEnv, path: Any, seed: int, *, deterministic: bool = True, max_steps: int = 1000,
                 fps: int = 50, project_pipeline: bool = False) -> Path:
    """Record one episode (needs an env created with render_mode='rgb_array'). Never updates the model."""
    import imageio.v2 as imageio

    frames = []
    if project_pipeline:
        venv.env_method("restart_seed_cycle")
    else:
        venv.seed(int(seed))
    obs = venv.reset()
    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, done, _ = venv.step(action)
        if done[0]:
            break
        frames.append(venv.get_images()[0])
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), frames, fps=fps, codec="libx264", macro_block_size=1, quality=7) if path.suffix == ".mp4" else imageio.mimsave(str(path), frames, duration=1000 / fps)
    return path


# --------------------------------------------------------------------------- #
# Rollouts and statistics
# --------------------------------------------------------------------------- #
def rollout_with_stats(
    model, venv: VecEnv, seeds: Sequence[int], *, deterministic: bool = True,
    to_native: Callable[[np.ndarray], np.ndarray] = lambda a: np.asarray(a, dtype=np.float32),
    seed_each_episode: bool = True, max_steps: int = 1000,
) -> Dict[str, pd.DataFrame]:
    """Run one episode per seed and record every executed action (native units), speed and progress."""
    steps: List[Dict[str, Any]] = []
    episodes: List[Dict[str, Any]] = []
    base = _base_env(venv)
    obs = None
    for ep, seed in enumerate(seeds):
        if seed_each_episode:
            venv.seed(int(seed))  # DummyVecEnv applies it on the next reset
            obs = venv.reset()
        elif obs is None:
            obs = venv.reset()   # cycling-seed envs auto-reset onto the next track after `done`; do not reset twice
        ret, n, first_row = 0.0, 0, len(steps)
        tiles_total = max(len(getattr(base, "track", [])), 1) if base is not None else 1  # track of THIS episode
        info: Dict[str, Any] = {}
        reward = np.zeros(1)
        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            native = to_native(action[0])
            car = getattr(base, "car", None) if base is not None else None
            speed = float(np.linalg.norm(car.hull.linearVelocity)) if car is not None else np.nan  # speed before the step
            obs, reward, done, infos = venv.step(action)
            info = infos[0]
            ret += float(reward[0])
            n += 1
            tiles = int(getattr(base, "tile_visited_count", 0)) if base is not None else 0  # 0 right after auto-reset
            steps.append({"episode": ep, "t": n, "steer": native[0], "gas": native[1], "brake": native[2],
                          "speed": speed, "tiles": tiles, "reward": float(reward[0])})
            if done[0] or n >= max_steps:
                break
        if info.get("lap_finished") is True:
            ended = "lap_complete"
        elif float(reward[0]) < -50:
            ended = "out_of_bounds"      # CarRacing gives -100 when the car leaves the playfield
        else:
            ended = "time_limit"
        tiles_visited = max(r["tiles"] for r in steps[first_row:])  # monotonic within an episode
        episodes.append({"episode": ep, "seed": int(seed), "reward": ret, "steps": n, "ended": ended,
                         "lap_finished": ended == "lap_complete", "tiles_visited": tiles_visited,
                         "tiles_total": tiles_total, "coverage": min(tiles_visited / tiles_total, 1.0)})
        if n >= max_steps and not done[0]:
            obs = venv.reset()
    return {"steps": pd.DataFrame(steps), "episodes": pd.DataFrame(episodes)}


def summarize_actions(steps: pd.DataFrame) -> Dict[str, Any]:
    """Per-dimension distribution plus indicators of collapse / saturation / one-sided steering."""
    out: Dict[str, Any] = {"n_steps": int(len(steps)), "per_dim": {}}
    for name in ACTION_NAMES:
        x = steps[name].to_numpy(dtype=float)
        out["per_dim"][name] = {
            "mean": float(x.mean()), "std": float(x.std()),
            "p05": float(np.percentile(x, 5)), "p50": float(np.percentile(x, 50)), "p95": float(np.percentile(x, 95)),
            "std_within_episode": float(steps.groupby("episode")[name].std().mean()),
            "std_of_episode_means": float(steps.groupby("episode")[name].mean().std(ddof=0)),
        }
    s, g, b = (steps[n].to_numpy(dtype=float) for n in ACTION_NAMES)
    out["indicators"] = {
        "steer_saturated_frac": float(np.mean(np.abs(s) > 0.95)),
        "steer_near_zero_frac": float(np.mean(np.abs(s) < 0.05)),
        "steer_left_frac": float(np.mean(s < -0.05)), "steer_right_frac": float(np.mean(s > 0.05)),
        "gas_saturated_frac": float(np.mean(g > 0.95)), "gas_mean": float(g.mean()),
        "brake_active_frac": float(np.mean(b > 0.1)), "brake_saturated_frac": float(np.mean(b > 0.95)),
        "gas_and_brake_together_frac": float(np.mean((g > 0.1) & (b > 0.1))),
        "mean_speed": float(np.nanmean(steps["speed"])) if steps["speed"].notna().any() else None,
        "zero_speed_frac": float(np.nanmean(steps["speed"] < 0.5)) if steps["speed"].notna().any() else None,
    }
    return out


def verdicts(summary: Dict[str, Any], episodes: pd.DataFrame) -> List[str]:
    """Human-readable, heuristic findings. Thresholds are rules of thumb, not statistical tests."""
    d, ind = summary["per_dim"], summary["indicators"]
    notes: List[str] = []
    if max(d[n]["std_within_episode"] for n in ACTION_NAMES) < 0.03:
        notes.append("POLICY COLLAPSE: every action dimension is (almost) constant inside an episode - the policy ignores "
                     "the image. Typical for SAC-from-pixels after the actor/encoder feature collapse.")
    elif d["steer"]["std_within_episode"] < 0.05:
        notes.append("NO STEERING: steering is nearly constant within episodes (std < 0.05) - it cannot follow curves.")
    if abs(d["steer"]["mean"]) > 0.15 and d["steer"]["std"] < 0.15:
        notes.append(f"STEERING BIAS: constant steering offset (mean {d['steer']['mean']:+.2f}) -> circles/leaves the road.")
    if ind["steer_left_frac"] < 0.05 or ind["steer_right_frac"] < 0.05:
        notes.append("ONE-SIDED STEERING: the agent (almost) never steers in one direction.")
    if ind["steer_saturated_frac"] > 0.4:
        notes.append("STEERING SATURATION: > 40 % of steps use full lock (bang-bang), often a sign of an over-large action noise "
                     "or unstable actor.")
    if ind["brake_active_frac"] > 0.5 and ind["gas_and_brake_together_frac"] > 0.4:
        notes.append("GAS+BRAKE TOGETHER on > 40 % of steps: typical of an affine [-1,1]->[0,1] action mapping that holds "
                     "brake ~0.5 for a fresh policy; the car then crawls.")
    if ind["zero_speed_frac"] is not None and ind["zero_speed_frac"] > 0.5:
        notes.append("STANDING STILL > 50 % of the time (speed < 0.5).")
    if ind["gas_saturated_frac"] > 0.8 and d["steer"]["std_within_episode"] < 0.1:
        notes.append("FULL THROTTLE, NO STEERING: drives straight until it leaves the playfield.")
    if (episodes["steps"] >= 1000).all() and episodes["reward"].mean() < 0:
        notes.append("All episodes hit the 1000-step limit with negative reward: surviving to the time limit is NOT "
                     "evidence of driving; the car is idle/crawling (reward is -0.1/frame without new tiles).")
    if not notes:
        notes.append("No collapse/saturation pattern detected by these heuristics (does not prove good driving - check "
                     "coverage and the video).")
    return notes


def format_report(name: str, summary: Dict[str, Any], episodes: pd.DataFrame, notes: Sequence[str]) -> str:
    lines = [f"=== {name} ===", f"episodes: reward mean {episodes['reward'].mean():.1f} ± {episodes['reward'].std(ddof=0):.1f}, "
             f"coverage {100 * episodes['coverage'].mean():.1f}%, laps {int(episodes['lap_finished'].sum())}/{len(episodes)}, ended: { {k: int(v) for k, v in episodes['ended'].value_counts().items()} }",
             "action        mean    std  p05    p50    p95  | std-in-episode"]
    for n in ACTION_NAMES:
        s = summary["per_dim"][n]
        lines.append(f"{n:6s}     {s['mean']:+6.2f} {s['std']:6.2f} {s['p05']:+5.2f} {s['p50']:+6.2f} {s['p95']:+6.2f}  | {s['std_within_episode']:.3f}")
    lines.append("indicators: " + ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in summary["indicators"].items()))
    lines += ["verdicts:"] + [f"  - {n}" for n in notes]
    return "\n".join(lines)


def plot_action_histograms(steps: pd.DataFrame, path: Path, title: str = "") -> Path:
    from .metrics import get_pyplot

    plt = get_pyplot()
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.4))
    for a, n in zip(ax[:3], ACTION_NAMES):
        a.hist(steps[n], bins=40, color="tab:blue")
        a.set(title=f"{n} (executed, native units)", xlabel=n, ylabel="steps")
    ax[3].plot(steps.loc[steps.episode == steps.episode.iloc[0], "steer"].to_numpy(), lw=0.8, label="steer")
    ax[3].plot(steps.loc[steps.episode == steps.episode.iloc[0], "gas"].to_numpy(), lw=0.8, label="gas")
    ax[3].set(title="first episode over time", xlabel="step"); ax[3].legend(fontsize=8)
    fig.suptitle(title); fig.tight_layout()
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120); plt.close(fig)
    return path


def diagnose(model, venv: VecEnv, seeds: Sequence[int], *, name: str, deterministic: bool,
             to_native: Callable[[np.ndarray], np.ndarray], out_dir: Optional[Path] = None,
             seed_each_episode: bool = True) -> Dict[str, Any]:
    data = rollout_with_stats(model, venv, seeds, deterministic=deterministic, to_native=to_native,
                              seed_each_episode=seed_each_episode)
    summary = summarize_actions(data["steps"])
    notes = verdicts(summary, data["episodes"])
    report = format_report(f"{name} ({'deterministic' if deterministic else 'stochastic'})", summary, data["episodes"], notes)
    print(report)
    result = {"name": name, "deterministic": deterministic, "seeds": list(map(int, seeds)), "summary": summary,
              "verdicts": notes, "episodes": data["episodes"].to_dict(orient="records")}
    if out_dir is not None:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        tag = f"{name}_{'det' if deterministic else 'sto'}"
        write_json(result, out_dir / f"diagnostics_{tag}.json")
        data["steps"].to_csv(out_dir / f"diagnostics_{tag}_steps.csv", index=False)
        plot_action_histograms(data["steps"], out_dir / f"diagnostics_{tag}.png", title=tag)
    return result


def run_comparison(specs: Sequence[Dict[str, Any]], seeds: Sequence[int], out_dir: Optional[Path] = None) -> pd.DataFrame:
    """Evaluate several models on IDENTICAL track seeds and return one summary row per model.

    Each spec: ``{"name", "path", "env": "legacy"|"project", "algo": "sac"|"ppo" (legacy), "deterministic": bool,
    "frame_stack": int (legacy), "config": path (project, optional)}``. A random-action baseline is spec
    ``{"name": "random", "env": "random"}``.
    """
    rows, all_eps = [], []
    for sp in specs:
        det = bool(sp.get("deterministic", True))
        if sp["env"] == "random":
            model, venv = _RandomModel(), make_legacy_env()
            to_native: Callable[[np.ndarray], np.ndarray] = lambda a: np.asarray(a, dtype=np.float32)
            seed_each = True
        elif sp["env"] == "legacy":
            model, venv = load_legacy_model(resolve_model_path(sp["path"]), sp["algo"], sp.get("frame_stack", 1))
            to_native, seed_each = (lambda a: np.asarray(a, dtype=np.float32)), True
        else:
            from .agent import load_model
            from .config import load_config
            from .environment import make_env
            from .utils import find_config_for_model

            mp = resolve_model_path(sp["path"])
            cfg = load_config(find_config_for_model(mp, sp.get("config")))
            venv = make_env(cfg, "eval", eval_seeds=list(seeds), n_envs=1)
            model = load_model(mp, cfg, env=venv)
            venv.env_method("restart_seed_cycle")
            to_native, seed_each = _to_native_project, False
        try:
            data = rollout_with_stats(model, venv, seeds, deterministic=det, to_native=to_native, seed_each_episode=seed_each)
        finally:
            venv.close()
        ep, st = data["episodes"], summarize_actions(data["steps"])
        rows.append({"model": sp["name"], "actions": "deterministic" if det else "stochastic", "episodes": len(ep),
                     "reward_mean": ep.reward.mean(), "reward_std": ep.reward.std(ddof=0), "reward_median": ep.reward.median(),
                     "coverage_mean": ep.coverage.mean(), "laps": int(ep.lap_finished.sum()),
                     "steer_std_in_episode": st["per_dim"]["steer"]["std_within_episode"],
                     "brake_active_frac": st["indicators"]["brake_active_frac"],
                     "verdict": verdicts(st, ep)[0][:60]})
        all_eps.append(ep.assign(model=sp["name"]))
    table = pd.DataFrame(rows)
    if out_dir is not None:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_dir / "comparison.csv", index=False)
        pd.concat(all_eps).to_csv(out_dir / "comparison_episodes.csv", index=False)
    return table


class _RandomModel:
    """Uniform random actions in the legacy (native) action space; stands in for an untrained baseline."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def predict(self, obs, deterministic: bool = True):
        a = self.rng.uniform([-1, 0, 0], [1, 1, 1], size=(obs.shape[0], 3)).astype(np.float32)
        return a, None


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Diagnose a saved CarRacing policy (action statistics, collapse checks).")
    ap.add_argument("--model", required=True)
    ap.add_argument("--env", choices=("legacy", "project"), default="legacy")
    ap.add_argument("--algo", choices=("sac", "ppo"), default=None, help="required for --env legacy (project: read from config)")
    ap.add_argument("--config", help="project mode: config.yaml (default: next to the model)")
    ap.add_argument("--frame-stack", type=int, default=1, help="legacy mode: frames stacked during training")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed-start", type=int, default=50000, help="first track seed (use the SAME value for every model you compare)")
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--both", action="store_true", help="report deterministic and stochastic actions")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", help="directory for JSON / CSV / PNG outputs")
    args = ap.parse_args(argv)
    setup_logging()
    seeds = [args.seed_start + i for i in range(args.episodes)]
    modes = [True, False] if args.both else [not args.stochastic]
    try:
        path = resolve_model_path(args.model)
        if args.env == "legacy":
            if not args.algo:
                raise ValueError("--algo sac|ppo is required with --env legacy.")
            venv = make_legacy_env(args.frame_stack)
            model = {"sac": SAC, "ppo": PPO}[args.algo].load(str(path), env=venv, device=args.device)
            to_native: Callable[[np.ndarray], np.ndarray] = lambda a: np.asarray(a, dtype=np.float32)
            seed_each = True
        else:
            from .agent import load_model
            from .config import load_config
            from .environment import make_env
            from .utils import find_config_for_model

            cfg = load_config(find_config_for_model(path, args.config))
            venv = make_env(cfg, "eval", eval_seeds=seeds, n_envs=1)
            model = load_model(path, cfg, env=venv, device=None if args.device == "auto" else args.device)
            to_native, seed_each = _to_native_project, False
            venv.env_method("restart_seed_cycle")
        try:
            for det in modes:
                diagnose(model, venv, seeds, name=path.stem, deterministic=det, to_native=to_native,
                         out_dir=Path(args.out) if args.out else None, seed_each_episode=seed_each)
        finally:
            venv.close()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        get_logger().error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
