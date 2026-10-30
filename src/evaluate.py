"""Independent policy evaluation on a separate, seed-controlled environment.

CLI::

    python -m src.evaluate --model models/sac_carracing/best_model.zip --episodes 10 --compare-baselines
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from stable_baselines3.common.vec_env import VecEnv

from .config import Config, load_config
from .environment import FINAL_SEED_OFFSET, close_env, make_env
from .metrics import (EpisodeResult, plot_eval_comparison, results_to_dataframe, save_results, summarize)
from .utils import (find_config_for_model, get_logger, resolve_model_path, run_dirs, set_global_seed,
                    setup_logging)

PolicyFn = Callable[[np.ndarray], np.ndarray]


def classify_end(info: Dict[str, Any]) -> str:
    """Why an episode ended, from the explicit flags added by ``TrackProgressWrapper``."""
    if info.get("lap_finished"):
        return "lap_complete"
    if info.get("stuck_truncated"):
        return "stuck_truncated"
    if info.get("terminated"):
        return "out_of_bounds"   # CarRacing terminates without a finished lap only when leaving the playfield
    if info.get("truncated"):
        return "time_limit"
    return "unknown"


def evaluate_episodes(policy_fn: PolicyFn, venv: VecEnv, n_episodes: int) -> Tuple[List[EpisodeResult], np.ndarray]:
    """Run ``n_episodes`` complete episodes on a single-env eval ``venv``.

    The env's seed cycle is restarted first, so repeated calls (and different policies) see the
    same tracks. Returns per-episode results and per-step inference latencies in ms.
    """
    if venv.num_envs != 1:
        raise ValueError("evaluate_episodes expects a single-environment VecEnv.")
    venv.env_method("restart_seed_cycle")
    seeds = venv.get_attr("seeds")[0]
    obs = venv.reset()
    results: List[EpisodeResult] = []
    latencies: List[float] = []
    ep_lat: List[float] = []
    ep_reward, ep_len, t_ep = 0.0, 0, time.perf_counter()
    while len(results) < n_episodes:
        t0 = time.perf_counter()
        action = policy_fn(obs)
        dt_ms = (time.perf_counter() - t0) * 1000
        latencies.append(dt_ms)
        ep_lat.append(dt_ms)
        obs, rewards, dones, infos = venv.step(action)
        ep_reward += float(rewards[0])
        ep_len += 1
        if dones[0]:
            info = infos[0]
            idx = len(results)
            results.append(EpisodeResult(
                episode=idx, seed=int(seeds[idx % len(seeds)]), reward=ep_reward, length=ep_len,
                coverage=float(info.get("coverage", 0.0)), lap_finished=bool(info.get("lap_finished", False)),
                end_reason=classify_end(info), wall_time_s=time.perf_counter() - t_ep,
                mean_latency_ms=float(np.mean(ep_lat)),
            ))
            ep_reward, ep_len, ep_lat, t_ep = 0.0, 0, [], time.perf_counter()
    return results, np.asarray(latencies)


def model_policy(model, deterministic: bool = True) -> PolicyFn:
    def fn(obs: np.ndarray) -> np.ndarray:
        action, _ = model.predict(obs, deterministic=deterministic)
        return action
    return fn


def random_policy(seed: int = 0) -> PolicyFn:
    """Uniform random actions in the agent action space [-1, 1]^3."""
    rng = np.random.default_rng(seed)
    return lambda obs: rng.uniform(-1.0, 1.0, size=(obs.shape[0], 3)).astype(np.float32)


def final_eval_seeds(cfg: Config, n_episodes: int, start: Optional[int] = None) -> List[int]:
    first = cfg.experiment.seed + FINAL_SEED_OFFSET if start is None else start
    return [first + i for i in range(n_episodes)]


def evaluate_policy_on_seeds(policy_fn: PolicyFn, cfg: Config, seeds: Sequence[int]) -> Dict[str, Any]:
    """Create a fresh eval env on ``seeds``, evaluate ``policy_fn`` once per seed, close the env."""
    env = make_env(cfg, "eval", eval_seeds=list(seeds), n_envs=1)
    try:
        t0 = time.perf_counter()
        results, lat = evaluate_episodes(policy_fn, env, len(seeds))
        runtime = time.perf_counter() - t0
    finally:
        close_env(env)
    return {"results": results, "summary": summarize(results, lat, runtime)}


def run_evaluation(
    cfg: Config, model, *, n_episodes: int, deterministic: bool = True, seed_start: Optional[int] = None,
    out_dir: Optional[Path] = None, compare_baselines: bool = False, prefix: str = "eval",
) -> Dict[str, Any]:
    """Evaluate ``model`` (and optional baselines) on identical held-out tracks and export results."""
    log = get_logger()
    seeds = final_eval_seeds(cfg, n_episodes, seed_start)
    mode = "deterministic" if deterministic else "stochastic"
    policies: Dict[str, PolicyFn] = {f"trained_{mode}": model_policy(model, deterministic)}
    if compare_baselines:
        from .agent import build_model

        probe = make_env(cfg, "eval", eval_seeds=seeds, n_envs=1)
        try:
            set_global_seed(cfg.experiment.seed)
            untrained = build_model(cfg, probe)
        finally:
            close_env(probe)
        policies["untrained_network"] = model_policy(untrained, deterministic=True)
        policies["random_uniform"] = random_policy(cfg.experiment.seed)

    summaries: Dict[str, Dict[str, Any]] = {}
    frames = {}
    for name, fn in policies.items():
        out = evaluate_policy_on_seeds(fn, cfg, seeds)
        summaries[name] = out["summary"]
        frames[name] = results_to_dataframe(out["results"])
        s = out["summary"]
        log.info("%-20s reward %.1f ± %.1f (median %.1f) | coverage %.1f%% | laps %.0f%% | %s | %.2f ms/step",
                 name, s["reward_mean"], s["reward_std"], s["reward_median"], 100 * s["coverage_mean"],
                 100 * s["lap_completion_rate"], s["termination_counts"], s["inference_latency_ms"]["mean"])
        if out_dir is not None:
            save_results(out["results"], s, out_dir, prefix=f"{prefix}_{name}")
    if out_dir is not None:
        plot_eval_comparison(summaries, frames, Path(out_dir) / f"{prefix}_comparison.png",
                             title=f"Evaluation on {n_episodes} held-out tracks")
    return {"seeds": seeds, "summaries": summaries}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate a trained CarRacing agent on held-out tracks.")
    ap.add_argument("--model", required=True, help="model .zip, or a checkpoint directory (latest is used)")
    ap.add_argument("--config", help="config.yaml used for training (default: the one saved next to the model)")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--stochastic", action="store_true", help="sample actions instead of using the mean")
    ap.add_argument("--compare-baselines", action="store_true", help="also evaluate an untrained net and a random policy")
    ap.add_argument("--seed-start", type=int, help="first evaluation track seed (default: experiment.seed + 20000)")
    ap.add_argument("--device", help="override experiment.device (cpu | cuda)")
    ap.add_argument("--out", help="output directory (default: results/<run>/)")
    args = ap.parse_args(argv)

    setup_logging()
    try:
        model_path = resolve_model_path(args.model)
        cfg = load_config(find_config_for_model(model_path, args.config))
        if args.device:
            cfg.experiment.device = args.device
        from .agent import load_model

        env = make_env(cfg, "eval", n_envs=1)
        try:
            model = load_model(model_path, cfg, env=env)
        finally:
            close_env(env)
        out_dir = Path(args.out) if args.out else run_dirs(cfg.experiment.name)["results"]
        run_evaluation(cfg, model, n_episodes=args.episodes, deterministic=not args.stochastic,
                       seed_start=args.seed_start, out_dir=out_dir, compare_baselines=args.compare_baselines,
                       prefix=f"eval_{model_path.stem}")
        get_logger().info("Results written to %s", out_dir)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        get_logger().error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
