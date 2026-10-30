#!/usr/bin/env python
"""Reproducible throughput benchmark: simulator speed and SAC training speed for config variants.

    python scripts/benchmark.py                       # default variants (a few minutes on CPU)
    python scripts/benchmark.py --steps 500 --out results/benchmarks

Each training variant: (1) collect ``learning_starts`` random steps (not timed), (2) time ``--steps`` agent
steps that include one gradient update per step (train_freq/gradient_steps from the variant).
Results are written to <out>/benchmark.csv and .json. Nothing here is a claim about learning quality.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.agent import build_model  # noqa: E402
from src.config import apply_overrides, load_config  # noqa: E402
from src.environment import close_env, make_env  # noqa: E402
from src.utils import collect_system_metadata, process_rss_gb, set_global_seed, write_json  # noqa: E402

BASE = ["algo.hyperparams.buffer_size=3000", "algo.hyperparams.learning_starts=200", "experiment.device=cpu"]
VARIANTS: Dict[str, List[str]] = {
    "64px gray stack4 (default)": [],
    "48px gray stack4": ["env.resize=48"],
    "96px gray stack4 (native size)": ["env.resize=96"],
    "64px RGB stack4": ["env.grayscale=false"],
    "64px gray stack1": ["env.frame_stack=1"],
    "64px gray stack4, action_repeat=4": ["env.action_repeat=4"],
    "64px gray stack4, train_freq=4": ["algo.hyperparams.train_freq=4"],
}


def bench_env(cfg_overrides: List[str], steps: int, n_envs: int = 1, vec_type: str = "dummy") -> float:
    cfg = apply_overrides(load_config(ROOT / "configs" / "sac.yaml"),
                          [*BASE, f"env.n_envs={n_envs}", f"env.vec_type={vec_type}", *cfg_overrides])
    env = make_env(cfg, "train")
    try:
        env.reset()
        t0 = time.perf_counter()
        for _ in range(steps // n_envs):
            env.step(np.stack([env.action_space.sample() for _ in range(n_envs)]))
        return steps / (time.perf_counter() - t0)
    finally:
        close_env(env)


def bench_train(overrides: List[str], steps: int) -> Dict[str, Any]:
    cfg = apply_overrides(load_config(ROOT / "configs" / "sac.yaml"), [*BASE, *overrides])
    set_global_seed(0)
    env = make_env(cfg, "train")
    try:
        model = build_model(cfg, env)
        model.learn(cfg.algo.hyperparams["learning_starts"])        # warm-up, not timed
        n0 = model.num_timesteps
        t0 = time.perf_counter()
        model.learn(steps, reset_num_timesteps=False)
        dt = time.perf_counter() - t0
        return {"train_steps_per_s": (model.num_timesteps - n0) / dt, "rss_gb_process_cumulative": process_rss_gb()}
    finally:
        close_env(env)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400, help="timed agent steps per training variant")
    ap.add_argument("--env-steps", type=int, default=600, help="agent steps per env-only measurement")
    ap.add_argument("--out", default=str(ROOT / "results" / "benchmarks"))
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    rows: List[Dict[str, Any]] = []
    print("== Simulator only (random actions, no learning) ==")
    for name, ov in [("1 env, dummy", (1, "dummy")), ("2 envs, dummy", (2, "dummy")), ("2 envs, subproc", (2, "subproc"))]:
        sps = bench_env([], args.env_steps, *ov)
        print(f"  {name:20s} {sps:8.1f} agent steps/s")
        rows.append({"kind": "env_only", "variant": name, "steps_per_s": sps})
    if not args.skip_train:
        print("== SAC training (CPU, includes gradient updates) ==")
        for name, ov in VARIANTS.items():
            r = bench_train(ov, args.steps)
            print(f"  {name:36s} {r['train_steps_per_s']:7.1f} steps/s   RSS {r['rss_gb_process_cumulative']:.2f} GiB (cumulative in process)")
            rows.append({"kind": "sac_train", "variant": name, "steps_per_s": r["train_steps_per_s"], "rss_gb_process_cumulative": r["rss_gb_process_cumulative"]})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "benchmark.csv", index=False)
    write_json({"system": collect_system_metadata(), "rows": rows, "timed_steps": args.steps}, out / "benchmark.json")
    print(f"Wrote {out / 'benchmark.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
