#!/usr/bin/env python
"""Pick the best checkpoint of a run by evaluating every checkpoint on the SAME held-out tracks.

    python scripts/select_checkpoint.py --run ppo_fast --episodes 8 --stochastic

Why: the in-training evaluation uses few episodes and (for PPO) can disagree with the actions the policy was trained
with. This re-scores all ``checkpoints/<run>/ckpt_*_steps.zip`` (+ final_model) on ``--seed-start`` tracks that differ from
the training-time evaluation tracks, writes ``results/<run>/checkpoint_selection.csv`` and copies the winner to
``models/<run>/best_selected.zip``. It never modifies the checkpoints or best_model.zip.
Report the winner's score on a *different* set of seeds (src.evaluate / run_comparison) to avoid selection bias.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.agent import load_model  # noqa: E402
from src.config import load_config  # noqa: E402
from src.environment import close_env, make_env  # noqa: E402
from src.evaluate import evaluate_episodes, model_policy  # noqa: E402
from src.metrics import summarize  # noqa: E402
from src.utils import get_logger, run_dirs, setup_logging  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--seed-start", type=int, default=30000)
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--min-step", type=int, default=0, help="ignore checkpoints below this step count")
    args = ap.parse_args()
    setup_logging()
    d = run_dirs(args.run)
    cfg = load_config(d["models"] / "config.yaml")
    cks = []
    for p in d["checkpoints"].glob("ckpt_*_steps.zip"):
        step = int(re.findall(r"_(\d+)_steps", p.name)[0])
        if step >= args.min_step:
            cks.append((step, p))
    if (d["models"] / "final_model.zip").is_file():
        cks.append((10**12, d["models"] / "final_model.zip"))
    if not cks:
        get_logger().error("No checkpoints found for run %s", args.run)
        return 1
    seeds = [args.seed_start + i for i in range(args.episodes)]
    rows = []
    for step, p in sorted(cks):
        env = make_env(cfg, "eval", eval_seeds=seeds, n_envs=1)
        try:
            model = load_model(p, cfg, env=env)
            res, _ = evaluate_episodes(model_policy(model, not args.stochastic), env, args.episodes)
        finally:
            close_env(env)
        s = summarize(res)
        rows.append({"checkpoint": p.name, "step": step if step < 10**12 else -1, "reward_mean": s["reward_mean"],
                     "reward_std": s["reward_std"], "coverage_mean": s["coverage_mean"], "laps": s["lap_completion_rate"]})
        get_logger().info("%-28s reward %.1f ± %.1f | coverage %.0f%% | laps %.0f%%", p.name, s["reward_mean"],
                          s["reward_std"], 100 * s["coverage_mean"], 100 * s["lap_completion_rate"])
    df = pd.DataFrame(rows)
    out = d["results"] / "checkpoint_selection.csv"
    df.to_csv(out, index=False)
    best = df.loc[df.reward_mean.idxmax()]
    src = d["checkpoints"] / best.checkpoint if (d["checkpoints"] / best.checkpoint).is_file() else d["models"] / best.checkpoint
    shutil.copy2(src, d["models"] / "best_selected.zip")
    get_logger().info("Best: %s -> %s (table: %s)", best.checkpoint, d["models"] / "best_selected.zip", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
