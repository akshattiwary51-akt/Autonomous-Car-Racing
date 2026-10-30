#!/usr/bin/env python
"""Controlled SAC vs PPO comparison (same env pipeline, CNN, seeds, step budget, held-out tracks).

    python scripts/compare_algorithms.py --timesteps 100000 --seeds 0 1 2
    python scripts/compare_algorithms.py --report-only          # re-aggregate existing runs

Runs ``configs/sac.yaml`` and ``configs/ppo.yaml`` for every seed (sequentially), then writes
results/comparison/{comparison_table.csv, comparison_curves.png, comparison_report.md}.
The report only states numbers that were measured; with few seeds it says so and makes no superiority claim.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import apply_overrides, load_config  # noqa: E402
from src.metrics import _plt  # noqa: E402
from src.train import train  # noqa: E402
from src.utils import get_logger, load_json_if_exists, run_dirs, setup_logging, write_json  # noqa: E402

CONFIGS = {"sac": ROOT / "configs" / "sac.yaml", "ppo": ROOT / "configs" / "ppo.yaml"}


def run_name(algo: str, seed: int) -> str:
    return f"cmp_{algo}_seed{seed}"


def aggregate(seeds: List[int], out: Path, timesteps: int) -> None:
    rows, curves = [], {}
    for algo in CONFIGS:
        for seed in seeds:
            d = run_dirs(run_name(algo, seed))["results"]
            summ = load_json_if_exists(d / "training_summary.json")
            hist = d / "eval_history.csv"
            if not summ or not hist.is_file():
                get_logger().warning("Missing results for %s seed %d - skipped.", algo, seed)
                continue
            fe = summ.get("final_evaluation", {}).get("summaries", {}).get("trained_deterministic")
            rows.append({
                "algo": algo, "seed": seed, "timesteps": summ["timesteps_reached"], "wall_time_min": summ["wall_time_s"] / 60,
                "steps_per_s": summ["steps_per_second_overall"],
                "final_reward_mean": fe["reward_mean"] if fe else np.nan, "final_reward_std": fe["reward_std"] if fe else np.nan,
                "final_coverage_mean": fe["coverage_mean"] if fe else np.nan,
                "final_lap_rate": fe["lap_completion_rate"] if fe else np.nan,
            })
            curves.setdefault(algo, []).append(pd.read_csv(hist))
    if not rows:
        raise SystemExit("No finished runs found; run without --report-only first.")
    table = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "comparison_table.csv", index=False)

    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for algo, hists in curves.items():
        for h in hists:
            ax[0].plot(h["timesteps"], h["mean_reward"], alpha=0.25, color="C0" if algo == "sac" else "C1")
        grid = np.unique(np.concatenate([h["timesteps"].to_numpy() for h in hists]))
        interp = np.stack([np.interp(grid, h["timesteps"], h["mean_reward"]) for h in hists])
        ax[0].plot(grid, interp.mean(0), lw=2.2, color="C0" if algo == "sac" else "C1", label=f"{algo.upper()} mean of {len(hists)} seed(s)")
    ax[0].set(title="Held-out evaluation reward during training", xlabel="agent steps", ylabel="mean eval reward")
    ax[0].legend()
    for i, algo in enumerate(CONFIGS):
        sub = table[table.algo == algo]
        ax[1].scatter(np.full(len(sub), i), sub["final_reward_mean"], s=40, color="C0" if algo == "sac" else "C1")
        if len(sub):
            ax[1].hlines(sub["final_reward_mean"].mean(), i - 0.2, i + 0.2, color="k")
    ax[1].set(xticks=[0, 1], xticklabels=[a.upper() for a in CONFIGS], title="Final held-out reward per seed", ylabel="mean reward")
    fig.tight_layout()
    fig.savefig(out / "comparison_curves.png", dpi=130)
    plt.close(fig)

    g = table.groupby("algo")
    lines = [f"# SAC vs PPO — measured comparison ({timesteps} agent steps per run)\n",
             "All numbers below come from the runs in `results/cmp_*`. Same observation pipeline, CNN extractor, seeds, "
             "step budget and held-out evaluation tracks for both algorithms.\n",
             "| algo | seeds | final reward (mean ± std over seeds) | coverage | lap completion | steps/s | wall time (min) |",
             "|---|---|---|---|---|---|---|"]
    for algo, sub in g:
        lines.append(f"| {algo.upper()} | {len(sub)} | {sub.final_reward_mean.mean():.1f} ± {sub.final_reward_mean.std(ddof=0):.1f} | "
                     f"{100 * sub.final_coverage_mean.mean():.1f}% | {100 * sub.final_lap_rate.mean():.0f}% | "
                     f"{sub.steps_per_s.mean():.1f} | {sub.wall_time_min.mean():.1f} |")
    n_min = int(g.size().min())
    lines += ["", "## How to read this", "",
              f"* Seeds per algorithm: {n_min}. " + ("With fewer than 3 seeds (or short budgets) differences are **not statistically meaningful**; "
                                                    "no algorithm is declared better." if n_min < 3 or timesteps < 200_000 else
                                                    "Compare the per-seed spread before drawing conclusions."),
              "* Sample efficiency: compare the curves at equal agent steps (`comparison_curves.png`). SAC is off-policy and reuses replay data; "
              "PPO is on-policy and discards each rollout after its update epochs.",
              "* Compute: `steps/s` is end-to-end throughput including evaluation; SAC performs gradient updates every step, PPO in batches.",
              "* Stability: look at the per-seed spread and the reward curve noise rather than only the final number."]
    (out / "comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=100_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--algos", nargs="+", default=["sac", "ppo"], choices=list(CONFIGS))
    ap.add_argument("--set", nargs="*", default=[], help="extra overrides applied to both configs")
    ap.add_argument("--out", default=str(ROOT / "results" / "comparison"))
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    setup_logging()
    if not args.report_only:
        for seed in args.seeds:
            for algo in args.algos:
                cfg = apply_overrides(load_config(CONFIGS[algo]), [
                    f"experiment.name={run_name(algo, seed)}", f"experiment.seed={seed}",
                    f"training.total_timesteps={args.timesteps}", *args.set])
                get_logger().info("=== %s seed %d ===", algo.upper(), seed)
                train(cfg, force=True)
    aggregate(args.seeds, Path(args.out), args.timesteps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
