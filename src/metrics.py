"""Episode metrics, summaries, CSV/JSON export and plots."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .utils import write_json

END_REASONS = ("lap_complete", "out_of_bounds", "time_limit", "stuck_truncated", "unknown")


@dataclass
class EpisodeResult:
    episode: int
    seed: Optional[int]
    reward: float             # native (unshaped) return
    length: int               # agent steps (each = action_repeat simulator frames)
    coverage: float           # visited tiles / total tiles at episode end
    lap_finished: bool
    end_reason: str           # one of END_REASONS
    wall_time_s: float
    mean_latency_ms: float    # mean model.predict() time per step


def _lat_stats(lat_ms: Sequence[float]) -> Dict[str, float]:
    if len(lat_ms) == 0:
        return {"mean": float("nan"), "p50": float("nan"), "p95": float("nan"), "max": float("nan")}
    a = np.asarray(lat_ms, dtype=float)
    return {"mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)), "max": float(a.max())}


def summarize(results: Sequence[EpisodeResult], latencies_ms: Sequence[float] = (),
              total_runtime_s: Optional[float] = None) -> Dict[str, Any]:
    """Aggregate episode results into one flat, JSON-friendly dict."""
    if not results:
        raise ValueError("Cannot summarize an empty list of episodes.")
    r = np.array([x.reward for x in results], dtype=float)
    ln = np.array([x.length for x in results], dtype=float)
    cov = np.array([x.coverage for x in results], dtype=float)
    reasons = Counter(x.end_reason for x in results)
    n = len(results)
    return {
        "n_episodes": n,
        "reward_mean": float(r.mean()), "reward_std": float(r.std()), "reward_median": float(np.median(r)),
        "reward_min": float(r.min()), "reward_max": float(r.max()),
        "length_mean": float(ln.mean()), "length_median": float(np.median(ln)),
        "coverage_mean": float(cov.mean()), "coverage_median": float(np.median(cov)),
        "lap_completion_rate": float(np.mean([x.lap_finished for x in results])),
        "termination_counts": {k: int(reasons.get(k, 0)) for k in END_REASONS if reasons.get(k, 0)},
        "failure_rate": float(1.0 - np.mean([x.lap_finished for x in results])),
        "inference_latency_ms": _lat_stats(latencies_ms),
        "eval_runtime_s": float(total_runtime_s if total_runtime_s is not None
                                else sum(x.wall_time_s for x in results)),
    }


def results_to_dataframe(results: Iterable[EpisodeResult]) -> pd.DataFrame:
    return pd.DataFrame([asdict(x) for x in results])


def save_results(results: Sequence[EpisodeResult], summary: Dict[str, Any], out_dir: Path,
                 prefix: str = "eval") -> Dict[str, Path]:
    """Write ``<prefix>_episodes.csv`` and ``<prefix>_summary.json`` into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{prefix}_episodes.csv"
    json_path = out_dir / f"{prefix}_summary.json"
    results_to_dataframe(results).to_csv(csv_path, index=False)
    write_json(summary, json_path)
    return {"csv": csv_path, "json": json_path}


# --------------------------------------------------------------------------- #
# Training logs
# --------------------------------------------------------------------------- #
def read_monitor_logs(directory: Path, mode: str = "train") -> pd.DataFrame:
    """Merge SB3 Monitor CSVs (``<mode>_env*.monitor.csv``) into one frame ordered by wall time.

    Columns: r (native reward), l (agent steps), t (seconds since env start), coverage,
    lap_finished, env, timesteps (cumulative agent steps over all envs, approximate ordering).
    """
    # A resumed run writes into ``resume_*`` sub-folders (Monitor would otherwise overwrite the old files);
    # parts are concatenated in the order they were written and step counts continue across them.
    groups: Dict[Path, List[Path]] = {}
    for path in Path(directory).rglob(f"{mode}_env*.monitor.csv"):
        groups.setdefault(path.parent, []).append(path)
    parts = []
    offset = 0
    for folder in sorted(groups, key=lambda d: max(p.stat().st_mtime for p in groups[d])):
        frames = []
        for i, path in enumerate(sorted(groups[folder])):
            try:
                df = pd.read_csv(path, skiprows=1)
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                continue
            if not df.empty:
                df["env"] = i
                frames.append(df)
        if not frames:
            continue
        part = pd.concat(frames, ignore_index=True).sort_values("t").reset_index(drop=True)
        part["timesteps"] = part["l"].cumsum() + offset
        offset = int(part["timesteps"].iloc[-1])
        parts.append(part)
    if not parts:
        return pd.DataFrame(columns=["r", "l", "t", "coverage", "lap_finished", "env", "timesteps"])
    return pd.concat(parts, ignore_index=True)


def rolling(y: np.ndarray, window: int) -> np.ndarray:
    window = max(1, min(window, len(y)))
    return pd.Series(y).rolling(window, min_periods=1).mean().to_numpy()


def get_pyplot():
    """matplotlib.pyplot with the non-interactive Agg backend (safe on servers and in Colab)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


_plt = get_pyplot


def plot_training_curves(monitor_df: pd.DataFrame, eval_df: Optional[pd.DataFrame], out_path: Path,
                         window: int = 20, title: str = "Training progress") -> Optional[Path]:
    """Reward progression, track coverage, and training stability (rolling std)."""
    if monitor_df.empty:
        return None
    plt = _plt()
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    x, r = monitor_df["timesteps"].to_numpy(), monitor_df["r"].to_numpy()
    ax[0].plot(x, r, alpha=0.25, color="tab:blue", label="episode")
    ax[0].plot(x, rolling(r, window), color="tab:blue", label=f"rolling mean ({window})")
    if eval_df is not None and not eval_df.empty:
        ax[0].errorbar(eval_df["timesteps"], eval_df["mean_reward"], yerr=eval_df["std_reward"],
                       color="tab:red", marker="o", capsize=3, label="held-out eval (mean±std)")
    ax[0].set(title="Reward progression (native reward)", xlabel="agent steps", ylabel="episode reward")
    ax[0].legend(fontsize=8)
    if "coverage" in monitor_df:
        cov = monitor_df["coverage"].to_numpy() * 100
        ax[1].plot(x, cov, alpha=0.25, color="tab:green")
        ax[1].plot(x, rolling(cov, window), color="tab:green", label="train coverage")
        if eval_df is not None and not eval_df.empty and "mean_coverage" in eval_df:
            ax[1].plot(eval_df["timesteps"], eval_df["mean_coverage"] * 100, color="tab:red", marker="o",
                       label="eval coverage")
        ax[1].set(title="Track coverage", xlabel="agent steps", ylabel="% of tiles visited", ylim=(0, 105))
        ax[1].legend(fontsize=8)
    std = pd.Series(r).rolling(max(window, 2), min_periods=2).std().to_numpy()
    ax[2].plot(x, std, color="tab:purple")
    ax[2].set(title=f"Stability: rolling std of reward ({window})", xlabel="agent steps", ylabel="std")
    fig.suptitle(title)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_eval_comparison(summaries: Dict[str, Dict[str, Any]], episodes: Dict[str, pd.DataFrame],
                         out_path: Path, title: str = "Evaluation") -> Path:
    """Box/strip plot of per-episode rewards plus coverage bars for several policies."""
    plt = _plt()
    names = list(summaries)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    data = [episodes[n]["reward"].to_numpy() for n in names]
    ax[0].boxplot(data, showmeans=True)
    ax[0].set_xticks(range(1, len(names) + 1))
    ax[0].set_xticklabels(names, rotation=15, ha="right")  # version-independent (labels= was renamed)
    for i, d in enumerate(data, start=1):
        ax[0].scatter(np.full(len(d), i) + np.random.default_rng(0).uniform(-0.08, 0.08, len(d)), d,
                      alpha=0.6, s=14, color="k")
    ax[0].set(title="Episode reward (same tracks for every policy)", ylabel="native reward")
    cov = [summaries[n]["coverage_mean"] * 100 for n in names]
    ax[1].bar(names, cov, color="tab:green")
    ax[1].set(title="Mean track coverage", ylabel="% tiles visited", ylim=(0, 105))
    fig.suptitle(title)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_interactive_training(monitor_df: pd.DataFrame, eval_df: Optional[pd.DataFrame],
                              out_path: Path) -> Optional[Path]:
    """Optional Plotly HTML version of the reward curve (skipped silently only if plotly is absent)."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
    if monitor_df.empty:
        return None
    fig = go.Figure()
    fig.add_scatter(x=monitor_df["timesteps"], y=monitor_df["r"], mode="markers", name="episode",
                    marker=dict(size=4, opacity=0.35))
    fig.add_scatter(x=monitor_df["timesteps"], y=rolling(monitor_df["r"].to_numpy(), 20), name="rolling mean (20)")
    if eval_df is not None and not eval_df.empty:
        fig.add_scatter(x=eval_df["timesteps"], y=eval_df["mean_reward"], name="eval mean", mode="lines+markers",
                        error_y=dict(type="data", array=eval_df["std_reward"]))
    fig.update_layout(title="Training reward", xaxis_title="agent steps", yaxis_title="native reward")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    return out_path


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
