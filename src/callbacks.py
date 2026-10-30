"""Training callbacks: held-out evaluation with best-model saving, and extra training metrics."""

from __future__ import annotations

import csv
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from .config import Config
from .environment import EVAL_SEED_OFFSET, close_env, make_env
from .evaluate import evaluate_episodes, model_policy
from .metrics import summarize
from .utils import get_logger, process_rss_gb

EVAL_HISTORY_FIELDS = ["timesteps", "mean_reward", "std_reward", "median_reward", "mean_length",
                       "mean_coverage", "lap_completion_rate", "eval_time_s", "is_best"]


class HeldOutEvalCallback(BaseCallback):
    """Periodically evaluate on a *separate* env with a fixed set of unseen tracks.

    * The eval env is built by ``make_env(mode="eval")``: native reward, no shaping, seeds
      ``experiment.seed + 10000 + i`` that differ from the training stream.
    * The seed cycle is restarted every time, so successive evaluations are comparable and the
      "best model" is chosen on identical tracks.
    * Results go to TensorBoard (``eval/*``) and ``eval_history.csv``.
    """

    def __init__(self, cfg: Config, best_model_dir: Path, history_csv: Path) -> None:
        super().__init__(verbose=0)
        self.cfg = cfg
        self.best_model_dir = Path(best_model_dir)
        self.history_csv = Path(history_csv)
        self.eval_freq = cfg.evaluation.eval_freq
        self.n_episodes = cfg.evaluation.n_eval_episodes
        self.deterministic = cfg.evaluation.deterministic
        self.best_reward = -np.inf
        self.eval_env = None
        self._last_eval = 0

    def _on_training_start(self) -> None:
        self.best_model_dir.mkdir(parents=True, exist_ok=True)
        self.history_csv.parent.mkdir(parents=True, exist_ok=True)
        if not self.history_csv.exists():
            with self.history_csv.open("w", newline="") as f:
                csv.writer(f).writerow(EVAL_HISTORY_FIELDS)
        else:  # resumed run: do not let a worse model overwrite the best one found before the interruption
            with self.history_csv.open(newline="") as f:
                prev = [float(row["mean_reward"]) for row in csv.DictReader(f)]
            if prev:
                self.best_reward = max(prev)
        seeds = [self.cfg.experiment.seed + EVAL_SEED_OFFSET + i for i in range(self.n_episodes)]
        self.eval_env = make_env(self.cfg, "eval", eval_seeds=seeds, n_envs=1)
        self._last_eval = self.num_timesteps  # supports resumed runs

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval >= self.eval_freq:
            self._last_eval = self.num_timesteps
            self._evaluate()
        return True

    def _evaluate(self) -> None:
        t0 = time.perf_counter()
        results, lat = evaluate_episodes(model_policy(self.model, self.deterministic), self.eval_env,
                                         self.n_episodes)
        s = summarize(results, lat)
        elapsed = time.perf_counter() - t0
        improved = s["reward_mean"] > self.best_reward
        if improved:
            self.best_reward = s["reward_mean"]
            self.model.save(self.best_model_dir / "best_model")
        row = [self.num_timesteps, s["reward_mean"], s["reward_std"], s["reward_median"], s["length_mean"],
               s["coverage_mean"], s["lap_completion_rate"], round(elapsed, 2), int(improved)]
        with self.history_csv.open("a", newline="") as f:
            csv.writer(f).writerow(row)
        self.logger.record("eval/mean_reward", s["reward_mean"])
        self.logger.record("eval/std_reward", s["reward_std"])
        self.logger.record("eval/mean_ep_length", s["length_mean"])
        self.logger.record("eval/mean_coverage", s["coverage_mean"])
        self.logger.record("eval/lap_completion_rate", s["lap_completion_rate"])
        self.logger.record("eval/time_s", elapsed)
        self.logger.dump(self.num_timesteps)
        get_logger().info("[eval @ %d] reward %.1f ± %.1f | coverage %.1f%% | laps %.0f%%%s", self.num_timesteps,
                          s["reward_mean"], s["reward_std"], 100 * s["coverage_mean"],
                          100 * s["lap_completion_rate"], "  <- new best" if improved else "")

    def _on_training_end(self) -> None:
        close_env(self.eval_env)
        self.eval_env = None


class TrainingStatsCallback(BaseCallback):
    """Log track coverage of finished training episodes, throughput and process memory."""

    def __init__(self, window: int = 50, every: int = 1000) -> None:
        super().__init__(verbose=0)
        self.coverage = deque(maxlen=window)
        self.every = every
        self._t0 = 0.0
        self._start_steps = 0
        self._last_log = 0

    def _on_training_start(self) -> None:
        self._t0 = time.perf_counter()
        self._start_steps = self._last_log = self.num_timesteps

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info and "coverage" in info["episode"]:
                self.coverage.append(float(info["episode"]["coverage"]))
        if self.coverage:
            self.logger.record("rollout/ep_coverage_mean", float(np.mean(self.coverage)))
        if self.num_timesteps - self._last_log >= self.every:
            self._last_log = self.num_timesteps
            elapsed = max(time.perf_counter() - self._t0, 1e-9)
            self.logger.record("perf/steps_per_second", (self.num_timesteps - self._start_steps) / elapsed)
            rss = process_rss_gb()
            if rss is not None:
                self.logger.record("perf/process_rss_gb", rss)
        return True
