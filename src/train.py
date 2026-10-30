"""Training entry point.

Examples::

    python -m src.train --config configs/sac.yaml
    python -m src.train --config configs/sac.yaml --set training.total_timesteps=50000 algo.hyperparams.buffer_size=20000
    python -m src.train --resume checkpoints/sac_carracing          # latest checkpoint of that run
    python -m src.train --config configs/ppo.yaml --run-name ppo_seed1 --seed 1
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

from .agent import build_model, load_model
from .callbacks import HeldOutEvalCallback, TrainingStatsCallback
from .config import Config, ConfigError, apply_overrides, load_config, save_config
from .environment import check_env_spaces, close_env, make_env
from .evaluate import model_policy, run_evaluation
from .metrics import plot_interactive_training, plot_training_curves, read_monitor_logs
from .utils import (collect_system_metadata, find_config_for_model, find_latest_checkpoint, get_logger,
                    process_rss_gb, resolve_model_path, run_dirs, set_global_seed, setup_logging, write_json)


def _install_sigterm_handler() -> None:
    """Turn SIGTERM (e.g. job scheduler / Colab stop) into KeyboardInterrupt so we save before exit."""
    def handler(signum, frame):  # noqa: ARG001
        raise KeyboardInterrupt("SIGTERM received")

    try:
        signal.signal(signal.SIGTERM, handler)
    except (ValueError, OSError):  # not in main thread / unsupported platform
        pass


def _prepare_run_dirs(cfg: Config, resume: bool, force: bool) -> Dict[str, Path]:
    dirs = run_dirs(cfg.experiment.name)
    if not resume and not force:
        existing = [d for k in ("models", "checkpoints") for d in [dirs[k]] if d.exists() and any(d.iterdir())]
        if existing:
            raise FileExistsError(
                f"Run '{cfg.experiment.name}' already has outputs in {existing[0]}. Use --run-name to start a new "
                "run, --resume to continue it, or --force to write into the same folders."
            )
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def train(cfg: Config, *, resume_from: Optional[Path] = None, force: bool = False,
          final_eval: bool = True) -> Dict[str, Any]:
    """Run one full training job and return a summary dict (also written to results/<run>/)."""
    log = get_logger()
    dirs = _prepare_run_dirs(cfg, resume=resume_from is not None, force=force)
    setup_logging(dirs["logs"] / "train.log")
    set_global_seed(cfg.experiment.seed)
    if cfg.experiment.torch_threads:
        import torch

        torch.set_num_threads(cfg.experiment.torch_threads)

    for folder in (dirs["models"], dirs["checkpoints"], dirs["results"]):
        save_config(cfg, folder / "config.yaml")
    meta = collect_system_metadata()
    write_json({**meta, "resumed_from": str(resume_from) if resume_from else None}, dirs["results"] / "metadata.json")
    log.info("Run '%s' | algo=%s | seed=%d | python %s | torch %s | gymnasium %s | sb3 %s",
             cfg.experiment.name, cfg.algo.name, cfg.experiment.seed, meta["python"], meta["torch"],
             meta["gymnasium"], meta["stable_baselines3"])

    monitor_dir = dirs["logs"] / "monitor"
    if resume_from is not None:
        monitor_dir = monitor_dir / f"resume_{int(time.time())}"  # keep the first part's CSVs intact
    train_env = make_env(cfg, "train", log_dir=monitor_dir)
    status = "completed"
    t_start = time.perf_counter()
    model = None
    best = dirs["models"] / "best_model.zip"
    try:
        log.info("Environment check: %s", check_env_spaces(train_env, cfg))
        if resume_from is not None:
            model = load_model(resume_from, cfg, env=train_env)
            log.info("Resumed %s from %s at %d steps", cfg.algo.name.upper(), resume_from, model.num_timesteps)
            if cfg.algo.name == "sac":
                # CheckpointCallback names buffers "<prefix>_replay_buffer_<steps>_steps.pkl"
                prefix = resume_from.stem.rsplit("_", 2)[0]
                buffer_file = resume_from.parent / f"{prefix}_replay_buffer_{model.num_timesteps}_steps.pkl"
                if buffer_file.is_file():
                    model.load_replay_buffer(buffer_file)
                    log.info("Restored replay buffer from %s (%d transitions)", buffer_file, model.replay_buffer.size())
                else:
                    model.learning_starts = model.num_timesteps + cfg.training.resume_warmup_steps
                    log.warning("No replay buffer saved with this checkpoint: the buffer restarts empty; %d random "
                                "warm-up steps are collected before gradient updates resume.",
                                cfg.training.resume_warmup_steps)
        else:
            model = build_model(cfg, train_env, tensorboard_log=dirs["logs"] / "tb")
        if resume_from is not None:
            model.tensorboard_log = str(dirs["logs"] / "tb")

        n_envs = train_env.num_envs
        callbacks = CallbackList([
            CheckpointCallback(save_freq=max(cfg.training.checkpoint_freq // n_envs, 1), save_path=str(dirs["checkpoints"]),
                               name_prefix="ckpt", save_replay_buffer=cfg.training.save_replay_buffer and cfg.algo.name == "sac"),
            HeldOutEvalCallback(cfg, best_model_dir=dirs["models"], history_csv=dirs["results"] / "eval_history.csv"),
            TrainingStatsCallback(),
        ])
        start_steps = model.num_timesteps
        remaining = cfg.training.total_timesteps - start_steps
        if remaining <= 0:
            log.info("Nothing to do: model already has %d >= total_timesteps=%d.", model.num_timesteps,
                     cfg.training.total_timesteps)
        else:
            log.info("Training for %d agent steps (%d simulator frames) on %s. TensorBoard: tensorboard --logdir %s",
                     remaining, remaining * cfg.env.action_repeat, model.device, dirs["logs"] / "tb")
            try:
                model.learn(total_timesteps=remaining, callback=callbacks, log_interval=cfg.training.log_interval,
                            tb_log_name="run", reset_num_timesteps=resume_from is None,
                            progress_bar=cfg.training.progress_bar)
            except KeyboardInterrupt:
                status = "interrupted"
                path = dirs["checkpoints"] / f"interrupted_{model.num_timesteps}_steps"
                model.save(path)
                log.warning("Interrupted. Saved %s.zip — resume with: python -m src.train --resume %s",
                            path, dirs["checkpoints"])
        model.save(dirs["models"] / "final_model")
        wall = time.perf_counter() - t_start
        steps_done = model.num_timesteps - start_steps
        summary: Dict[str, Any] = {
            "run": cfg.experiment.name, "status": status, "algo": cfg.algo.name, "seed": cfg.experiment.seed,
            "timesteps_reached": int(model.num_timesteps), "wall_time_s": round(wall, 1),
            "steps_per_second_overall": round(steps_done / max(wall, 1e-9), 2), "device": str(model.device),
            "process_rss_gb": process_rss_gb(), "final_model": str(dirs["models"] / "final_model.zip"),
        }
        summary["best_model"] = str(best) if best.is_file() else None
    finally:
        close_env(train_env)

    _make_plots(dirs)
    if final_eval and cfg.evaluation.final_n_episodes > 0 and model is not None:
        which = "best" if best.is_file() else "final"
        eval_model = load_model(best, cfg) if best.is_file() else model
        log.info("Final held-out evaluation of the %s model (%d episodes, deterministic) + baselines",
                 which, cfg.evaluation.final_n_episodes)
        ev = run_evaluation(cfg, eval_model, n_episodes=cfg.evaluation.final_n_episodes, deterministic=True,
                            out_dir=dirs["results"], compare_baselines=True, prefix="final")
        summary["final_evaluation"] = {"model": which, **ev}
    write_json(summary, dirs["results"] / "training_summary.json")
    _print_summary(summary)
    return summary


def _make_plots(dirs: Dict[str, Path]) -> None:
    import pandas as pd

    mon = read_monitor_logs(dirs["logs"] / "monitor", "train")
    hist_path = dirs["results"] / "eval_history.csv"
    hist = pd.read_csv(hist_path) if hist_path.is_file() else None
    plot_training_curves(mon, hist, dirs["results"] / "training_curves.png")
    plot_interactive_training(mon, hist, dirs["results"] / "training_curves.html")


def _print_summary(s: Dict[str, Any]) -> None:
    log = get_logger()
    log.info("=" * 70)
    log.info("TRAINING SUMMARY  run=%s  status=%s", s["run"], s["status"])
    log.info("  steps reached : %d   wall time: %.1f min   throughput: %.1f steps/s   device: %s",
             s["timesteps_reached"], s["wall_time_s"] / 60, s["steps_per_second_overall"], s["device"])
    if s.get("final_evaluation"):
        for name, v in s["final_evaluation"]["summaries"].items():
            log.info("  %-20s reward %.1f ± %.1f | coverage %.1f%% | lap completion %.0f%%", name, v["reward_mean"],
                     v["reward_std"], 100 * v["coverage_mean"], 100 * v["lap_completion_rate"])
    log.info("  best model    : %s", s.get("best_model"))
    log.info("=" * 70)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train a CarRacing agent (SAC by default).")
    ap.add_argument("--config", help="YAML/JSON config (default: configs/sac.yaml, or the one saved with --resume)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values, e.g. training.total_timesteps=50000")
    ap.add_argument("--run-name", help="experiment.name override (output folder name)")
    ap.add_argument("--seed", type=int, help="experiment.seed override")
    ap.add_argument("--timesteps", type=int, help="training.total_timesteps override")
    ap.add_argument("--device", help="experiment.device override (auto|cpu|cuda)")
    ap.add_argument("--resume", help="checkpoint .zip or checkpoint directory (latest) to continue from")
    ap.add_argument("--force", action="store_true", help="allow writing into an existing run folder")
    ap.add_argument("--no-final-eval", action="store_true")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging()
    _install_sigterm_handler()
    try:
        resume_path = resolve_model_path(args.resume) if args.resume else None
        if args.config:
            cfg_path = Path(args.config)
        elif resume_path is not None:
            cfg_path = find_config_for_model(resume_path)
        else:
            cfg_path = Path(__file__).resolve().parents[1] / "configs" / "sac.yaml"
        cfg = load_config(cfg_path)
        overrides: List[str] = list(args.set)
        if args.run_name:
            overrides.append(f"experiment.name={args.run_name}")
        if args.seed is not None:
            overrides.append(f"experiment.seed={args.seed}")
        if args.timesteps is not None:
            overrides.append(f"training.total_timesteps={args.timesteps}")
        if args.device:
            overrides.append(f"experiment.device={args.device}")
        if overrides:
            cfg = apply_overrides(cfg, overrides)
        train(cfg, resume_from=resume_path, force=args.force, final_eval=not args.no_final_eval)
    except (ConfigError, FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        get_logger().error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
