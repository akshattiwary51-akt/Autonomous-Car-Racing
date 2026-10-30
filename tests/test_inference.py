"""Agent construction, smoke training, checkpoint round-trip, evaluation metrics, inference and video."""

import json

import numpy as np
import pytest
import torch

from src.agent import build_model, check_replay_buffer_memory, load_model
from src.config import apply_overrides
from src.environment import close_env, make_env
from src.evaluate import evaluate_episodes, model_policy, random_policy, run_evaluation
from src.feature_extractor import CustomCNN
from src.inference import run_inference
from src.metrics import summarize
from src.utils import find_latest_checkpoint, resolve_model_path


def test_agent_initialises_and_predicts_valid_actions(smoke_cfg):
    env = make_env(smoke_cfg, "train")
    try:
        model = build_model(smoke_cfg, env)
        assert isinstance(model.policy.actor.features_extractor, CustomCNN)
        action, _ = model.predict(env.reset(), deterministic=True)
        assert action.shape == (1, 3) and np.all(np.abs(action) <= 1.0)
    finally:
        close_env(env)


def test_cnn_rejects_too_small_input():
    import gymnasium as gym

    space = gym.spaces.Box(0, 255, (2, 6, 6), dtype=np.uint8)  # 6x6 input cannot fit an 8x8 kernel
    with pytest.raises(ValueError, match="too large"):
        CustomCNN(space, 32, (8,), (8,), (4,))


def test_replay_buffer_memory_guard(smoke_cfg):
    cfg = apply_overrides(smoke_cfg, ["algo.hyperparams.buffer_size=2000000000"])
    with pytest.raises(RuntimeError, match="Lower algo.hyperparams.buffer_size"):
        check_replay_buffer_memory(cfg)


def test_smoke_training_checkpoint_roundtrip_and_inference(tmp_cfg):
    from src.train import train

    cfg = apply_overrides(tmp_cfg, ["training.total_timesteps=160", "training.checkpoint_freq=80",
                                    "evaluation.eval_freq=80", "evaluation.final_n_episodes=1",
                                    "algo.hyperparams.learning_starts=60"])
    summary = train(cfg, final_eval=False)
    assert summary["status"] == "completed" and summary["timesteps_reached"] >= 160

    from src.utils import run_dirs

    d = run_dirs(cfg.experiment.name)
    ckpt = find_latest_checkpoint(d["checkpoints"])
    assert ckpt.name.startswith("ckpt_") and (d["models"] / "final_model.zip").is_file()
    assert (d["models"] / "best_model.zip").is_file()
    assert (d["results"] / "eval_history.csv").is_file() and (d["results"] / "training_summary.json").is_file()
    assert (d["models"] / "config.yaml").is_file() and (d["checkpoints"] / "config.yaml").is_file()

    # reload -> identical deterministic actions to the in-memory trained model
    env = make_env(cfg, "eval", eval_seeds=[1])
    try:
        reloaded = load_model(d["models"] / "final_model.zip", cfg, env=env)
        again = load_model(d["models"] / "final_model.zip", cfg, env=env)
        obs = env.reset()
        a1, _ = reloaded.predict(obs, deterministic=True)
        a2, _ = again.predict(obs, deterministic=True)
        np.testing.assert_allclose(a1, a2)
        assert reloaded.num_timesteps >= 160
    finally:
        close_env(env)

    # run_evaluation produces valid, exported metrics (+ baselines)
    out = d["results"] / "evalcheck"
    ev = run_evaluation(cfg, reloaded, n_episodes=1, out_dir=out, compare_baselines=True, prefix="t")
    assert set(ev["summaries"]) == {"trained_deterministic", "untrained_network", "random_uniform"}
    s = ev["summaries"]["trained_deterministic"]
    assert s["n_episodes"] == 1 and 0 <= s["coverage_mean"] <= 1 and s["inference_latency_ms"]["mean"] > 0
    assert (out / "t_trained_deterministic_summary.json").is_file() and (out / "t_comparison.png").is_file()
    assert json.loads((out / "t_random_uniform_summary.json").read_text())["n_episodes"] == 1

    # inference with the checkpoint (headless, frames collected)
    frames = []
    res = run_inference(cfg, reloaded, episodes=1, collect_frames=True, max_steps=5,
                        frame_callback=lambda f, *a: frames.append(f))
    assert len(res) == 1 and res[0]["steps"] == 5 and len(frames) == 5 and frames[0].ndim == 3

    # resume continues the step counter instead of restarting
    import pandas as pd

    from src.metrics import read_monitor_logs

    before = len(pd.read_csv(d["results"] / "eval_history.csv"))
    n_before = len(read_monitor_logs(d["logs"] / "monitor"))
    cfg2 = apply_overrides(cfg, ["training.total_timesteps=200"])
    s2 = train(cfg2, resume_from=resolve_model_path(d["checkpoints"]), final_eval=False)
    assert s2["timesteps_reached"] >= 200 > 160
    # history is appended (not overwritten) and monitor CSVs of the first part are kept
    assert len(pd.read_csv(d["results"] / "eval_history.csv")) >= before
    mon = read_monitor_logs(d["logs"] / "monitor")
    assert len(mon) >= n_before and mon["timesteps"].is_monotonic_increasing
    assert any(p.name.startswith("resume_") for p in (d["logs"] / "monitor").iterdir())


def test_missing_checkpoint_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        resolve_model_path(tmp_path / "missing.zip")
    with pytest.raises(FileNotFoundError, match="No '\\*_steps.zip' checkpoints"):
        find_latest_checkpoint(tmp_path)


def test_evaluate_episodes_is_deterministic_and_complete(smoke_cfg):
    cfg = apply_overrides(smoke_cfg, ["env.action_repeat=4"])  # 250 steps/episode keeps the test short
    seeds = [21, 22]
    outs = []
    for _ in range(2):
        env = make_env(cfg, "eval", eval_seeds=seeds)
        try:
            outs.append(evaluate_episodes(random_policy(0), env, 2))
        finally:
            close_env(env)
    (r1, l1), (r2, _) = outs
    assert [x.reward for x in r1] == [x.reward for x in r2]
    assert [x.seed for x in r1] == seeds and all(x.length == 250 for x in r1)
    assert all(x.end_reason in ("time_limit", "out_of_bounds", "lap_complete") for x in r1)
    s = summarize(r1, l1)
    assert s["reward_std"] >= 0 and s["reward_min"] <= s["reward_median"] <= s["reward_max"]
    assert sum(s["termination_counts"].values()) == 2


def test_summarize_rejects_empty():
    with pytest.raises(ValueError):
        summarize([])


def test_video_recording(tmp_cfg, tmp_path):
    from src.record_video import record_episodes

    path, res = record_episodes(tmp_cfg, None, str(tmp_path / "v.mp4"), episodes=1, max_steps=8)
    assert path.exists() and path.stat().st_size > 0 and res[0]["steps"] == 8


def test_window_mode_without_display_gives_clear_error(smoke_cfg, monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    import sys

    if not sys.platform.startswith("linux"):
        pytest.skip("display check only applies on Linux")
    with pytest.raises(RuntimeError, match="No display"):
        run_inference(smoke_cfg, None, window=True)


def test_feature_norm_option():
    import gymnasium as gym

    space = gym.spaces.Box(0, 255, (4, 64, 64), dtype=np.uint8)
    net = CustomCNN(space, 32, norm="layernorm")
    out = net(torch.rand(3, 4, 64, 64))
    assert out.shape == (3, 32) and out.abs().max() <= 1.0  # Tanh-bounded
    with pytest.raises(ValueError, match="norm must be"):
        CustomCNN(space, 32, norm="batchnorm")


def test_random_shift_augmentation_only_in_training_mode():
    import gymnasium as gym

    space = gym.spaces.Box(0, 255, (4, 64, 64), dtype=np.uint8)
    net = CustomCNN(space, 32, augment_pad=4)
    x = torch.rand(8, 4, 64, 64)
    shifted = net._random_shift(x)
    assert shifted.shape == x.shape and not torch.equal(shifted, x)
    net.eval()
    np.testing.assert_allclose(net(x).detach().numpy(), net(x).detach().numpy())  # deterministic at inference
    net.train()
    assert not torch.allclose(net(x), net(x))  # stochastic while training
