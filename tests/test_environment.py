import numpy as np
import pytest

from src.config import apply_overrides
from src.environment import ActionRescaleWrapper, check_env_spaces, close_env, make_env, observation_shape


def test_create_reset_and_spaces(smoke_cfg):
    env = make_env(smoke_cfg, "train")
    try:
        info = check_env_spaces(env, smoke_cfg)
        obs = env.reset()
        assert obs.shape == (1, *observation_shape(smoke_cfg)) and obs.dtype == np.uint8
        assert info["action_shape"] == (3,)
        assert env.action_space.low.min() == -1 and env.action_space.high.max() == 1
    finally:
        close_env(env)


@pytest.mark.parametrize("gray, size, stack", [(True, 64, 4), (False, 48, 2), (True, None, 1)])
def test_preprocessing_variants(smoke_cfg, gray, size, stack):
    cfg = apply_overrides(smoke_cfg, [f"env.grayscale={str(gray).lower()}", f"env.resize={size if size else 'null'}",
                                      f"env.frame_stack={stack}"])
    env = make_env(cfg, "eval", eval_seeds=[1])
    try:
        obs = env.reset()
        assert obs.shape[1:] == observation_shape(cfg)
    finally:
        close_env(env)


def test_actions_stay_valid_and_mapping(smoke_cfg):
    env = make_env(smoke_cfg, "train")
    try:
        env.reset()
        for _ in range(20):
            a = np.stack([env.action_space.sample()])
            assert env.action_space.contains(a[0])
            env.step(a)
        # wrapper mapping: zero action -> straight, half gas, no brake; extremes clip into the native box
        w = ActionRescaleWrapper(__import__("gymnasium").make("CarRacing-v3"))
        np.testing.assert_allclose(w.action(np.zeros(3)), [0.0, 0.5, 0.0])
        np.testing.assert_allclose(w.action(np.array([5.0, 5.0, 5.0])), [1.0, 1.0, 1.0])
        np.testing.assert_allclose(w.action(np.array([-5.0, -5.0, -5.0])), [-1.0, 0.0, 0.0])
        w.close()
    finally:
        close_env(env)


def test_episode_ends_and_reports_flags(smoke_cfg):
    """A stationary car runs into the 1000-frame limit: truncated=True, terminated=False, auto-reset happens."""
    env = make_env(smoke_cfg, "eval", eval_seeds=[3, 4])
    try:
        env.reset()
        stop = np.array([[0.0, -1.0, 1.0]], dtype=np.float32)
        for t in range(1, 2000):
            _, _, dones, infos = env.step(stop)
            if dones[0]:
                break
        assert dones[0], "episode never ended"
        info = infos[0]
        assert info["truncated"] and not info["terminated"]
        assert t == 1000 // smoke_cfg.env.action_repeat
        assert "terminal_observation" in info and 0 <= info["coverage"] <= 1
        assert info["episode"]["l"] == t
    finally:
        close_env(env)


def test_eval_seed_cycle_gives_identical_tracks(smoke_cfg):
    env = make_env(smoke_cfg, "eval", eval_seeds=[11, 12])

    def track() -> np.ndarray:  # the first rendered frame is black for every track, so compare geometry
        return np.array(env.venv.envs[0].unwrapped.track)[:, 2:4]

    try:
        env.env_method("restart_seed_cycle")
        env.reset()
        a = track()
        env.env_method("restart_seed_cycle")
        env.reset()
        np.testing.assert_array_equal(a, track())
        env.reset()  # next seed -> different track
        assert a.shape != track().shape or not np.array_equal(a, track())
    finally:
        close_env(env)


def test_training_seed_is_reproducible(smoke_cfg):
    obs = []
    for _ in range(2):
        env = make_env(smoke_cfg, "train", seed=7)
        try:
            obs.append(env.reset())
        finally:
            close_env(env)
    np.testing.assert_array_equal(obs[0], obs[1])


def test_reward_shaping_keeps_native_reward_and_truncates_when_stuck(smoke_cfg):
    cfg = apply_overrides(smoke_cfg, ["env.reward_shaping.enabled=true", "env.reward_shaping.stuck_patience=30",
                                      "env.reward_shaping.stuck_penalty=5", "env.reward_shaping.steer_change_penalty=0.1"])
    env = make_env(cfg, "train", log_dir=None)
    try:
        env.reset()
        stop = np.array([[0.0, -1.0, 1.0]], dtype=np.float32)
        for _ in range(100):
            _, r, dones, infos = env.step(stop)
            if dones[0]:
                break
        assert dones[0] and infos[0].get("stuck_truncated")
        assert "native_reward" in infos[0]
    finally:
        close_env(env)
    # the eval env never shapes rewards
    ev = make_env(cfg, "eval", eval_seeds=[0])
    try:
        ev.reset()
        _, _, _, infos = ev.step(np.array([[0.0, 0.0, 0.0]], dtype=np.float32))
        assert "native_reward" not in infos[0]  # RewardShapingWrapper is only applied in train mode
    finally:
        close_env(ev)


def test_invalid_mode_raises(smoke_cfg):
    with pytest.raises(ValueError, match="mode"):
        make_env(smoke_cfg, "bogus")


def test_vectorised_subproc_env(smoke_cfg):
    cfg = apply_overrides(smoke_cfg, ["env.n_envs=2", "env.vec_type=subproc"])
    env = make_env(cfg, "train")
    try:
        obs = env.reset()
        assert obs.shape[0] == 2
        obs, r, d, _ = env.step(np.stack([env.action_space.sample() for _ in range(2)]))
        assert r.shape == (2,)
    finally:
        close_env(env)


def test_reward_scale_only_affects_training_reward(smoke_cfg):
    cfg = apply_overrides(smoke_cfg, ["env.reward_scale=0.1"])
    stop = np.array([[0.0, -1.0, 1.0]], dtype=np.float32)
    outs = {}
    for mode in ("train", "eval"):
        env = make_env(cfg, mode, **({"seed": 5} if mode == "train" else {"eval_seeds": [5]}))
        try:
            env.reset()
            outs[mode] = env.step(stop)[1][0]
        finally:
            close_env(env)
    assert outs["eval"] != 0 and outs["train"] == pytest.approx(outs["eval"] * 0.1, rel=1e-4)


def test_speed_bonus_is_gated_on_progress(smoke_cfg):
    cfg = apply_overrides(smoke_cfg, ["env.reward_shaping.enabled=true", "env.reward_shaping.speed_bonus=1.0",
                                      "env.reward_shaping.progress_window=3"])
    env = make_env(cfg, "train", seed=1)
    try:
        env.reset()
        stop = np.array([[0.0, -1.0, 1.0]], dtype=np.float32)       # no gas: no speed, no progress -> no bonus
        for _ in range(10):
            _, _, _, infos = env.step(stop)
        assert "speed_bonus" not in infos[0] or infos[0]["speed_bonus"] < 0.05
        go = np.array([[0.0, 1.0, -1.0]], dtype=np.float32)          # full throttle on the road start: bonus appears
        got = []
        for _ in range(40):
            _, _, _, infos = env.step(go)
            got.append(infos[0].get("speed_bonus", 0.0))
        assert max(got) > 0.0 and max(got) <= 1.5 + 1e-6
    finally:
        close_env(env)
    with pytest.raises(Exception):
        apply_overrides(smoke_cfg, ["env.reward_shaping.speed_ref=0"])
