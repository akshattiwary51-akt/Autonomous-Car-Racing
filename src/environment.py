"""Environment factory for CarRacing-v3 with a consistent observation/action interface.

Wrapper chain (inner -> outer)::

    CarRacing-v3
      -> TrackProgressWrapper   adds tiles_visited / tiles_total / terminated / truncated to ``info``
      -> ActionRepeatWrapper    repeats each action N times (sums rewards)
      -> Monitor                records *native* episode reward/length (before any shaping)
      -> RewardShapingWrapper   optional, training only
      -> ObservationWrapper     optional grayscale + resize (uint8 HxWxC)
      -> ActionRescaleWrapper   agent action space is [-1, 1]^3 for steer / gas / brake
    -> DummyVecEnv | SubprocVecEnv
    -> VecFrameStack            channels-last frame stacking
    (SB3 later adds VecTransposeImage when the model wraps the env)

Observations are always ``uint8`` arrays of shape (H, W, C * frame_stack).
Actions are always float32 in [-1, 1]^3 and are mapped to the native
(steer in [-1, 1], gas in [0, 1], brake in [0, 1]) space by the rescale wrapper
(see ``ActionRescaleWrapper`` for the exact mapping, including the one-sided brake).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, SupportsFloat, Tuple

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecFrameStack

from .config import Config, EnvConfig, ShapingConfig

EVAL_SEED_OFFSET = 10_000   # seeds used by the in-training EvalCallback
FINAL_SEED_OFFSET = 20_000  # held-out seeds used for post-training evaluation


# --------------------------------------------------------------------------- #
# Wrappers
# --------------------------------------------------------------------------- #
class TrackProgressWrapper(gym.Wrapper):
    """Expose track progress that CarRacing keeps as plain attributes.

    ``tile_visited_count`` and ``track`` are public attributes of the CarRacing
    implementation, so coverage is reliably measurable. Also records explicit
    ``terminated`` / ``truncated`` flags in ``info`` (survive VecEnv auto-reset).
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        base = self.env.unwrapped
        total = max(len(getattr(base, "track", [])), 1)
        visited = int(getattr(base, "tile_visited_count", 0))
        info = dict(info)
        info["tiles_visited"] = visited
        info["tiles_total"] = total
        info["coverage"] = visited / total
        info["terminated"] = bool(terminated)
        info["truncated"] = bool(truncated)
        info.setdefault("lap_finished", False)
        return obs, reward, terminated, truncated, info


class ActionRepeatWrapper(gym.Wrapper):
    """Repeat each action ``repeat`` times, summing the native rewards."""

    def __init__(self, env: gym.Env, repeat: int):
        super().__init__(env)
        self.repeat = repeat

    def step(self, action):
        total = 0.0
        for _ in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total += float(reward)
            if terminated or truncated:
                break
        return obs, total, terminated, truncated, info


class RewardShapingWrapper(gym.Wrapper):
    """Optional shaped reward built **only** from signals the simulator exposes.

    Inputs: native reward ``r``, the executed native action ``(steer, gas, brake)``.
    Formula (per agent step)::

        r' = r - c_steer * |steer_t - steer_{t-1}| - c_brake * min(gas_t, brake_t)
         + c_speed * min(speed / speed_ref, 1.5)   only if a new tile was reached within the last N steps
        stuck: after ``stuck_patience`` consecutive steps with r <= 0 the episode is
               *truncated* (not terminated) and ``stuck_penalty`` is subtracted once.

    Expected effect: smoother steering, no simultaneous throttle+brake, and shorter
    wasted episodes when the car sits on grass/at standstill (native reward is
    -0.1 per frame while no new tile is touched). The optional speed term is the only positive bonus and
    is gated on recent tile progress (a stationary or off-track car gets none), but it DOES change the training reward. The native
    reward is kept in ``info['native_reward']`` and Monitor logs it unshaped.
    """

    def __init__(self, env: gym.Env, cfg: ShapingConfig):
        super().__init__(env)
        self.cfg = cfg
        self._prev_steer = 0.0
        self._nonpositive_streak = 0
        self._since_progress = 0

    def reset(self, **kwargs):
        self._prev_steer = 0.0
        self._nonpositive_streak = 0
        self._since_progress = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        native = float(reward)
        steer, gas, brake = float(action[0]), float(action[1]), float(action[2])
        shaped = native
        shaped -= self.cfg.steer_change_penalty * abs(steer - self._prev_steer)
        shaped -= self.cfg.brake_gas_penalty * min(gas, brake)
        self._prev_steer = steer
        # progress-gated speed bonus: paid only while new tiles keep being reached (native reward > 0 means a new
        # tile was touched), so spinning or driving fast on grass earns nothing. Changes the task's reward!
        self._since_progress = 0 if native > 0 else self._since_progress + 1
        if self.cfg.speed_bonus > 0 and self._since_progress < self.cfg.progress_window:
            car = getattr(self.env.unwrapped, "car", None)
            if car is not None:
                speed = float(np.linalg.norm(car.hull.linearVelocity))
                bonus = self.cfg.speed_bonus * min(speed / self.cfg.speed_ref, 1.5)
                shaped += bonus
                info = dict(info, speed_bonus=bonus)
        if self.cfg.stuck_patience > 0:
            self._nonpositive_streak = self._nonpositive_streak + 1 if native <= 0 else 0
            if self._nonpositive_streak >= self.cfg.stuck_patience and not (terminated or truncated):
                truncated = True
                shaped -= self.cfg.stuck_penalty
                info = dict(info, stuck_truncated=True)
        info = dict(info, native_reward=native)
        return obs, shaped, terminated, truncated, info


class RewardScaleWrapper(gym.RewardWrapper):
    """Multiply the reward by a constant (training only). Keeps value-function targets O(1) for PPO/SAC.

    Placed after ``Monitor``, so logged/evaluated episode rewards stay on the native scale.
    """

    def __init__(self, env: gym.Env, scale: float):
        super().__init__(env)
        self.scale = float(scale)

    def reward(self, reward: SupportsFloat) -> float:
        return float(reward) * self.scale


class ObservationWrapper(gym.ObservationWrapper):
    """Optional grayscale + square resize, always returning uint8 (H, W, C)."""

    def __init__(self, env: gym.Env, grayscale: bool, resize: Optional[int]):
        super().__init__(env)
        old = env.observation_space
        assert isinstance(old, spaces.Box) and len(old.shape) == 3, "expected an image Box space"
        h, w, c = old.shape
        self.grayscale, self.size = grayscale, resize
        out_h, out_w = (resize, resize) if resize else (h, w)
        out_c = 1 if grayscale else c
        self.observation_space = spaces.Box(0, 255, (out_h, out_w, out_c), dtype=np.uint8)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        if self.grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        if self.size:
            obs = cv2.resize(obs, (self.size, self.size), interpolation=cv2.INTER_AREA)
        if obs.ndim == 2:
            obs = obs[..., None]
        return np.ascontiguousarray(obs, dtype=np.uint8)


class ActionRescaleWrapper(gym.ActionWrapper):
    """Agent-facing action space [-1, 1]^3 -> native ``(steer, gas, brake)``.

    SAC/PPO with squashed Gaussians assume a symmetric, normalised action space;
    CarRacing's native space is Box([-1, 0, 0], [1, 1, 1]) which SB3 would warn about.

    Mapping::

        steer = a0                      (already in [-1, 1])
        gas   = (a1 + 1) / 2            affine, [-1, 1] -> [0, 1]
        brake = max(a2, 0)              one-sided: negative output means "no brake"

    The one-sided brake matters for exploration: a freshly initialised policy outputs
    values around 0, and an affine mapping would then hold the brake at ~0.5 while also
    applying ~0.5 gas, so the car barely moves and early data is nearly useless.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        native = env.action_space
        assert isinstance(native, spaces.Box) and native.shape == (3,), "expected CarRacing's 3-D Box action space"
        self.action_space = spaces.Box(-1.0, 1.0, native.shape, dtype=np.float32)

    def action(self, action: np.ndarray) -> np.ndarray:
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        return np.array([a[0], (a[1] + 1.0) * 0.5, max(float(a[2]), 0.0)], dtype=np.float32)


class SeedCycleWrapper(gym.Wrapper):
    """Make every reset use the next seed from a fixed list (cyclically).

    Gives evaluation a fixed, reproducible set of tracks that never overlaps with
    the (differently seeded) training tracks, and lets different policies be
    compared on identical tracks.
    """

    def __init__(self, env: gym.Env, seeds: Sequence[int]):
        super().__init__(env)
        if len(seeds) == 0:
            raise ValueError("SeedCycleWrapper needs at least one seed.")
        self.seeds = list(seeds)
        self._i = 0

    def restart_seed_cycle(self) -> None:
        """Start again from the first seed (used so every evaluation sees the same tracks)."""
        self._i = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        seed = self.seeds[self._i % len(self.seeds)]
        self._i += 1
        return self.env.reset(seed=seed, options=options)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def _make_single_env(
    env_cfg: EnvConfig, *, seed: int, render_mode: Optional[str], monitor_path: Optional[str],
    use_shaping: bool, eval_seeds: Optional[Sequence[int]],
) -> gym.Env:
    env = gym.make(
        env_cfg.id, continuous=env_cfg.continuous, render_mode=render_mode,
        lap_complete_percent=env_cfg.lap_complete_percent, domain_randomize=env_cfg.domain_randomize,
    )
    env = TrackProgressWrapper(env)
    if env_cfg.action_repeat > 1:
        env = ActionRepeatWrapper(env, env_cfg.action_repeat)
    env = Monitor(env, filename=monitor_path, info_keywords=("coverage", "lap_finished"))
    if use_shaping and env_cfg.reward_shaping.enabled:
        env = RewardShapingWrapper(env, env_cfg.reward_shaping)
    if use_shaping and env_cfg.reward_scale != 1.0:
        env = RewardScaleWrapper(env, env_cfg.reward_scale)
    env = ObservationWrapper(env, env_cfg.grayscale, env_cfg.resize)
    env = ActionRescaleWrapper(env)
    if eval_seeds is not None:
        env = SeedCycleWrapper(env, eval_seeds)
    else:
        env.reset(seed=seed)  # seeds the track RNG; later resets continue the stream
        env.action_space.seed(seed)
    return env


def make_env(
    cfg: Config,
    mode: str = "train",
    *,
    seed: Optional[int] = None,
    n_envs: Optional[int] = None,
    render_mode: Optional[str] = None,
    log_dir: Optional[Path] = None,
    eval_seeds: Optional[Sequence[int]] = None,
) -> VecEnv:
    """Build a vectorised, frame-stacked environment.

    Args:
        cfg: experiment configuration.
        mode: ``"train"`` (shaping allowed, seeded streams), ``"eval"`` or ``"visualize"``
            (native reward, no shaping, fixed seed cycle).
        seed: base seed. Train env ``i`` uses ``seed + i``. Eval defaults to
            ``seed + EVAL_SEED_OFFSET`` so tracks differ from training ones.
        n_envs: overrides ``cfg.env.n_envs`` (eval/visualize always use 1 unless given).
        render_mode: ``None`` | ``"rgb_array"`` | ``"human"``. Keep ``None`` for training.
        log_dir: where per-env Monitor CSVs are written (``None`` disables files).
        eval_seeds: explicit seed list for eval/visualize (cycled on each reset).
    """
    if mode not in ("train", "eval", "visualize"):
        raise ValueError(f"mode must be 'train', 'eval' or 'visualize', got '{mode}'")
    base_seed = cfg.experiment.seed if seed is None else seed
    train = mode == "train"
    count = (cfg.env.n_envs if train else 1) if n_envs is None else n_envs
    if not train and eval_seeds is None:
        n_ep = max(cfg.evaluation.n_eval_episodes, 1)
        start = base_seed + (EVAL_SEED_OFFSET if mode == "eval" else FINAL_SEED_OFFSET)
        eval_seeds = [start + i for i in range(n_ep)]
    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)

    def thunk(i: int) -> Callable[[], gym.Env]:
        monitor = str(Path(log_dir) / f"{mode}_env{i}") if log_dir is not None else None
        # Parallel eval envs are not supported by the seed-cycle; one env per eval/visualize.
        return lambda: _make_single_env(
            cfg.env, seed=base_seed + i, render_mode=render_mode, monitor_path=monitor,
            use_shaping=train, eval_seeds=None if train else eval_seeds,
        )

    thunks = [thunk(i) for i in range(count)]
    if cfg.env.vec_type == "subproc" and count > 1:
        venv: VecEnv = SubprocVecEnv(thunks, start_method="spawn" if _is_windows() else "fork")
    else:
        venv = DummyVecEnv(thunks)
    if train:
        venv.seed(base_seed)
    if cfg.env.frame_stack > 1:
        venv = VecFrameStack(venv, n_stack=cfg.env.frame_stack, channels_order="last")
    return venv


def _is_windows() -> bool:
    import sys

    return sys.platform.startswith("win")


def close_env(env: Optional[VecEnv]) -> None:
    """Close an env without masking the original error when called from ``finally``."""
    if env is None:
        return
    try:
        env.close()
    except Exception as exc:  # pragma: no cover - best-effort cleanup, but never silent
        from .utils import get_logger

        get_logger().warning("Error while closing environment: %r", exc)


def observation_shape(cfg: Config) -> Tuple[int, int, int]:
    """Observation shape (H, W, C*stack) produced by ``make_env`` for this config."""
    side = cfg.env.resize or 96
    channels = (1 if cfg.env.grayscale else 3) * cfg.env.frame_stack
    return side, side, channels


def check_env_spaces(venv: VecEnv, cfg: Config) -> Dict[str, Any]:
    """Validate observation/action spaces and return a small description dict."""
    obs_space, act_space = venv.observation_space, venv.action_space
    expected = observation_shape(cfg)
    if tuple(obs_space.shape) != expected or obs_space.dtype != np.uint8:
        raise RuntimeError(f"Unexpected observation space {obs_space}; expected uint8 {expected}.")
    if not (isinstance(act_space, spaces.Box) and act_space.shape == (3,)
            and np.allclose(act_space.low, -1) and np.allclose(act_space.high, 1)):
        raise RuntimeError(f"Unexpected action space {act_space}; expected Box(-1, 1, (3,)).")
    return {"observation_shape": tuple(obs_space.shape), "action_shape": tuple(act_space.shape)}
