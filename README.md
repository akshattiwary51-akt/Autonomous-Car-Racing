# Autonomous Car Racing with Deep Reinforcement Learning

A modular, reproducible pipeline that trains a racing car to drive on randomly generated tracks from raw
pixels, using **Soft Actor-Critic (SAC)** from Stable-Baselines3 with a CNN feature extractor on
Gymnasium's `CarRacing` environment. It includes training with checkpoint/resume, held-out evaluation,
baselines, headless visualization and video recording, a SAC-vs-PPO experiment script, benchmarks and tests.

> **Status / honesty note.** Everything in this repository was executed and tested (see
> [Verification status](#verification-status)). Best measured agent: a PPO policy trained 200k steps on a 2-core CPU that
> follows the road and covers ≈ 75 % of held-out tracks (reward ≈ 650) but **completes no laps**; SAC from pixels did not learn
> in the CPU budget available. See [Results](#results-measured). None of this is converged driving performance. A full-quality agent needs far more training
> (hours on a GPU) — commands for that are below.

## Contents
1. [Overview](#1-overview) · 2. [RL concepts](#2-reinforcement-learning-concepts) · 3. [SAC + CNN](#3-sac-architecture-and-cnn-feature-extraction) ·
4. [Architecture](#4-system-architecture) · 5. [Install](#5-installation) · 6. [Compatibility notes](#6-dependency-and-compatibility-notes) ·
7. [Train](#7-training) · 8. [Resume](#8-resuming-training-and-loading-checkpoints) · 9. [Evaluate](#9-evaluation) ·
10. [Visualize / video](#10-visualization-and-video-recording) · 11. [TensorBoard](#11-tensorboard) ·
12. [Interpreting results](#12-interpreting-results) · 13. [Troubleshooting](#13-troubleshooting) · 14. [Limitations](#14-known-limitations-and-future-work)

---

## 1. Overview

**Goal:** learn a policy `π(a | image)` that outputs continuous *steering, gas, brake* and drives a car around
procedurally generated tracks without any hand-written driving logic.

| Item | Choice |
|---|---|
| Environment | `CarRacing-v3` (Gymnasium ≥ 1.0, Box2D). **`CarRacing-v2` no longer exists** in Gymnasium 1.x (`DeprecatedEnv`); v3 is its direct successor (same task, bug fixes) |
| Algorithm | SAC (primary), PPO (optional comparison) — Stable-Baselines3 2.x |
| Observation | 96×96 RGB → grayscale → 64×64 → 4 stacked frames (`uint8`, 64×64×4) |
| Action | continuous `[-1, 1]³` → (steer, gas, brake), see [§4](#action-mapping) |
| Reward | native CarRacing reward (+1000/N per new tile, −0.1 per frame, −100 for leaving the playfield) |
| Tracking | TensorBoard, Monitor CSVs, CSV/JSON/PNG/HTML result files |

## 2. Reinforcement learning concepts

* **MDP / policy:** the agent observes a stack of frames, outputs an action, and receives a reward. The
  objective is the discounted return `E[Σ γᵗ rₜ]` (γ = 0.99).
* **Frame stacking:** a single image does not reveal speed or steering rate; 4 stacked frames make the state
  approximately Markov.
* **Action repeat (2):** each decision lasts 2 simulator frames – halves the decision horizon (easier credit
  assignment), doubles the effective simulated time per agent step.
* **Off-policy learning / replay buffer:** SAC stores transitions and reuses them many times (sample efficient,
  but the buffer of image observations is the main memory cost).
* **Maximum-entropy RL:** SAC maximises `return + α·entropy`. Automatic entropy tuning adapts α toward a target
  entropy of `−dim(A) = −3`, giving exploration early and determinism later without hand-tuning.
* **Twin critics + target networks (τ = 0.005):** reduce Q-value over-estimation and stabilise bootstrapping.
* **Terminated vs truncated:** leaving the playfield / finishing the lap is `terminated`; the 1000-frame limit is
  `truncated`. SB3 bootstraps through truncation (timeout handling), and evaluation reports them separately.
* **Held-out evaluation:** the best model is selected on tracks that are never used for training, with the same
  tracks every time so scores are comparable.

## 3. SAC architecture and CNN feature extraction

```
obs (4×64×64, uint8) ──/255──► CustomCNN ──► 256-d features
                                  Conv 32@8×8/4 → ReLU
                                  Conv 64@4×4/2 → ReLU
                                  Conv 64@3×3/1 → ReLU → Flatten → Linear(→256) → ReLU
actor:  features → MLP[256,256] → Gaussian(μ,σ) → tanh → action ∈ [-1,1]³
critics (×2): [features(obs), action] → MLP[256,256] → Q
```
* The actor and each critic have **their own CNN** (`share_features_extractor: false`, the SB3 SAC default):
  critic gradients do not distort the actor's features. PPO shares one CNN between actor and critic.
* `src/feature_extractor.py` infers the flattened size from the observation space, so any `resize`,
  `frame_stack` and conv settings work, and it raises a clear error when kernels do not fit the image.
* 1.68 M trainable parameters in the default config.

### Hyperparameter rationale (`configs/sac.yaml`)
| Setting | Value | Why |
|---|---|---|
| `grayscale`, `resize` | on, 64 | colour carries little extra signal for road following; 3× / 2.25× less replay memory and compute |
| `frame_stack` | 4 | infer velocity; benchmarked cost is modest (see Performance) |
| `learning_rate` | 3e-4 | SAC standard |
| `buffer_size` | 50 000 | 64×64×4 obs ×2 (obs+next_obs) ≈ **1.6 GiB**; raise on large-RAM machines (1e5 ≈ 3.3 GiB). A guard aborts with advice if it does not fit in RAM |
| `learning_starts` | 5 000 | fills the buffer with diverse data before updates |
| `batch_size` | 128 | stable gradient estimates for image inputs |
| `train_freq` / `gradient_steps` | 1 / 1 | update-to-data ratio 1 (best sample efficiency, slowest on CPU → `configs/sac_cpu.yaml` uses 4/1) |
| `ent_coef` | auto | adaptive exploration |
| `gamma`, `tau` | 0.99, 0.005 | standard |

These are sensible starting points, **not tuned optima**; no hyperparameter search was run.

### Latest result: PPO profile trained to 400k steps — measured
`ppo_fast.yaml` continued from its 160k-step checkpoint to 400k steps (PPO resume restores the optimiser; 240k further steps,
≈ 100 min on 2 CPU cores). In-training held-out evaluation (deterministic actions, 5 tracks) rose from ≈ 200 (160k–240k) to 512
(400k). Checkpoints 260k–400k were then re-scored with sampled actions on 8 selection tracks (seeds 30000+); the 400k checkpoint won
(850 ± 32, 95 % coverage). **Final test** on 20 *different* tracks (seeds 20000–20019, identical for every model, no selection
on them), via `run_comparison` (`results/ppo_fast_long/final_comparison/`):

| model | actions | reward (mean ± std) | coverage | laps (of 20) |
|---|---|---|---|---|
| random actions | - | -30 ± 6 | 7 % | 0 |
| SAC demo, 20k steps | det. | 61 ± 43 | 16 % | 0 |
| PPO @ ~200k steps (earlier run) | stoch. | 653 ± 118 | 75 % | 0 |
| **PPO @ 400k steps** | **stoch.** | **789 ± 162** | **89 %** | **1** |
| PPO @ 400k steps | det. | 584 ± 211 | 69 % | 0 |

Pre-declared success criteria (see the Colab notebook): coverage ≥ 80 % **met (89 %)**, reward ≥ 700 **met (789)**, laps ≥ 50 % of
episodes **NOT met (5 %)**. So this is a road-following policy, not yet a reliable lap-completing driver: one video
(`videos/ppo_fast_long_best.mp4`, seed 20000, 96.9 % coverage) stays on the road through all curves, but it is slow (gas ≈ 0.4) so most episodes hit the 1000-frame limit before a lap
is registered. Single training seed; checkpoint quality fluctuates (the 320k checkpoint scored only 482 on the selection tracks), so a different seed may land
elsewhere. Weights: `models/ppo_fast_long/best_selected.zip`.

**Training further did not help (measured).** Continuing the same run from 400k to 600k steps gave a plateau, not a gain. Re-scored
with sampled actions on the 8 selection tracks the 400k checkpoint was still best (850 ± 32, 95 % coverage; 420k-600k: 704-834, coverage 80-93 %,
0-25 % laps). On the 20 test tracks: 560k → 797 ± 157, 89 % coverage, 2 laps; 600k → 799 ± 121, 90 %, 0 laps (vs 789 ± 162, 89 %, 1 lap at 400k).
Differences between these are within noise for 20 episodes and one seed. Reaching reliable laps therefore likely needs a change
(speed incentive, e.g. a throughput/lap-time term, or a different exploration scheme) rather than more of the same training.
Files: `results/ppo_600k/`.

### Earlier result: PPO @ 200k steps (`configs/ppo_fast.yaml`) — measured
After SAC failed (see feature-collapse section below), a PPO profile tuned for small CPUs was trained for 200 000 agent
steps (800 000 simulator frames, 2 subprocess envs, ≈ 95 min, 36 steps/s). Differences from `ppo.yaml`: action repeat 4,
`reward_scale: 0.1` (training only), optional stuck-truncation (`reward_shaping.stuck_patience: 40`, training only),
`log_std_init: -1`, lr 1e-4, 512×2 steps per update, batch 256. Evaluation always uses the native reward.

Final evaluation (`python -m src.evaluate --model models/ppo_fast/best_stochastic.zip --episodes 10 --stochastic --compare-baselines`),
10 held-out tracks (seeds 20000-20009), same tracks for every policy:

| policy | reward (mean ± std) | median | coverage | laps | endings |
|---|---|---|---|---|---|
| **PPO checkpoint @160k (stochastic actions)** | **652 ± 94** | 629 | **75 %** | 0 / 10 | 10 time-limit |
| untrained network | -41 ± 20 | -49 | 8.5 % | 0 / 10 | 10 out-of-bounds |
| uniform-random actions | -40 ± 20 | -46 | 6.0 % | 0 / 10 | 10 time-limit |

What this does and does **not** show:
* The agent genuinely learned to steer from pixels: it follows the road through the first corners and covers ~¾ of
  the track on average. **It completes no laps**: in the recorded video it leaves the track part-way and sits on the grass
  until the time limit (no recovery behaviour).
* The 160k checkpoint was chosen among 7 checkpoints using 8 *other* tracks (seeds 30000+), then evaluated on the 10 tracks
  above; single seed, so treat the numbers as indicative, not as a robust estimate. Checkpoint quality fluctuates
  (the 200k checkpoint scored 341 ± 262 on the selection tracks).
* **Deterministic vs stochastic matters for PPO:** for the same 80k checkpoint on identical tracks, stochastic actions
  scored 524 and the deterministic mean only 138. The policy is trained with action noise, clipping and a one-sided brake, so its
  mean action is not the behaviour it learned. `ppo_fast.yaml` therefore uses `evaluation.deterministic: false`;
  SAC/other configs keep deterministic evaluation as the default.
* Reward scaling was added after a first run (same settings, unscaled) peaked at 369 training reward and then degraded with
  value loss in the hundreds; the scaled run kept improving (training reward 527, coverage 69 % at 86k). A single run each, not a
  controlled ablation.
* Files: `results/ppo_fast/` (training curves, eval history, final evaluation), `videos/ppo_fast_best.mp4`;
  weights are git-ignored (`models/ppo_fast/best_stochastic.zip`).

Reproduce: `python -m src.train --config configs/ppo_fast.yaml` (≈ 95 min on 2 cores; weights vary run to run).

### Known training failure (SAC): feature collapse (observed, not solved)
In two CPU runs of `configs/sac_cpu.yaml` (seed 0, 20k and 33k steps) the held-out score *fell* after 5k–10k steps.
Diagnosis on the saved checkpoints: ≈ 75 % of the CNN output units were dead and the features were nearly identical for
every observation (std across frames ≈ 0.0002), so the actor emitted **one constant action** regardless of the image
(steer 0.12, gas ≈ 0.47 → the car circles or leaves the track). Even an *untrained* network's features varied very little
(std 0.0015), because grayscale road frames change little.

Two options were added to counter this (both **off by default**, because no run here was long enough to show they fix it):
* `algo.policy.feature_norm: layernorm` — Linear→LayerNorm→Tanh head (≈ 45× more feature variance at initialisation).
* `algo.policy.share_features_extractor: true` — SB3 then trains the encoder through the critic only and gives the actor
  detached features (the DrQ / SAC-AE recipe).

Measured with 12k steps, CPU profile: LayerNorm alone collapsed again (action std 0); LayerNorm + shared encoder gave a
held-out reward trend of -77 → -60 → -59 instead of degrading, but the policy was still input-independent
(action std 0) and coverage stayed ≈ 6 %, i.e. **not a demonstrated fix**. Untested next steps: far more gradient updates
(`train_freq: 1` on a GPU), a pretrained / auxiliary-loss encoder, PPO as a sanity baseline for the pipeline.
* `algo.policy.augment_pad: 4` — DrQ random-shift augmentation inside the CNN (training mode only; unit-tested). Tried
  together with the two options above for 12k steps (≈ 2 500 gradient updates): held-out reward -63 → -60, coverage ≈ 6 %,
  policy still input-independent. Such methods usually need on the order of 10⁵ updates, which this CPU cannot reach in
  reasonable time, so the combination remains **unproven here**. Suggested GPU recipe to test:
  `--set algo.policy.feature_norm=layernorm algo.policy.share_features_extractor=true algo.policy.augment_pad=4 algo.hyperparams.train_freq=1`.

## 4. System architecture

```
configs/            sac.yaml  ppo.yaml  sac_cpu.yaml  sac_smoke.yaml (tests)
src/
  config.py         typed dataclass config, YAML/JSON loading, strict validation, CLI overrides
  environment.py    env factory: wrappers, seeding, vec envs, frame stacking, spaces check
  feature_extractor.py  CNN for image observations
  agent.py          build/load SAC or PPO, replay-buffer memory guard
  callbacks.py      HeldOutEvalCallback (best model + eval_history.csv), TrainingStatsCallback
  train.py          training CLI: config → env → agent → callbacks → resume/interrupt → summary + plots
  evaluate.py       independent evaluation, baselines, CSV/JSON/plots
  inference.py      run a policy: window / headless / frames
  record_video.py   MP4/GIF recording with HUD
  metrics.py        episode metrics, summaries, plotting
  utils.py          seeds, device, logging, metadata, paths
scripts/            setup_environment.py  compare_algorithms.py  benchmark.py
tests/              pytest suite
notebooks/          autonomous_car_racing_demo.ipynb (Colab)
```

Wrapper chain (inner → outer): `CarRacing-v3 → TrackProgress → ActionRepeat → Monitor → [RewardShaping, train only] →
Observation (gray/resize) → ActionRescale → (Dummy|Subproc)VecEnv → VecFrameStack`.
* `Monitor` sits *before* shaping, so logged episode rewards are always the **native** reward.
* `TrackProgress` reads `tile_visited_count` / `track` (public attributes of CarRacing) to report **coverage**,
  and records explicit `terminated` / `truncated` flags that survive VecEnv auto-reset.
* Training, evaluation and visualization use separate env instances; evaluation envs use fixed seeds
  (`seed+10000+i` during training, `seed+20000+i` for final evaluation) and never reward-shape.

### Action mapping
`a ∈ [-1,1]³ → steer = a₀, gas = (a₁+1)/2, brake = max(a₂, 0)`. The brake is one-sided on purpose: with an
affine map a fresh policy (outputs ≈ 0) would hold brake ≈ 0.5 while also applying gas ≈ 0.5 and barely move,
producing useless early data. (Observed directly in a first smoke run, which is why the mapping was changed.)

### Reward shaping (optional, off by default)
`env.reward_shaping` (training env only; implemented in `RewardShapingWrapper`):

`r' = r − c_steer·|steerₜ − steerₜ₋₁| − c_brake·min(gasₜ, brakeₜ)`, plus: after `stuck_patience` consecutive
steps with native `r ≤ 0` the episode is *truncated* and `stuck_penalty` is subtracted once.

It uses only the native reward and the agent's own actions (no hidden simulator state), contains **no positive
bonus** (so nothing can be farmed), and the native reward stays in `info["native_reward"]`. Expected effect:
smoother steering, no simultaneous gas+brake, fewer wasted idle episodes. **Its benefit has not been
measured** — run a controlled comparison (same seeds/budget) with
`--set env.reward_shaping.enabled=true env.reward_shaping.steer_change_penalty=0.05 ...` before trusting it.
Tile coverage / progress are *not* used as shaping because the native +1000/N-per-tile reward already is that signal.

## 5. Installation

Python 3.10 – 3.13 (verified: **3.13.17**; the original brief asked for 3.11 "or a verified compatible version").

### Linux
```bash
git clone <your-repo-url> autonomous-car-racing && cd autonomous-car-racing
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/setup_environment.py        # checks every dependency + runs a CarRacing step
```
If Box2D fails to build: `sudo apt-get install -y swig build-essential python3-dev` and retry.

### Windows (PowerShell)
```powershell
py -3.11 -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\setup_environment.py
```
`box2d` ships wheels for common Python versions; if pip tries to compile, either use Python 3.11/3.12, install
`swig` (`pip install swig`) plus the Visual Studio Build Tools, or use `conda install -c conda-forge box2d-py`.

### Google Colab (GPU)
Use `notebooks/autonomous_car_racing_demo.ipynb` (Runtime → Change runtime type → GPU). It installs swig, the
requirements, trains, evaluates and displays videos inline (no GUI window needed).

### Exact versions used for testing
`requirements-verified.txt` (Python 3.13, Linux, CPU). For a CUDA machine install the matching PyTorch wheel from
<https://pytorch.org> first, then `pip install -r requirements.txt`.

## 6. Dependency and compatibility notes
* **CarRacing-v2 → v3.** `gym.make("CarRacing-v2")` raises `DeprecatedEnv` in Gymnasium ≥ 1.0; configs are validated
  to require `CarRacing-v3`. Only the Gymnasium API (5-tuple `step`, `reset(seed=)`) is used – no legacy `gym`.
* **SB3 ≥ 2.4 / Gymnasium ≥ 1.0 / NumPy 2** work together (versions in `requirements-verified.txt`).
* **GPU ≠ automatically faster.** The Box2D simulator and its software renderer run on the CPU. On a small CNN the GPU
  speeds up the *gradient updates*, not the simulation, so the overall speed-up is bounded by the environment
  throughput (≈ 77 agent steps/s on one core here). The CPU/GPU split was benchmarked only on CPU in this repo.
* Models are pickled with a reference to `src.feature_extractor.CustomCNN`, so run commands from the repo root
  with `python -m src.<module>` (the scripts add the root to `sys.path` themselves).

## 7. Training

```bash
python -m src.train --config configs/sac.yaml                         # full SAC run (GPU recommended)
python -m src.train --config configs/sac_cpu.yaml                     # CPU-friendly profile
python -m src.train --config configs/sac.yaml --run-name sac_seed1 --seed 1 --timesteps 500000
python -m src.train --config configs/sac.yaml --set algo.hyperparams.buffer_size=100000 env.resize=96
python -m src.train --config configs/ppo.yaml                         # PPO (4 parallel envs)
python -m src.train --config configs/sac_smoke.yaml --run-name smoke  # ~1-2 min end-to-end check
```
Outputs for run `<name>`:

| Path | Content |
|---|---|
| `models/<name>/best_model.zip` | best on held-out eval tracks |
| `models/<name>/final_model.zip`, `config.yaml` | last model + the exact config |
| `checkpoints/<name>/ckpt_<steps>_steps.zip` | periodic checkpoints (`interrupted_<steps>_steps.zip` on Ctrl-C / SIGTERM) |
| `logs/<name>/tb/` · `logs/<name>/monitor/*.csv` · `logs/<name>/train.log` | TensorBoard, per-episode CSVs, text log |
| `results/<name>/eval_history.csv`, `training_curves.png/.html` | evaluation during training, plots |
| `results/<name>/final_*_summary.json / _episodes.csv`, `final_comparison.png` | final held-out evaluation + baselines |
| `results/<name>/metadata.json`, `training_summary.json` | versions, hardware, git commit, timing |

A run folder that already contains outputs is never silently overwritten (`--force` / `--run-name` / `--resume`).
Every `--set key=value` is validated; unknown keys or out-of-range values fail with a clear message.

## 8. Resuming training and loading checkpoints
```bash
python -m src.train --resume checkpoints/sac_carracing                      # latest checkpoint of the run
python -m src.train --resume checkpoints/sac_carracing/ckpt_100000_steps.zip --timesteps 400000
```
The step counter, networks, optimiser state and entropy coefficient continue. **Replay buffers are not saved by
default** (they are GBs); the buffer then restarts empty and `training.resume_warmup_steps` (default 1000) random
steps are collected before gradient updates resume. Set `training.save_replay_buffer: true` to store it next to
checkpoints for an exact resume. PPO has no buffer, so it resumes exactly.

## 9. Evaluation
```bash
python -m src.evaluate --model models/sac_carracing/best_model.zip --episodes 20 --compare-baselines
python -m src.evaluate --model checkpoints/sac_carracing --episodes 10 --stochastic     # latest checkpoint, sampled actions
```
Reports mean / std / median reward, episode length, **track coverage** (visited tiles / total tiles), lap-completion
rate, termination reasons (`lap_complete`, `out_of_bounds`, `time_limit`), total runtime and per-step inference
latency (mean/p50/p95). `--compare-baselines` evaluates an *untrained* network of the same architecture and a
*uniform-random* policy on **the same tracks**. Output: `results/<run>/eval_*_summary.json`, `*_episodes.csv`,
`*_comparison.png`. Deterministic actions are the default; `--stochastic` is reported separately.

## 10. Visualization and video recording
```bash
python -m src.inference --model models/sac_carracing/best_model.zip --episodes 3 --render      # live window (needs a display)
python -m src.inference --model models/sac_carracing/best_model.zip --video videos/sac.mp4      # headless MP4 with HUD
python -m src.inference --random --episodes 1 --video videos/random.mp4                          # random-policy comparison
python -m src.record_video --model models/sac_carracing/best_model.zip --episodes 2 --out videos/sac.gif
python -m src.inference --model ... --frames-dir videos/frames --frame-every 5                    # PNG frames
```
Training never renders. `--render` fails with a clear message when no display exists (servers, Colab) – use
`--video` / `--frames-dir`, which render off-screen. Videos play at 25 fps (50 simulator fps ÷ action repeat 2) and
show episode, step, cumulative reward and coverage.

## Diagnosing a model that "just drives straight" (`src/diagnostics.py`)
```bash
# model trained outside this repo on a plain CarRacing-v3 (96x96 RGB, native actions)
python -m src.diagnostics --model sac_final.zip --env legacy --algo sac --episodes 5 --both --out results/diag
# model trained with this repo (config read from the model folder)
python -m src.diagnostics --model models/ppo_fast/best_stochastic.zip --env project --stochastic
```
Reports per-dimension steering / gas / brake statistics (mean, std, percentiles, **std within an episode**), saturation, left/right
balance, gas+brake overlap, speed, track coverage and heuristic verdicts (policy collapse, no steering, steering bias, brake held
by an affine action mapping, idle episodes that "survive" 1000 steps). Verdicts are rules of thumb, not proofs. Evaluation never
modifies a model. `run_comparison(...)` evaluates several models on **identical track seeds**; `scripts/select_checkpoint.py`
re-scores all checkpoints of a run on separate seeds.

Measured with it in this project (same 3 tracks, seeds 50000+):

| model | actions | reward | steer std in episode | verdict |
|---|---|---|---|---|
| default-style SB3 SAC, 300 steps (plain env, stand-in for a barely-trained model) | det. | -28 (1000 steps) | 0.000 | collapse; gas 0.49 + brake 0.50 together |
| this repo's SAC demo (20k steps) | det. | 81 | 0.000 | collapse (constant action) |
| PPO profile (`ppo_fast`, 200k steps) | stoch. | 459 | 0.56 | no collapse pattern |

## Colab workflow for an existing SAC project (`notebooks/colab_road_following.ipynb`)
Numbered cells: mount Drive → unpack code → persist outputs in a **separate** Drive folder (old models are only read) →
diagnose old models → success criteria fixed *before* training → train PPO (auto-resume, single-job lock) → TensorBoard →
checkpoint selection on separate seeds → old-vs-new comparison on identical seeds with PASS/FAIL → videos →
optional experimental SAC (`configs/sac_drq.yaml`, unvalidated) → resume instructions and troubleshooting.
The older `autonomous_car_racing_demo.ipynb` is a short demo; use the new one for the full workflow.

## 11. TensorBoard
```bash
tensorboard --logdir logs                 # then open http://localhost:6006
# Colab:  %load_ext tensorboard   then   %tensorboard --logdir logs
```
Scalars: `rollout/ep_rew_mean`, `rollout/ep_len_mean`, `rollout/ep_coverage_mean`, `train/critic_loss`,
`train/actor_loss`, `train/ent_coef`, `eval/mean_reward`, `eval/mean_coverage`, `eval/lap_completion_rate`,
`perf/steps_per_second`, `perf/process_rss_gb`.

## 12. Interpreting results
* Native reward ≈ `1000 − 0.1·frames` for a full lap, so **≈ 900** ≈ a clean lap; ≈ 0 means idling/wandering;
  strongly negative values come from the −100 out-of-playfield penalty or long idle periods.
* Prefer **coverage** and **lap-completion rate** alongside reward; reward alone hides *why* an episode ended.
* Judge learning by the **held-out** curve (`eval/*`, `results/<run>/eval_history.csv`), averaged over several
  seeds. With 3–10 evaluation episodes the standard error is large; do not read single evaluations as progress.
* `untrained_network` and `random_uniform` baselines tell you whether any learning happened at all.
* `results/<run>/training_curves.png` shows reward, coverage and rolling-std (stability) vs. agent steps
  (1 agent step = 2 simulator frames).

## 13. Troubleshooting
| Symptom | Fix |
|---|---|
| `DeprecatedEnv: Environment version v2 for CarRacing is deprecated` | use `CarRacing-v3` (the repo already does) |
| `ModuleNotFoundError: Box2D` / swig build errors | Linux `apt install swig build-essential`; Windows: Python 3.11/3.12 wheels, or `conda install -c conda-forge box2d-py`; then `pip install "gymnasium[box2d]"` |
| `pygame.error: No available video device` / `No display found for --render` | headless machine: use `--video`/`--frames-dir`; `SDL_VIDEODRIVER=dummy` is set automatically |
| MP4 not written | `pip install imageio-ffmpeg`; the code falls back to GIF with a warning |
| `Replay buffer would need X GiB…` | lower `algo.hyperparams.buffer_size`, `env.resize`, `env.frame_stack`, or set `optimize_memory_usage: true` (note: timeouts then count as terminal) |
| `Device 'cuda' requested but CUDA is not available` | use `--device cpu` or install a CUDA PyTorch build; Colab: enable the GPU runtime |
| `CNN kernels/strides … too large for the observation shape` | increase `env.resize` or reduce `cnn_kernels/strides` |
| `Could not load … The config must match` | pass `--config` of the training run (shape/algorithm/CNN must match) |
| `Run '<name>' already has outputs` | `--run-name`, `--resume`, or `--force` |
| Training is very slow on CPU | use `configs/sac_cpu.yaml`, `env.resize=48`, `train_freq: 4`, `experiment.torch_threads: 1-2`, `env.n_envs: 2` with `vec_type: subproc` |
| CUDA out-of-memory | lower `batch_size`; replay buffer lives in RAM, not VRAM |

## 14. Known limitations and future work
* **No reliable lap-completing agent is shipped.** The best (PPO, 400k steps) covers ≈ 89 % of unseen tracks and finished a lap in 1 of 20 episodes (it is slow and hits the time limit); longer training, speed incentives, or more seeds are needed for reliable laps.
* GPU throughput, mixed precision (AMP) and multi-seed SAC-vs-PPO results were **not measured** (no GPU / limited
  compute here). SB3 has no built-in AMP support, so it was not added.
* Evaluation with few episodes is noisy; the PPO/SAC comparison script supports multiple seeds, but only a tiny
  smoke run of it was executed.
* Ideas: data augmentation (random shift / DrQ), an auxiliary autoencoder or pretrained encoder, larger UTD ratios
  on GPU, vectorised SAC (`n_envs > 1` is supported), curriculum on track difficulty, domain randomisation
  (`env.domain_randomize`), reward-shaping ablations with multiple seeds.

---

## Performance (measured)
Hardware for all measurements: 2 CPU cores, 7 GB RAM, **no GPU**, Python 3.13, PyTorch 2.14 (CPU). Produced by
`python scripts/benchmark.py` (`results/benchmarks/benchmark.csv`); 400 timed agent steps per row after a warm-up,
single run each (no repeats → expect ±10–20 % noise).

| Simulator only (random actions) | agent steps/s |
|---|---|
| 1 env (dummy) | 77 |
| 2 envs (dummy) | 76 |
| 2 envs (subprocess) | **125** (≈ 1.6× — vectorisation helps only with real parallel workers) |

| SAC training variant (CPU, includes gradient updates) | steps/s |
|---|---|
| 64 px gray stack 4 (default `sac.yaml`) | 6.4 |
| 48 px gray stack 4 | 10.1 |
| 96 px gray stack 4 (native size) | 3.1 |
| 64 px **RGB** stack 4 | 4.3 |
| 64 px gray, no frame stack | 9.2 |
| 64 px gray stack 4, `action_repeat=4` | 5.9 (per agent step; each step simulates 2× the frames) |
| 64 px gray stack 4, `train_freq=4` (`sac_cpu.yaml`) | **19.3** |

Take-aways: the CNN update dominates CPU SAC cost (the simulator alone does ~77 steps/s); resolution, colour and
frame-stack all cost real throughput; fewer updates per step is the largest lever but lowers sample efficiency
(not quantified here). Process RSS after these runs was 0.84–1.20 GiB (cumulative within one process, so only an
upper-bound indication); the replay buffer for the default config is estimated at 1.6 GiB. `process_rss_gb` is
also logged to TensorBoard during training.

At 6.4 steps/s the default 300 000-step run would take ≈ 13 h on this machine – hence the CPU profile and the
recommendation to use a GPU runtime for full training.

## Results (measured)

**Demo run** `python -m src.train --config configs/sac_cpu.yaml --run-name sac_cpu_demo` — 20 000 agent steps
(40 000 simulator frames), seed 0, CPU only, 16.9 min (19.8 steps/s end-to-end). Final evaluation: 10 held-out tracks
(seeds 20000-20009), deterministic actions, identical tracks for all three policies.

| policy | reward (mean ± std) | median | track coverage | laps completed | episode endings |
|---|---|---|---|---|---|
| SAC `best_model` (selected at 5 000 steps) | 57.0 ± 57.2 | 48.7 | 16.0 % | 0 / 10 | 9 time-limit, 1 out-of-bounds |
| untrained network (same architecture) | 2.6 ± 45.1 | -3.9 | 10.3 % | 0 / 10 | 10 time-limit |
| uniform-random actions | -44.5 ± 16.6 | -49.5 | 5.6 % | 0 / 10 | 10 time-limit |

Inference latency ≈ 2.0 ms/step (CPU, batch of one).

**How to read it — this agent cannot drive.** 20k steps is a tiny fraction of the budget CarRacing needs. The
held-out evaluation at 10k, 15k and 20k steps was *worse* than at 5k (≈ -69 reward: the deterministic policy
floors the throttle and leaves the playfield after ~80 steps), so "best model" is the early 5k checkpoint, whose edge over the
untrained network (57 vs 3 reward, n = 10, std ≈ 50) is within noise for a single seed. The run demonstrates that the
pipeline works end to end (train → checkpoint → best-model selection → resume → evaluation → plots → video); it is
**not** evidence that the hyperparameters learn to drive. Files: `results/sac_cpu_demo/` (`training_curves.png`,
`final_comparison.png`, `final_*_summary.json`, `eval_history.csv`). The folder is git-ignored; re-create it with the command above.

The SAC-vs-PPO script (`scripts/compare_algorithms.py`) was executed only as a 2 048-step, single-seed smoke test to
verify it runs; **no comparison between the algorithms is claimed**. Reward shaping was implemented and unit-tested but no
ablation was run.


## Verification status
Executed in this repository's environment (Linux, 2 CPU cores, no GPU, Python 3.13.17, Gymnasium 1.4.0,
SB3 2.9.0, PyTorch 2.14.1):

| Check | Command | Status |
|---|---|---|
| Dependency/compat check + CarRacing-v3 step | `python scripts/setup_environment.py` | passed |
| Test suite (41 tests: env, config validation, agent init, smoke training, checkpoint save/reload, resume, evaluation metrics, inference, video, error messages, subprocess vec-env) | `python -m pytest -q` | **41 passed** |
| SAC training end-to-end (smoke + 20k-step demo) | `python -m src.train ...` | passed |
| PPO training end-to-end (1 024 steps, smoke) | `python -m src.train --config configs/ppo.yaml ...` | passed |
| CLI resume of the demo run (20 000 → 21 000 steps) | `python -m src.train --resume checkpoints/sac_cpu_demo --timesteps 21000` | passed |
| Evaluate CLI with baselines; headless MP4/GIF recording; `--render` without display error | `python -m src.evaluate / src.inference / src.record_video` | passed |
| Benchmark | `python scripts/benchmark.py` | passed (numbers above) |

**Not verified here** (no hardware/time): CUDA training and GPU speed-ups, a live `--render` window, Windows/macOS
installs, the Colab notebook itself (its commands mirror the verified CLI), multi-seed SAC-vs-PPO results, and any
training long enough to produce a competent driver.

### Definition of done — quick path (≈ 3 min)
```bash
pip install -r requirements.txt
python scripts/setup_environment.py
python -m src.train --config configs/sac_smoke.yaml --run-name smoke --force   # env → agent → train → checkpoint → eval
python -m src.evaluate --model models/smoke/best_model.zip --episodes 2 --compare-baselines
python -m src.inference --model models/smoke/best_model.zip --max-steps 100 --video videos/smoke.mp4
python -m pytest -q
```


## Speed-bonus experiment (negative result)

Hypothesis: the 400k PPO agent drives the track but is slightly too slow, so most episodes hit the 1000-frame limit at ~90-100% coverage
instead of finishing a lap. **This modifies the training reward** (not the CarRacing-v3 benchmark; evaluation still uses the native reward).
`reward_shaping.speed_bonus=0.5 * min(speed/60, 1.5)` is paid only if a new tile was reached within the last 8 agent steps
(`configs/ppo_speed.yaml`). Fine-tuned 400k -> 550k steps, checkpoint chosen on seeds 30000+ (8 eps, stochastic), then scored on
unseen seeds 20000-20019 against the 400k model:

| model | reward (mean ± std) | coverage | laps |
|---|---|---|---|
| ppo 400k (baseline) | 789 ± 162 | 88.7% | 1/20 |
| ppo + speed bonus (ckpt 440k) | 760 ± 163 | 86.1% | 0/20 |

Conclusion: no measurable improvement; differences are within noise (20 episodes). The lap-completion criterion (>= 50%) is **still not met**
and the speed bonus is not recommended. Limits of this experiment: one seed, one bonus coefficient, CPU-only budget.
