import numpy as np
import pandas as pd

from src.diagnostics import (_RandomModel, load_legacy_model, make_legacy_env, rollout_with_stats, summarize_actions,
                             verdicts)


def _steps(steer, gas, brake, n=200, episodes=2):
    rows = []
    for ep in range(episodes):
        for t in range(n):
            rows.append({"episode": ep, "t": t, "steer": steer(t), "gas": gas(t), "brake": brake(t), "speed": 5.0, "tiles": t, "reward": 0.0})
    return pd.DataFrame(rows)


def _eps(reward=-30.0, steps=1000):
    return pd.DataFrame([{"episode": 0, "reward": reward, "steps": steps, "ended": "time_limit", "lap_finished": False,
                          "tiles_visited": 5, "tiles_total": 300, "coverage": 0.02}])


def test_collapsed_policy_is_flagged():
    st = _steps(lambda t: -0.01, lambda t: 0.49, lambda t: 0.5)
    notes = verdicts(summarize_actions(st), _eps())
    text = " ".join(notes)
    assert "POLICY COLLAPSE" in text and "GAS+BRAKE" in text and "NOT evidence of driving" in text


def test_healthy_policy_is_not_flagged_as_collapsed():
    st = _steps(lambda t: np.sin(t / 10) * 0.6, lambda t: 0.5 + 0.2 * np.cos(t / 7), lambda t: 0.0)
    notes = verdicts(summarize_actions(st), _eps(reward=600, steps=250))
    assert not any("COLLAPSE" in n or "NO STEERING" in n for n in notes)


def test_rollout_on_legacy_env_records_native_actions():
    venv = make_legacy_env()
    try:
        data = rollout_with_stats(_RandomModel(), venv, [3], max_steps=25)
    finally:
        venv.close()
    s, e = data["steps"], data["episodes"]
    assert len(s) == 25 and s.steer.between(-1, 1).all() and s.gas.between(0, 1).all() and s.brake.between(0, 1).all()
    assert e.tiles_total.iloc[0] > 50 and 0 <= e.coverage.iloc[0] <= 1
