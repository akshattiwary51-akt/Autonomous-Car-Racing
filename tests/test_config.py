import pytest

from src.config import ConfigError, apply_overrides, config_from_dict, load_config, save_config


def test_default_configs_load(smoke_cfg):
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "configs"
    for name in ("sac.yaml", "ppo.yaml", "sac_smoke.yaml"):
        cfg = load_config(root / name)
        assert cfg.env.id == "CarRacing-v3"
    assert load_config(root / "sac.yaml").algo.name == "sac"
    assert load_config(root / "ppo.yaml").algo.name == "ppo"


def test_unknown_key_is_rejected():
    with pytest.raises(ConfigError, match="Unknown key"):
        config_from_dict({"env": {"framestack": 4}})


@pytest.mark.parametrize("data, msg", [
    ({"algo": {"name": "dqn"}}, "algo.name"),
    ({"env": {"id": "CarRacing-v2"}}, "v2 was removed"),
    ({"env": {"frame_stack": 0}}, "frame_stack"),
    ({"env": {"resize": 8}}, "resize"),
    ({"algo": {"hyperparams": {"gamma": 1.5}}}, "gamma"),
    ({"algo": {"hyperparams": {"tau": 0}}}, "tau"),
    ({"algo": {"hyperparams": {"bogus": 1}}}, "Unsupported sac hyperparameter"),
    ({"training": {"total_timesteps": 0}}, "total_timesteps"),
    ({"experiment": {"name": "bad name!"}}, "experiment.name"),
    ({"experiment": {"device": "tpu"}}, "device"),
    ({"algo": {"policy": {"cnn_channels": [32], "cnn_kernels": [8, 4], "cnn_strides": [4]}}}, "equal length"),
])
def test_invalid_values_raise_clear_errors(data, msg):
    with pytest.raises(ConfigError, match=msg):
        config_from_dict(data)


def test_missing_file_message(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_roundtrip_and_overrides(tmp_path, smoke_cfg):
    save_config(smoke_cfg, tmp_path / "c.yaml")
    again = load_config(tmp_path / "c.yaml")
    assert again.to_dict() == smoke_cfg.to_dict()
    changed = apply_overrides(again, ["training.total_timesteps=123", "algo.hyperparams.batch_size=16",
                                      "env.reward_shaping.enabled=true"])
    assert changed.training.total_timesteps == 123
    assert changed.algo.hyperparams["batch_size"] == 16
    assert changed.env.reward_shaping.enabled is True
    with pytest.raises(ConfigError):
        apply_overrides(again, ["training.nope=1"])
    with pytest.raises(ConfigError):
        apply_overrides(again, ["no_equals_sign"])
