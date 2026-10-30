"""Shared pytest fixtures. Tests use a tiny CPU config so the whole suite stays fast."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from src.config import Config, load_config  # noqa: E402


@pytest.fixture(scope="session")
def smoke_cfg() -> Config:
    return load_config(ROOT / "configs" / "sac_smoke.yaml")


@pytest.fixture()
def tmp_cfg(smoke_cfg: Config, tmp_path, monkeypatch) -> Config:
    """Smoke config whose outputs go to a temp dir (run_dirs() reads PROJECT_ROOT)."""
    import src.utils as utils

    monkeypatch.setattr(utils, "PROJECT_ROOT", tmp_path)
    cfg = load_config(ROOT / "configs" / "sac_smoke.yaml")
    cfg.experiment.name = "pytest_run"
    return cfg
