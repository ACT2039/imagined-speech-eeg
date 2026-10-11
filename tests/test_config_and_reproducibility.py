"""Unit tests for configuration loading, environment overrides, and deterministic seeding."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from imagined_speech.config import LabelParsingConfig, ProjectConfig, load_config
from imagined_speech.exceptions import ConfigurationError
from imagined_speech.utils.reproducibility import set_global_seed


def test_default_config_loading() -> None:
    """Default ProjectConfig initializes with valid defaults."""
    cfg = load_config()
    assert cfg.seed == 42
    assert cfg.discovery.expected_tasks == ("Char", "Digit", "Image")
    assert cfg.label_parsing.case_mode == "lower"
    assert cfg.edf_audit.inspect_signal_properties is True


def test_load_config_from_yaml_and_env_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """YAML values are loaded and overridden by environment variables and explicit arguments."""
    yaml_path = tmp_path / "custom.yaml"
    yaml_path.write_text(
        """
project:
  name: "custom-isr"
  seed: 101
  log_level: "DEBUG"
paths:
  data_root: "./yaml_data"
  output_dir: "./yaml_out"
label_parsing:
  case_mode: "upper"
""",
        encoding="utf-8",
    )

    monkeypatch.setenv("ISR_DATA_ROOT", str(tmp_path / "env_data"))
    monkeypatch.setenv("ISR_SEED", "202")

    cfg = load_config(config_path=yaml_path)
    assert cfg.project_name == "custom-isr"
    assert cfg.seed == 202
    assert cfg.data_root == tmp_path / "env_data"
    assert cfg.output_dir == Path("./yaml_out")
    assert cfg.label_parsing.case_mode == "upper"

    # Explicit arguments override environment variables
    cfg_explicit = load_config(
        config_path=yaml_path,
        data_root=tmp_path / "cli_data",
        seed=303,
    )
    assert cfg_explicit.data_root == tmp_path / "cli_data"
    assert cfg_explicit.seed == 303


def test_invalid_config_raises_configuration_error(tmp_path: Path) -> None:
    """Invalid configuration values raise explicit ConfigurationError."""
    with pytest.raises(ConfigurationError):
        load_config(config_path=tmp_path / "nonexistent.yaml")

    with pytest.raises(ConfigurationError):
        ProjectConfig(seed=-5)

    with pytest.raises(ConfigurationError):
        LabelParsingConfig(case_mode="invalid_mode")


def test_set_global_seed_reproducibility() -> None:
    """set_global_seed produces identical random sequences across repeated calls."""
    rng1 = set_global_seed(12345)
    seq1_np_global = np.random.rand(5)
    seq1_rng = rng1.random(5)

    rng2 = set_global_seed(12345)
    seq2_np_global = np.random.rand(5)
    seq2_rng = rng2.random(5)

    np.testing.assert_allclose(seq1_np_global, seq2_np_global)
    np.testing.assert_allclose(seq1_rng, seq2_rng)

    with pytest.raises(ConfigurationError):
        set_global_seed(-1)
