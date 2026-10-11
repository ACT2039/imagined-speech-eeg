"""Configuration dataclasses and YAML/environment loader (Rules 4, 6, 8, 13)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from imagined_speech.exceptions import ConfigurationError

DEFAULT_EXPECTED_TASKS: tuple[str, ...] = ("Char", "Digit", "Image")

DEFAULT_KNOWN_EEG_CHANNELS: tuple[str, ...] = (
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
    "FP1", "FP2", "FZ", "CZ", "PZ", "OZ", "C3", "C4", "P3", "P4",
    "F1", "F2", "FC1", "FC2", "FC3", "FC4", "CP1", "CP2", "CP3", "CP4", "CP5", "CP6",
    "TP7", "TP8", "PO3", "PO4", "PO7", "PO8", "T3", "T4", "T5", "T6",
)

DEFAULT_KNOWN_NON_EEG_SUBSTRINGS: tuple[str, ...] = (
    "COUNTER", "INTERPOLATED", "RAW_CQ", "GYRO", "ACC", "MAG", "BATTERY",
    "MARKER", "TIMESTAMP", "SYSTEM", "FUNC_ID", "FUNC_VALUE", "SYNC",
    "STI", "TRIG", "STATUS", "ANNOT", "EDF ANNOTATIONS", "CQ_", "DEV_ID",
    "EOG", "ECG", "EKG", "EMG",
)


@dataclass(frozen=True)
class DiscoveryConfig:
    """Configuration for dataset directory discovery."""

    expected_tasks: tuple[str, ...] = DEFAULT_EXPECTED_TASKS
    allowed_extensions: tuple[str, ...] = (".edf",)
    allow_missing_task_dirs: bool = False
    recursive_within_task: bool = True

    def __post_init__(self) -> None:
        if not self.expected_tasks:
            raise ConfigurationError("expected_tasks must contain at least one task name.")
        if not self.allowed_extensions:
            raise ConfigurationError("allowed_extensions must contain at least one file extension.")


@dataclass(frozen=True)
class LabelParsingConfig:
    """Configuration for parsing and normalizing recording labels (Rules 11, 12)."""

    delimiters: tuple[str, ...] = ("_", "-")
    case_mode: str = "lower"
    extract_candidate_prefix: bool = True

    def __post_init__(self) -> None:
        valid_case_modes = {"lower", "upper", "preserve_case"}
        if self.case_mode not in valid_case_modes:
            raise ConfigurationError(
                f"Invalid case_mode {self.case_mode!r}. Expected one of {sorted(valid_case_modes)}."
            )
        if not self.delimiters:
            raise ConfigurationError("delimiters must contain at least one delimiter string.")


@dataclass(frozen=True)
class EDFAuditConfig:
    """Configuration for EDF header and in-memory signal validation (Rule 10)."""

    inspect_signal_properties: bool = True
    max_inspection_duration_sec: float | None = None
    known_eeg_channel_names: tuple[str, ...] = DEFAULT_KNOWN_EEG_CHANNELS
    known_non_eeg_substrings: tuple[str, ...] = DEFAULT_KNOWN_NON_EEG_SUBSTRINGS
    flat_std_threshold: float = 1.0e-12

    def __post_init__(self) -> None:
        if (
            self.max_inspection_duration_sec is not None
            and self.max_inspection_duration_sec <= 0
        ):
            raise ConfigurationError(
                "max_inspection_duration_sec must be positive when specified."
            )
        if self.flat_std_threshold < 0:
            raise ConfigurationError("flat_std_threshold must be non-negative.")


@dataclass(frozen=True)
class ProjectConfig:
    """Top-level configuration for the Imagined Speech Recognition project."""

    project_name: str = "imagined-speech-recognition"
    seed: int = 42
    log_level: str = "INFO"
    data_root: Path = field(default_factory=lambda: Path("./data/raw"))
    output_dir: Path = field(default_factory=lambda: Path("./outputs/phase1_audit"))
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    label_parsing: LabelParsingConfig = field(default_factory=LabelParsingConfig)
    edf_audit: EDFAuditConfig = field(default_factory=EDFAuditConfig)

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ConfigurationError(
                f"Project seed must be a non-negative integer, got {self.seed!r}."
            )


def load_config(
    config_path: Path | str | None = None,
    *,
    data_root: Path | str | None = None,
    output_dir: Path | str | None = None,
    seed: int | None = None,
    log_level: str | None = None,
) -> ProjectConfig:
    """Load project configuration from YAML, environment variables, and explicit overrides.

    Precedence order (highest to lowest):
    1. Explicit keyword arguments passed to ``load_config`` (e.g., from CLI flags).
    2. Environment variables (``ISR_DATA_ROOT``, ``ISR_OUTPUT_DIR``, ``ISR_SEED``, ``ISR_LOG_LEVEL``).
    3. YAML configuration file values (if ``config_path`` is provided).
    4. Dataclass defaults.

    Parameters
    ----------
    config_path : Path | str | None
        Optional path to a YAML configuration file.
    data_root : Path | str | None
        Optional override for the dataset root directory.
    output_dir : Path | str | None
        Optional override for the output directory.
    seed : int | None
        Optional override for the global random seed.
    log_level : str | None
        Optional override for logging verbosity.

    Returns
    -------
    ProjectConfig
        Validated immutable project configuration.
    """
    raw: dict[str, Any] = {}
    if config_path is not None:
        path = Path(config_path)
        if not path.is_file():
            raise ConfigurationError(f"Configuration file not found: {path}")
        try:
            with path.open("r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle)
                if loaded is not None:
                    if not isinstance(loaded, dict):
                        raise ConfigurationError(
                            f"Top-level YAML content must be a mapping, got {type(loaded).__name__}."
                        )
                    raw = loaded
        except yaml.YAMLError as exc:
            raise ConfigurationError(f"Failed to parse YAML configuration {path}: {exc}") from exc

    project_sec = raw.get("project", {}) or {}
    paths_sec = raw.get("paths", {}) or {}
    discovery_sec = raw.get("discovery", {}) or {}
    label_sec = raw.get("label_parsing", {}) or {}
    audit_sec = raw.get("edf_audit", {}) or {}

    env_data_root = os.environ.get("ISR_DATA_ROOT")
    env_output_dir = os.environ.get("ISR_OUTPUT_DIR")
    env_seed_raw = os.environ.get("ISR_SEED")
    env_log_level = os.environ.get("ISR_LOG_LEVEL")

    resolved_seed: int = project_sec.get("seed", 42)
    if env_seed_raw is not None:
        try:
            resolved_seed = int(env_seed_raw)
        except ValueError as exc:
            raise ConfigurationError(
                f"Environment variable ISR_SEED must be an integer, got {env_seed_raw!r}."
            ) from exc
    if seed is not None:
        resolved_seed = seed

    resolved_data_root = Path(
        data_root
        if data_root is not None
        else (env_data_root if env_data_root is not None else paths_sec.get("data_root", "./data/raw"))
    )
    resolved_output_dir = Path(
        output_dir
        if output_dir is not None
        else (
            env_output_dir
            if env_output_dir is not None
            else paths_sec.get("output_dir", "./outputs/phase1_audit")
        )
    )
    resolved_log_level = str(
        log_level
        if log_level is not None
        else (env_log_level if env_log_level is not None else project_sec.get("log_level", "INFO"))
    )

    discovery_cfg = DiscoveryConfig(
        expected_tasks=tuple(discovery_sec.get("expected_tasks", DEFAULT_EXPECTED_TASKS)),
        allowed_extensions=tuple(discovery_sec.get("allowed_extensions", (".edf",))),
        allow_missing_task_dirs=bool(discovery_sec.get("allow_missing_task_dirs", False)),
        recursive_within_task=bool(discovery_sec.get("recursive_within_task", True)),
    )

    label_cfg = LabelParsingConfig(
        delimiters=tuple(label_sec.get("delimiters", ("_", "-"))),
        case_mode=str(label_sec.get("case_mode", "lower")),
        extract_candidate_prefix=bool(label_sec.get("extract_candidate_prefix", True)),
    )

    audit_cfg = EDFAuditConfig(
        inspect_signal_properties=bool(audit_sec.get("inspect_signal_properties", True)),
        max_inspection_duration_sec=(
            float(audit_sec["max_inspection_duration_sec"])
            if audit_sec.get("max_inspection_duration_sec") is not None
            else None
        ),
        known_eeg_channel_names=tuple(
            audit_sec.get("known_eeg_channel_names", DEFAULT_KNOWN_EEG_CHANNELS)
        ),
        known_non_eeg_substrings=tuple(
            audit_sec.get("known_non_eeg_substrings", DEFAULT_KNOWN_NON_EEG_SUBSTRINGS)
        ),
        flat_std_threshold=float(audit_sec.get("flat_std_threshold", 1.0e-12)),
    )

    return ProjectConfig(
        project_name=str(project_sec.get("name", "imagined-speech-recognition")),
        seed=resolved_seed,
        log_level=resolved_log_level,
        data_root=resolved_data_root,
        output_dir=resolved_output_dir,
        discovery=discovery_cfg,
        label_parsing=label_cfg,
        edf_audit=audit_cfg,
    )
