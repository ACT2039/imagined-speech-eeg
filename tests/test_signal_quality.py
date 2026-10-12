"""Unit tests for Phase 2 EEG signal-quality statistics, visualizations, and reports."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from imagined_speech.cli import main as cli_main
from imagined_speech.config import ProjectConfig, SignalQualityConfig
from imagined_speech.data.edf_audit import run_dataset_audit
from imagined_speech.data.signal_quality import (
    SelectedRecordingInput,
    compute_channel_signal_statistics,
    inspect_recording_signal_quality,
    load_and_validate_manifest,
    run_signal_quality_analysis,
    select_representative_recordings,
)
from imagined_speech.exceptions import SignalQualityError
from tests.conftest import write_synthetic_edf


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_compute_channel_signal_statistics_normal_oscillatory_signal() -> None:
    """Summary statistics accurately measure oscillatory signal properties without false flags."""
    sfreq = 128.0
    t = np.linspace(0.0, 4.0, int(4 * sfreq), endpoint=False)
    rng = np.random.default_rng(42)
    sig = 25.0 * np.sin(2.0 * np.pi * 10.0 * t) + rng.normal(0.0, 2.0, size=t.size)

    stat = compute_channel_signal_statistics(
        sig,
        sfreq_hz=sfreq,
        relative_source_path="Char/Sub01_A.edf",
        original_filename="Sub01_A.edf",
        task_category="Char",
        raw_label="A",
        normalized_label="a",
        unverified_candidate_prefix="Sub01",
        raw_channel_name="AF3",
        physical_unit="uV",
    )

    assert stat.n_samples == 512
    assert stat.sfreq_hz == 128.0
    assert stat.duration_sec == 4.0
    assert stat.n_non_finite == 0
    assert stat.n_finite == 512
    assert stat.physical_unit == "uV"
    assert stat.is_candidate_eeg is True
    assert stat.measured_min == pytest.approx(float(np.min(sig)))
    assert stat.measured_max == pytest.approx(float(np.max(sig)))
    assert stat.measured_mean == pytest.approx(float(np.mean(sig)))
    assert stat.measured_std == pytest.approx(float(np.std(sig)))
    assert stat.measured_amplitude_range_ptp == pytest.approx(float(np.ptp(sig)))
    assert stat.heuristic_has_non_finite is False
    assert stat.heuristic_is_flatline is False
    assert stat.heuristic_is_near_flatline is False
    assert stat.heuristic_potential_clipping is False
    assert stat.heuristic_suspicious_repeated_values is False
    assert stat.heuristic_is_counter_like is False
    assert stat.heuristic_quality_flags == ""


def test_compute_channel_signal_statistics_non_finite_values() -> None:
    """Non-finite samples (NaN, Inf) are counted and flagged while finite stats are preserved."""
    sig = np.array([10.0, -10.0, 5.0, -5.0, np.nan, np.inf, -np.inf, 0.0], dtype=float)
    stat = compute_channel_signal_statistics(
        sig,
        sfreq_hz=128.0,
        relative_source_path="Digit/Sub01_2.edf",
        original_filename="Sub01_2.edf",
        task_category="Digit",
        raw_channel_name="F7",
        physical_unit="uV",
    )

    assert stat.n_samples == 8
    assert stat.n_non_finite == 3
    assert stat.n_finite == 5
    assert stat.measured_min == pytest.approx(-10.0)
    assert stat.measured_max == pytest.approx(10.0)
    assert stat.measured_mean == pytest.approx(0.0)
    assert stat.heuristic_has_non_finite is True
    assert "non_finite_samples_present" in stat.heuristic_quality_flags

    # All non-finite signal returns None for measured amplitude metrics without crashing
    all_nan = np.full(32, np.nan)
    nan_stat = compute_channel_signal_statistics(
        all_nan,
        sfreq_hz=128.0,
        relative_source_path="Digit/Sub01_2.edf",
        original_filename="Sub01_2.edf",
        task_category="Digit",
        raw_channel_name="F7",
    )
    assert nan_stat.n_finite == 0
    assert nan_stat.n_non_finite == 32
    assert nan_stat.measured_min is None
    assert nan_stat.heuristic_has_non_finite is True
    assert "all_samples_non_finite" in nan_stat.heuristic_quality_flags


def test_compute_channel_signal_statistics_flatline_and_near_flatline() -> None:
    """Flatline and near-flatline signals are distinguished accurately."""
    flat_sig = np.full(256, 42.0, dtype=float)
    flat_stat = compute_channel_signal_statistics(
        flat_sig,
        sfreq_hz=128.0,
        relative_source_path="Char/Sub01_B.edf",
        original_filename="Sub01_B.edf",
        task_category="Char",
        raw_channel_name="O1",
    )
    assert flat_stat.measured_std == pytest.approx(0.0)
    assert flat_stat.measured_amplitude_range_ptp == pytest.approx(0.0)
    assert flat_stat.measured_max_constant_run_samples == 256
    assert flat_stat.measured_max_constant_run_fraction == pytest.approx(1.0)
    assert flat_stat.heuristic_is_flatline is True
    assert flat_stat.heuristic_is_near_flatline is False
    assert "flatline" in flat_stat.heuristic_quality_flags

    # Near-flatline: non-zero variance but contains a long constant dropout run (>25% of recording)
    rng = np.random.default_rng(9)
    near_flat = rng.normal(0.0, 15.0, size=256)
    near_flat[50:150] = 0.0  # 100 consecutive identical samples (39% of signal)
    nf_stat = compute_channel_signal_statistics(
        near_flat,
        sfreq_hz=128.0,
        relative_source_path="Char/Sub01_B.edf",
        original_filename="Sub01_B.edf",
        task_category="Char",
        raw_channel_name="O2",
    )
    assert nf_stat.heuristic_is_flatline is False
    assert nf_stat.heuristic_is_near_flatline is True
    assert nf_stat.measured_max_constant_run_samples >= 100
    assert "near_flatline" in nf_stat.heuristic_quality_flags


def test_compute_channel_signal_statistics_clipping_and_repeated_values() -> None:
    """Clipped/saturated signals and heavily quantized signals trigger heuristic indicators."""
    t = np.linspace(0.0, 2.0, 256, endpoint=False)
    clipped = np.clip(100.0 * np.sin(2.0 * np.pi * 4.0 * t), -40.0, 40.0)

    clip_stat = compute_channel_signal_statistics(
        clipped,
        sfreq_hz=128.0,
        relative_source_path="Image/Sub01_Cat.edf",
        original_filename="Sub01_Cat.edf",
        task_category="Image",
        raw_channel_name="T7",
        edf_physical_min=-40.0,
        edf_physical_max=40.0,
    )
    assert clip_stat.measured_extreme_rail_fraction > 0.15
    assert clip_stat.heuristic_potential_clipping is True
    assert "potential_clipping_or_rail_saturation" in clip_stat.heuristic_quality_flags

    # Invalid sfreq raises SignalQualityError
    with pytest.raises(SignalQualityError):
        compute_channel_signal_statistics(
            clipped,
            sfreq_hz=0.0,
            relative_source_path="Image/Sub01_Cat.edf",
            original_filename="Sub01_Cat.edf",
            task_category="Image",
        )


def test_select_representative_recordings_and_manifest_validation(tmp_path: Path) -> None:
    """Representative selection is deterministic, spans labels/durations, and validates inputs."""
    for task in ("Char", "Digit", "Image"):
        (tmp_path / task).mkdir()

    rows = []
    for task, labels in (
        ("Char", ["a", "b", "c", "d"]),
        ("Digit", ["1", "2", "3", "4"]),
        ("Image", ["apple", "car", "dog", "tree"]),
    ):
        for idx, lbl in enumerate(labels):
            fname = f"Sub01_{lbl}.edf"
            rel = f"{task}/{fname}"
            (tmp_path / rel).write_bytes(b"placeholder")
            rows.append(
                {
                    "original_filename": fname,
                    "relative_source_path": rel,
                    "task_category": task,
                    "raw_label": lbl,
                    "normalized_label": lbl,
                    "unverified_candidate_prefix": "Sub01",
                    "is_valid_edf": True,
                    "duration_sec": float(2 + idx * 2),
                }
            )

    manifest_df = pd.DataFrame(rows)
    sel_1 = select_representative_recordings(
        manifest_df,
        tmp_path,
        recordings_per_task=2,
        specific_recordings=("Char/Sub01_c.edf",),
        seed=42,
    )
    sel_2 = select_representative_recordings(
        manifest_df,
        tmp_path,
        recordings_per_task=2,
        specific_recordings=("Char/Sub01_c.edf",),
        seed=42,
    )
    assert [s.relative_source_path for s in sel_1] == [s.relative_source_path for s in sel_2]
    assert sel_1[0].relative_source_path == "Char/Sub01_c.edf"
    assert sel_1[0].selection_reason == "explicit_user_request"

    # Missing specific recording raises SignalQualityError
    with pytest.raises(SignalQualityError, match="not found in the manifest"):
        select_representative_recordings(
            manifest_df,
            tmp_path,
            specific_recordings=("Char/NonExistent.edf",),
        )

    # Missing required columns in manifest raises SignalQualityError
    with pytest.raises(SignalQualityError, match="missing required columns"):
        load_and_validate_manifest(pd.DataFrame({"original_filename": ["a.edf"]}))


def test_inspect_recording_reports_missing_channels_and_preserves_native_units(
    tmp_path: Path,
) -> None:
    """Missing expected EEG channels are reported and MNE SI scaling is inverted to native uV."""
    sfreq = 128
    t = np.linspace(0.0, 2.0, sfreq * 2, endpoint=False)
    ch_af3 = 30.0 * np.sin(2.0 * np.pi * 8.0 * t)
    ch_f7 = 15.0 * np.cos(2.0 * np.pi * 12.0 * t)
    signals = np.vstack([ch_af3, ch_f7])

    edf_path = write_synthetic_edf(
        file_path=tmp_path / "Char" / "Sub01_A.edf",
        channel_names=["AF3", "F7"],
        signals=signals,
        sfreq=sfreq,
        physical_dims=["uV", "uV"],
    )

    rec_input = SelectedRecordingInput(
        original_filename="Sub01_A.edf",
        relative_source_path="Char/Sub01_A.edf",
        absolute_path=edf_path,
        task_category="Char",
        raw_label="A",
        normalized_label="a",
        unverified_candidate_prefix="Sub01",
        selection_reason="unit_test",
    )

    sq_cfg = SignalQualityConfig(
        expected_eeg_channels=("AF3", "F7", "O1", "O2"),
        generate_plots=False,
    )
    insp, native_matrix, units = inspect_recording_signal_quality(rec_input, sq_config=sq_cfg)

    assert insp.is_readable is True
    assert insp.missing_expected_eeg_channels == ("O1", "O2")
    assert units == ["uV", "uV"]
    assert native_matrix is not None

    # Verify that native uV amplitudes (~30 uV peak) are preserved, not scaled to 3e-5 Volts
    af3_stat = insp.channel_statistics[0]
    assert af3_stat.physical_unit == "uV"
    assert af3_stat.measured_max == pytest.approx(30.0, abs=0.2)
    assert af3_stat.measured_min == pytest.approx(-30.0, abs=0.2)


def test_run_signal_quality_pipeline_and_cli_end_to_end(tmp_path: Path) -> None:
    """End-to-end Phase 2 pipeline writes CSV, JSON, Markdown, and PNG plots while preserving raw EDFs."""
    data_root = tmp_path / "raw"
    audit_dir = tmp_path / "phase1_audit"
    sq_dir = tmp_path / "phase2_sq"

    for task in ("Char", "Digit", "Image"):
        (data_root / task).mkdir(parents=True)

    sfreq = 128
    t = np.linspace(0.0, 2.0, sfreq * 2, endpoint=False)
    rng = np.random.default_rng(123)
    sig_af3 = 20.0 * np.sin(2.0 * np.pi * 10.0 * t) + rng.normal(0.0, 1.5, size=t.size)
    sig_t7 = 12.0 * np.cos(2.0 * np.pi * 14.0 * t) + rng.normal(0.0, 1.5, size=t.size)
    sig_o1_flat = np.zeros_like(t)
    signals = np.vstack([sig_af3, sig_t7, sig_o1_flat])

    edf_char = write_synthetic_edf(
        data_root / "Char" / "Sub01_A.edf",
        ["AF3", "T7", "O1"],
        signals,
        sfreq=sfreq,
    )
    write_synthetic_edf(
        data_root / "Digit" / "Sub01_7.edf",
        ["AF3", "T7", "O1"],
        signals,
        sfreq=sfreq,
    )
    write_synthetic_edf(
        data_root / "Image" / "Sub01_Apple.edf",
        ["AF3", "T7", "O1"],
        signals,
        sfreq=sfreq,
    )
    # Add an invalid file in Image to ensure graceful error handling
    (data_root / "Image" / "Sub01_Corrupt.edf").write_bytes(b"corrupt_edf_bytes")

    hash_before = _sha256(edf_char)

    phase1_cfg = ProjectConfig(seed=42, data_root=data_root, output_dir=audit_dir)
    run_dataset_audit(phase1_cfg, save_outputs=True)

    # Also simulate a file disappearing after manifest creation to test unreadable selected file handling
    manifest_df = pd.read_csv(audit_dir / "manifest.csv")
    sq_cfg = ProjectConfig(
        seed=42,
        data_root=data_root,
        output_dir=sq_dir,
        signal_quality=SignalQualityConfig(recordings_per_task=2, generate_plots=True),
    )

    result = run_signal_quality_analysis(
        config=sq_cfg,
        manifest_source=manifest_df,
        output_dir=sq_dir,
        save_outputs=True,
    )

    # Raw EDF file on disk is strictly unchanged
    assert _sha256(edf_char) == hash_before

    # Verify required output files
    assert (sq_dir / "signal_statistics.csv").is_file()
    assert (sq_dir / "signal_quality_summary.json").is_file()
    assert (sq_dir / "signal_quality_report.md").is_file()
    assert (sq_dir / "figures").is_dir()
    assert len(result.generated_figures) >= 5

    # Verify CLI --mode signal-quality execution
    cli_out_dir = tmp_path / "cli_sq_out"
    exit_code = cli_main(
        [
            "--mode",
            "signal-quality",
            "--data-root",
            str(data_root),
            "--manifest",
            str(audit_dir / "manifest.csv"),
            "--output-dir",
            str(cli_out_dir),
            "--recordings-per-task",
            "1",
            "--seed",
            "42",
        ]
    )
    assert exit_code == 0
    assert (cli_out_dir / "signal_statistics.csv").is_file()
    assert (cli_out_dir / "signal_quality_summary.json").is_file()
    assert (cli_out_dir / "signal_quality_report.md").is_file()
