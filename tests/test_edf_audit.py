"""Unit tests for EDF header parsing, channel role validation, signal quality checks, and CLI (Rules 9, 10, 11, 12)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from imagined_speech.cli import main as cli_main
from imagined_speech.config import ProjectConfig
from imagined_speech.data.edf_audit import (
    detect_counter_like_signal,
    read_raw_edf_header,
    run_dataset_audit,
)
from imagined_speech.exceptions import EDFValidationError
from tests.conftest import write_synthetic_edf


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_detect_counter_like_signal() -> None:
    """detect_counter_like_signal identifies wrapping sample counters and ignores oscillatory EEG."""
    # Emotiv-style wrapping packet counter 0..127 repeated 4 times
    counter_sig = np.tile(np.arange(128, dtype=float), 4)
    assert detect_counter_like_signal(counter_sig) is True

    # Scaled to Volts (as MNE does when converting uV -> V)
    assert detect_counter_like_signal(counter_sig * 1e-6) is True

    # Realistic multi-frequency EEG-like sine + noise is not a counter
    rng = np.random.default_rng(42)
    t = np.linspace(0, 4.0, 512, endpoint=False)
    eeg_sig = 20.0 * np.sin(2 * np.pi * 10.0 * t) + rng.normal(0, 5.0, size=t.size)
    assert detect_counter_like_signal(eeg_sig) is False


def test_edf_audit_distinguishes_eeg_non_eeg_and_preserves_raw_files(
    tmp_path: Path,
) -> None:
    """run_dataset_audit validates channel roles/signals without assuming all MNE 'eeg' channels are EEG."""
    data_root = tmp_path / "raw_data"
    output_dir = tmp_path / "audit_out"
    for task in ("Char", "Digit", "Image"):
        (data_root / task).mkdir(parents=True)

    sfreq = 128
    n_samples = sfreq * 4  # 4 seconds
    rng = np.random.default_rng(7)
    t = np.linspace(0, 4.0, n_samples, endpoint=False)

    # Channel 0: AF3 (valid oscillatory EEG)
    ch_af3 = 25.0 * np.sin(2 * np.pi * 10.0 * t) + rng.normal(0, 2.0, n_samples)
    # Channel 1: F7 (valid oscillatory EEG)
    ch_f7 = 18.0 * np.cos(2 * np.pi * 12.0 * t) + rng.normal(0, 2.0, n_samples)
    # Channel 2: O1 (flatline / dead EEG electrode)
    ch_o1_flat = np.zeros(n_samples, dtype=float)
    # Channel 3: COUNTER (non-EEG wrapping packet counter 0..127)
    ch_counter = np.tile(np.arange(128, dtype=float), 4)
    # Channel 4: GYROX (non-EEG motion sensor)
    ch_gyrox = 2.0 * np.sin(2 * np.pi * 1.0 * t)
    # Channel 5: MYSTERY_CH (unknown channel name with oscillatory signal -> must require review)
    ch_mystery = 10.0 * np.sin(2 * np.pi * 8.0 * t)

    signals = np.vstack([ch_af3, ch_f7, ch_o1_flat, ch_counter, ch_gyrox, ch_mystery])
    ch_names = ["AF3", "F7", "O1", "COUNTER", "GYROX", "MYSTERY_CH"]

    edf_char = write_synthetic_edf(
        file_path=data_root / "Char" / "Sub01_A.edf",
        channel_names=ch_names,
        signals=signals,
        sfreq=sfreq,
        patient_id="X X X X",
    )
    edf_digit = write_synthetic_edf(
        file_path=data_root / "Digit" / "Sub01_5.edf",
        channel_names=ch_names,
        signals=signals,
        sfreq=sfreq,
        patient_id="X X X X",
    )
    # Write a corrupted/truncated EDF file in Image to verify graceful error reporting
    corrupt_edf = data_root / "Image" / "Sub01_Apple.edf"
    corrupt_edf.write_bytes(b"0       corrupted_short_file")

    hash_before = _sha256(edf_char)

    cfg = ProjectConfig(
        seed=42,
        data_root=data_root,
        output_dir=output_dir,
    )
    result = run_dataset_audit(cfg, save_outputs=True)

    # Rule 9: Raw EDF file must remain completely unchanged on disk
    assert _sha256(edf_char) == hash_before
    assert edf_digit.exists()

    # Check summary counts
    assert result.summary["n_discovered_files"] == 3
    assert result.summary["n_valid_edf_files"] == 2
    assert result.summary["n_invalid_edf_files"] == 1

    # Check manifest records
    by_rel = {r.relative_source_path: r for r in result.manifest_records}
    char_rec = by_rel["Char/Sub01_A.edf"]
    assert char_rec.is_valid_edf is True
    assert char_rec.original_filename == "Sub01_A.edf"
    assert char_rec.raw_label == "A"
    assert char_rec.normalized_label == "a"
    assert char_rec.unverified_candidate_prefix == "Sub01"
    assert char_rec.sfreq_hz == 128.0
    assert char_rec.duration_sec == 4.0
    assert char_rec.n_channels_total == 6
    # AF3, F7, O1 are candidate EEG names, but O1 is flat so only 2 are usable EEG
    assert char_rec.n_candidate_eeg_channels == 3
    assert char_rec.n_usable_eeg_channels == 2
    assert char_rec.n_flat_eeg_channels == 1
    # COUNTER and GYROX are non-EEG
    assert char_rec.n_non_eeg_channels == 2
    # MYSTERY_CH is flagged as unknown
    assert char_rec.n_unknown_channels == 1

    # Corrupt file record is preserved in manifest with is_valid_edf=False
    img_rec = by_rel["Image/Sub01_Apple.edf"]
    assert img_rec.is_valid_edf is False
    assert img_rec.validation_error is not None
    assert img_rec.raw_label == "Apple"
    assert img_rec.normalized_label == "apple"

    # Verify channel audit details for Char/Sub01_A.edf
    char_channels = {
        c.raw_channel_name: c
        for c in result.channel_records
        if c.relative_source_path == "Char/Sub01_A.edf"
    }
    assert char_channels["AF3"].audited_channel_role == "candidate_eeg"
    assert char_channels["AF3"].has_usable_eeg_signal is True

    assert char_channels["O1"].audited_channel_role == "candidate_eeg"
    assert char_channels["O1"].is_flat is True
    assert char_channels["O1"].has_usable_eeg_signal is False

    # Even though MNE assigns 'eeg' to COUNTER and GYROX, our audit rejects them as EEG (Rule 10)
    assert char_channels["COUNTER"].mne_assigned_type == "eeg"
    assert char_channels["COUNTER"].is_candidate_eeg is False
    assert char_channels["COUNTER"].is_counter_like is True
    assert "mne_misclassified_as_eeg" in char_channels["COUNTER"].audit_flags

    assert char_channels["GYROX"].is_candidate_eeg is False
    assert char_channels["MYSTERY_CH"].audited_channel_role == "unknown_needs_review"
    assert char_channels["MYSTERY_CH"].is_candidate_eeg is False

    # Verify saved CSVs and JSON exist and match
    manifest_df = pd.read_csv(output_dir / "manifest.csv")
    channel_df = pd.read_csv(output_dir / "channel_audit.csv")
    assert len(manifest_df) == 3
    assert len(channel_df) == 12  # 2 valid files * 6 channels


def test_read_raw_edf_header_rejects_truncated_file(tmp_path: Path) -> None:
    """read_raw_edf_header raises EDFValidationError on empty or truncated files."""
    empty_file = tmp_path / "empty.edf"
    empty_file.write_bytes(b"")
    with pytest.raises(EDFValidationError):
        read_raw_edf_header(empty_file)


def test_cli_end_to_end(tmp_path: Path) -> None:
    """CLI main() runs dataset discovery and audit and exits with status code 0."""
    data_root = tmp_path / "cli_raw"
    out_dir = tmp_path / "cli_out"
    for task in ("Char", "Digit", "Image"):
        (data_root / task).mkdir(parents=True)

    sfreq = 128
    signals = np.random.default_rng(0).normal(0, 10, size=(2, sfreq * 2))
    write_synthetic_edf(data_root / "Char" / "S1_A.edf", ["AF3", "F7"], signals, sfreq=sfreq)
    write_synthetic_edf(data_root / "Digit" / "S1_1.edf", ["AF3", "F7"], signals, sfreq=sfreq)
    write_synthetic_edf(data_root / "Image" / "S1_Cat.edf", ["AF3", "F7"], signals, sfreq=sfreq)

    exit_code = cli_main(
        [
            "--data-root",
            str(data_root),
            "--output-dir",
            str(out_dir),
            "--seed",
            "99",
        ]
    )
    assert exit_code == 0
    assert (out_dir / "manifest.csv").is_file()
    assert (out_dir / "channel_audit.csv").is_file()
    assert (out_dir / "audit_summary.json").is_file()
