"""Unit tests for Phase 3B reusable EEG preprocessing pipeline and QC screening."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from imagined_speech.config import (
    DEFAULT_PRIMARY_EEG_MONTAGE,
    PreprocessingConfig,
    load_config,
)
from imagined_speech.data.edf_audit import run_dataset_audit
from imagined_speech.data.preprocessing import (
    SCREENING_NOTE_TEXT,
    extract_and_validate_eeg_channels,
    load_preprocessed_recording,
    preprocess_eeg_array,
    preprocess_single_recording,
    run_phase3b_preprocessing,
    validate_boundary_behavior,
    verify_phase3b_outputs,
)
from imagined_speech.data.preprocessing_diagnostic import run_phase3a_diagnostic
from imagined_speech.data.signal_quality import run_signal_quality_analysis
from imagined_speech.exceptions import (
    ConfigurationError,
    EDFValidationError,
    PreprocessingError,
)
from tests.conftest import write_synthetic_edf


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_channel_ordering_and_auxiliary_exclusion(tmp_path: Path) -> None:
    """Verify that candidate EEG channels are reordered to canonical montage and auxiliary channels are excluded."""
    sfreq = 128
    n_samples = sfreq * 12
    t = np.arange(n_samples, dtype=np.float64) / sfreq

    # Create distinct identifiable signals for the 14 EEG channels
    canonical_14 = list(DEFAULT_PRIMARY_EEG_MONTAGE)
    eeg_signals = {
        name: 4230.0 + (idx + 1) * 2.0 * np.sin(2.0 * np.pi * 10.0 * t)
        for idx, name in enumerate(canonical_14)
    }

    # Interleave auxiliary channels and scramble the 14 EEG channel order
    scrambled_eeg = list(reversed(canonical_14))
    all_names = [
        "COUNTER",
        *scrambled_eeg[:5],
        "GYROX",
        *scrambled_eeg[5:10],
        "TIMESTAMP",
        *scrambled_eeg[10:],
        "INTERPOLATED",
    ]
    aux_signals = {
        "COUNTER": np.linspace(0, 1000, n_samples),
        "GYROX": np.cos(2.0 * np.pi * 1.0 * t),
        "TIMESTAMP": np.arange(n_samples, dtype=float),
        "INTERPOLATED": np.zeros(n_samples),
    }

    combined_matrix = np.vstack([
        eeg_signals[name] if name in eeg_signals else aux_signals[name]
        for name in all_names
    ])

    edf_path = write_synthetic_edf(
        file_path=tmp_path / "Char" / "Sub01_scrambled.edf",
        channel_names=all_names,
        signals=combined_matrix,
        sfreq=sfreq,
    )

    extracted = extract_and_validate_eeg_channels(edf_path)

    # 1. Output order must exactly match canonical 14 montage
    assert extracted.channel_names == DEFAULT_PRIMARY_EEG_MONTAGE
    assert extracted.eeg_signals_uV.shape == (14, n_samples)
    assert extracted.sfreq_hz == 128.0
    assert extracted.duration_sec == 12.0

    # 2. Auxiliary channels must be completely excluded from signal array
    assert set(extracted.excluded_auxiliary_channels) == {
        "COUNTER",
        "GYROX",
        "TIMESTAMP",
        "INTERPOLATED",
    }

    # 3. Waveform for each canonical channel must match its original identity (within 16-bit EDF quantization)
    for idx, name in enumerate(DEFAULT_PRIMARY_EEG_MONTAGE):
        np.testing.assert_allclose(
            extracted.eeg_signals_uV[idx],
            eeg_signals[name],
            rtol=1e-4,
            atol=1e-2,
        )

    # 4. Error case: Missing expected EEG channel
    missing_edf = write_synthetic_edf(
        file_path=tmp_path / "Char" / "Sub01_missing.edf",
        channel_names=canonical_14[:-1],  # missing AF4
        signals=np.vstack([eeg_signals[n] for n in canonical_14[:-1]]),
        sfreq=sfreq,
    )
    with pytest.raises(PreprocessingError, match="missing expected candidate EEG channels"):
        extract_and_validate_eeg_channels(missing_edf)

    # 5. Error case: Duplicate expected EEG channel
    dup_names = ["AF3", *canonical_14]  # duplicate AF3
    dup_matrix = np.vstack([eeg_signals["AF3"], *(eeg_signals[n] for n in canonical_14)])
    dup_edf = write_synthetic_edf(
        file_path=tmp_path / "Char" / "Sub01_dup.edf",
        channel_names=dup_names,
        signals=dup_matrix,
        sfreq=sfreq,
    )
    with pytest.raises(PreprocessingError, match="duplicate candidate EEG channels"):
        extract_and_validate_eeg_channels(dup_edf)


def test_calibration_and_physical_units(tmp_path: Path) -> None:
    """Verify that MNE SI Volts scaling is inverted to native uV and non-uV units raise PreprocessingError."""
    sfreq = 128
    n_samples = sfreq * 11
    t = np.arange(n_samples, dtype=np.float64) / sfreq
    canonical_14 = list(DEFAULT_PRIMARY_EEG_MONTAGE)

    # Signal in native uV with ~4230 uV DC offset + 15 uV alpha wave
    sig_uV = 4230.0 + 15.0 * np.sin(2.0 * np.pi * 10.0 * t)
    matrix = np.vstack([sig_uV] * 14)

    edf_uv = write_synthetic_edf(
        file_path=tmp_path / "Digit" / "Sub01_uv.edf",
        channel_names=canonical_14,
        signals=matrix,
        sfreq=sfreq,
        physical_dims=["uV"] * 14,
    )
    extracted = extract_and_validate_eeg_channels(edf_uv)
    assert extracted.physical_units == tuple(["uV"] * 14)
    # Native mean must be ~4230 uV (not 0.004230 V)
    assert np.isclose(float(np.mean(extracted.eeg_signals_uV[0])), 4230.0, atol=1.0)

    # Non-standard physical dimension (e.g. "mV" or "V") raises PreprocessingError
    edf_mv = write_synthetic_edf(
        file_path=tmp_path / "Digit" / "Sub01_mv.edf",
        channel_names=canonical_14,
        signals=matrix,
        sfreq=sfreq,
        physical_dims=["mV"] * 14,
    )
    with pytest.raises(PreprocessingError, match="unsupported physical unit"):
        extract_and_validate_eeg_channels(edf_mv)


@pytest.mark.parametrize("duration_sec", [11, 12, 15, 22])
def test_filtering_and_configurable_parameters(duration_sec: int) -> None:
    """Verify per-channel mean subtraction + 4th-order Butterworth 0.5-45 Hz bandpass across durations."""
    sfreq = 128.0
    n_samples = int(sfreq * duration_sec)
    t = np.arange(n_samples, dtype=np.float64) / sfreq

    # 4230 uV DC offset + 12 uV 10 Hz alpha wave (in-band) + 6 uV 58 Hz noise (out-of-band)
    alpha = 12.0 * np.sin(2.0 * np.pi * 10.0 * t)
    noise_58 = 6.0 * np.sin(2.0 * np.pi * 58.0 * t)
    raw = 4230.0 + alpha + noise_58
    signals_2d = np.vstack([raw, raw * 0.5])
    raw_copy = signals_2d.copy()

    prep_res = preprocess_eeg_array(signals_2d, sfreq_hz=sfreq)

    # 1. Caller array must never be mutated
    np.testing.assert_array_equal(signals_2d, raw_copy)

    # 2. Subtracted DC mean must match ~4230 uV
    assert np.isclose(prep_res.subtracted_channel_means_uV[0], 4230.0, atol=0.1)

    # 3. Post-filter residual mean must be near 0
    assert abs(float(np.mean(prep_res.preprocessed_signals_uV[0]))) < 0.01

    # 4. Interior samples must preserve 10 Hz alpha while attenuating 58 Hz noise
    interior = slice(128, n_samples - 128)
    residual = prep_res.preprocessed_signals_uV[0, interior] - alpha[interior]
    assert float(np.std(residual)) < 0.05

    # 5. Configurable options: apply_mean_subtraction=False preserves mean
    cfg_no_mean = PreprocessingConfig(apply_mean_subtraction=False, expected_sfreq_hz=sfreq)
    res_no_mean = preprocess_eeg_array(signals_2d, sfreq_hz=sfreq, preprocessing_config=cfg_no_mean)
    assert res_no_mean.subtracted_channel_means_uV[0] == 0.0

    # 6. Configurable options: reference_mode='none' vs 'car'
    cfg_car = PreprocessingConfig(reference_mode="car", expected_sfreq_hz=sfreq)
    res_car = preprocess_eeg_array(signals_2d, sfreq_hz=sfreq, preprocessing_config=cfg_car)
    assert res_car.filter_metadata.car_applied is True
    # Instantaneous mean across channels after CAR is ~0 at every sample
    np.testing.assert_allclose(np.mean(res_car.preprocessed_signals_uV, axis=0), 0.0, atol=1e-11)


def test_boundary_behavior_and_padding_metadata() -> None:
    """Verify symmetric reflection padding metadata and boundary vs. interior validation."""
    sfreq = 128.0
    # 12-second signal: 1536 samples > 768 samples -> padlen is not capped
    signals_12s = np.zeros((14, int(sfreq * 12)), dtype=np.float64)
    res_12s = preprocess_eeg_array(signals_12s, sfreq_hz=sfreq)
    f_meta_12s = res_12s.filter_metadata
    assert f_meta_12s.filter_padtype == "even"
    assert f_meta_12s.target_padlen_samples == 768
    assert f_meta_12s.effective_padlen_samples == 768
    assert f_meta_12s.padlen_was_capped is False

    # Short 4-second signal (512 samples < 768 samples) -> padlen is capped at n_samples - 1 = 511
    signals_4s = np.zeros((14, int(sfreq * 4)), dtype=np.float64)
    res_4s = preprocess_eeg_array(signals_4s, sfreq_hz=sfreq)
    f_meta_4s = res_4s.filter_metadata
    assert f_meta_4s.effective_padlen_samples == 511
    assert f_meta_4s.padlen_was_capped is True

    # Shortest 11-second recording in dataset (1408 samples at 128 Hz):
    # n_samples - 1 = 1407 >= 768 -> receives full 6.0 s (768 samples) padding without capping
    signals_11s = np.zeros((14, int(sfreq * 11)), dtype=np.float64)
    res_11s = preprocess_eeg_array(signals_11s, sfreq_hz=sfreq)
    f_meta_11s = res_11s.filter_metadata
    assert f_meta_11s.target_padlen_samples == 768
    assert f_meta_11s.effective_padlen_samples == 768
    assert f_meta_11s.effective_padlen_sec == 6.0
    assert f_meta_11s.padlen_was_capped is False

    # Boundary validation helper metrics on regular sinusoid (ratio ~ 1.0)
    t = np.linspace(0, 10, int(10 * sfreq), endpoint=False)
    oscillatory = 15.0 * np.sin(2.0 * np.pi * 8.0 * t)
    b_vals = validate_boundary_behavior(oscillatory[np.newaxis, :], sfreq_hz=sfreq)
    assert len(b_vals) == 1
    assert b_vals[0].boundary_window_samples == 128
    assert b_vals[0].boundary_all_finite is True
    assert np.isclose(b_vals[0].boundary_to_interior_std_ratio, 1.0, atol=0.1)

    # Unusual boundary-to-interior ratio (e.g. onset blink or transient burst in the first second)
    # is captured faithfully as a diagnostic metric without raising an error or claiming filter failure
    burst_signal = oscillatory.copy()
    burst_signal[:128] += 40.0 * np.sin(2.0 * np.pi * 3.0 * t[:128])
    b_vals_burst = validate_boundary_behavior(burst_signal[np.newaxis, :], sfreq_hz=sfreq)
    assert b_vals_burst[0].boundary_all_finite is True
    assert b_vals_burst[0].boundary_to_interior_std_ratio > 1.8


def test_post_filter_qc_screening_flags_and_preservation(tmp_path: Path) -> None:
    """Verify that exploratory screening flags (<0.5 uV, >1000 uV) are computed without rejecting data."""
    sfreq = 128
    n_samples = sfreq * 11
    t = np.arange(n_samples, dtype=np.float64) / sfreq
    canonical_14 = list(DEFAULT_PRIMARY_EEG_MONTAGE)

    # Channel 0: normal oscillatory signal (std ~ 10 uV, ptp ~ 28 uV) -> unflagged
    ch0 = 4230.0 + 14.0 * np.sin(2.0 * np.pi * 10.0 * t)
    # Channel 1: near-flatline signal (std ~ 0.1 uV < 0.5 uV) -> low_std flag
    ch1 = 4230.0 + 0.1 * np.sin(2.0 * np.pi * 10.0 * t)
    # Channel 2: high amplitude excursion (ptp ~ 1400 uV > 1000 uV) -> high_ptp flag
    ch2 = 4230.0 + 700.0 * np.sin(2.0 * np.pi * 10.0 * t)

    signals_list = [ch0, ch1, ch2] + [ch0] * 11
    matrix = np.vstack(signals_list)

    edf_path = write_synthetic_edf(
        file_path=tmp_path / "Char" / "Sub01_qc.edf",
        channel_names=canonical_14,
        signals=matrix,
        sfreq=sfreq,
    )

    manifest_row = {
        "relative_source_path": "Char/Sub01_qc.edf",
        "original_filename": "Sub01_qc.edf",
        "task_category": "Char",
        "raw_label": "a",
        "normalized_label": "a",
        "unverified_candidate_prefix": "Sub01",
    }

    out_dir = tmp_path / "phase3b_out"
    rec_manifest, qc_records, prep_res = preprocess_single_recording(
        edf_path=edf_path,
        manifest_row=manifest_row,
        output_dir=out_dir,
        save_npz=True,
    )

    # 1. Output recording must be retained in outputs (all 690 policy)
    assert rec_manifest.recording_retained_in_outputs is True
    assert rec_manifest.n_eeg_channels == 14
    assert prep_res.preprocessed_signals_uV.shape == (14, n_samples)

    # 2. Check channel 0: unflagged
    assert qc_records[0].screening_flag_any is False
    assert qc_records[0].screening_status == "unflagged_under_screening_rule"

    # 3. Check channel 1: low std flag
    assert qc_records[1].screening_flag_low_std is True
    assert qc_records[1].screening_flag_any is True
    assert "low_post_filter_std" in qc_records[1].screening_flag_reasons

    # 4. Check channel 2: high ptp flag
    assert qc_records[2].screening_flag_high_ptp is True
    assert qc_records[2].screening_flag_any is True
    assert "high_post_filter_ptp" in qc_records[2].screening_flag_reasons

    # 5. Recording summary counts
    assert rec_manifest.n_screening_flagged_channels == 2
    assert rec_manifest.is_unflagged_allow_0 is False
    assert rec_manifest.is_unflagged_allow_1 is False
    assert rec_manifest.is_unflagged_allow_2 is True
    assert qc_records[0].screening_note == SCREENING_NOTE_TEXT


def test_error_handling_and_safeguards(tmp_path: Path) -> None:
    """Verify explicit error handling for sampling rate mismatches, corrupt files, and unsafe output paths."""
    sfreq = 128
    t = np.arange(sfreq * 11, dtype=float) / sfreq
    matrix = np.vstack([4230.0 + 10.0 * np.sin(2.0 * np.pi * 8.0 * t)] * 14)
    edf_path = write_synthetic_edf(
        file_path=tmp_path / "raw" / "Char" / "Sub01_A.edf",
        channel_names=list(DEFAULT_PRIMARY_EEG_MONTAGE),
        signals=matrix,
        sfreq=sfreq,
    )

    # 1. Sampling rate mismatch
    cfg_wrong_sfreq = PreprocessingConfig(expected_sfreq_hz=256.0)
    with pytest.raises(PreprocessingError, match="has sfreq_hz=128.0, expected 256.0"):
        extract_and_validate_eeg_channels(edf_path, preprocessing_config=cfg_wrong_sfreq)

    # 2. Corrupt EDF file
    corrupt_path = tmp_path / "raw" / "Char" / "Sub01_corrupt.edf"
    corrupt_path.write_bytes(b"invalid_edf_header_bytes")
    with pytest.raises(EDFValidationError):
        extract_and_validate_eeg_channels(corrupt_path)

    # 3. Signal length too short
    with pytest.raises(PreprocessingError, match="too short for zero-phase filtering"):
        preprocess_eeg_array(np.ones((14, 20)), sfreq_hz=128.0)

    # 4. Refuse writing Phase 3B outputs inside raw data_root
    cfg_project = load_config(data_root=tmp_path / "raw", output_dir=tmp_path / "raw")
    manifest_df = pd.DataFrame([
        {
            "relative_source_path": "Char/Sub01_A.edf",
            "original_filename": "Sub01_A.edf",
            "task_category": "Char",
            "raw_label": "a",
            "normalized_label": "a",
            "unverified_candidate_prefix": "Sub01",
            "is_valid_edf": True,
        }
    ])
    with pytest.raises(PreprocessingError, match="Refusing to write Phase 3B outputs inside raw data_root"):
        run_phase3b_preprocessing(cfg_project, manifest_source=manifest_df, output_dir=tmp_path / "raw")

    # 5. Invalid PreprocessingConfig parameters
    with pytest.raises(ConfigurationError):
        PreprocessingConfig(expected_eeg_channels=())
    with pytest.raises(ConfigurationError):
        PreprocessingConfig(bandpass_low_hz=50.0, bandpass_high_hz=10.0)
    with pytest.raises(ConfigurationError):
        PreprocessingConfig(reference_mode="invalid_mode")


def test_end_to_end_phase3b_smoke_and_provenance_preservation(tmp_path: Path) -> None:
    """End-to-end test verifying smoke test, full run, NPZ round-trip, and preservation of earlier phases."""
    data_root = tmp_path / "raw"
    p1_dir = tmp_path / "phase1_audit"
    p2_dir = tmp_path / "phase2_signal_quality"
    p3a_dir = tmp_path / "phase3a_diagnostic"
    smoke_dir = tmp_path / "phase3b_smoke"
    full_dir = tmp_path / "phase3b_preprocessing"

    eeg_names = list(DEFAULT_PRIMARY_EEG_MONTAGE)
    rng = np.random.default_rng(42)

    # Create 6 synthetic recordings across Char, Digit, Image
    for task, lbl in (("Char", "A"), ("Digit", "1"), ("Image", "Up")):
        t_dir = data_root / task
        t_dir.mkdir(parents=True, exist_ok=True)
        for prefix in ("S01", "S02"):
            signals = 4230.0 + 15.0 * rng.standard_normal((len(eeg_names), 128 * 12))
            write_synthetic_edf(
                t_dir / f"{prefix}_{task}_{lbl}.edf",
                channel_names=eeg_names,
                signals=signals,
                sfreq=128,
            )

    cfg = load_config(data_root=data_root, output_dir=p1_dir, seed=42)
    run_dataset_audit(cfg, save_outputs=True)
    run_signal_quality_analysis(
        cfg,
        manifest_source=p1_dir / "manifest.csv",
        output_dir=p2_dir,
        save_outputs=True,
    )
    run_phase3a_diagnostic(
        cfg,
        manifest_source=p1_dir / "manifest.csv",
        output_dir=p3a_dir,
        save_outputs=True,
    )

    # Snapshot files before Phase 3B
    raw_files_before = {p.as_posix(): _sha256(p) for p in data_root.rglob("*.edf")}
    p1_files_before = {p.as_posix(): _sha256(p) for p in p1_dir.glob("*") if p.is_file()}
    p2_files_before = {p.as_posix(): _sha256(p) for p in p2_dir.glob("*") if p.is_file()}
    p3a_files_before = {p.as_posix(): _sha256(p) for p in p3a_dir.glob("*") if p.is_file()}

    manifest_df = pd.read_csv(p1_dir / "manifest.csv")

    # 1. Smoke test on 3 recordings (1 per task)
    smoke_paths = [
        "Char/S01_Char_A.edf",
        "Digit/S01_Digit_1.edf",
        "Image/S01_Image_Up.edf",
    ]
    smoke_res = run_phase3b_preprocessing(
        config=cfg,
        manifest_source=manifest_df,
        output_dir=smoke_dir,
        selected_relative_paths=smoke_paths,
        save_outputs=True,
        verify_completeness=True,
    )
    assert smoke_res.summary["n_recordings_processed"] == 3
    assert len(smoke_res.manifest_df) == 3
    assert len(smoke_res.channel_qc_df) == 3 * 14
    assert len(smoke_res.generated_npz_files) == 3

    # Validate smoke NPZ loading
    sample_npz = Path(smoke_res.generated_npz_files[0])
    loaded = load_preprocessed_recording(sample_npz)
    assert loaded.eeg_signals_uV.shape == (14, 128 * 12)
    assert loaded.channel_names == DEFAULT_PRIMARY_EEG_MONTAGE
    assert loaded.sfreq_hz == 128.0
    assert loaded.metadata["relative_source_path"] == "Char/S01_Char_A.edf"

    # 2. Full Phase 3B run across all 6 recordings
    full_res = run_phase3b_preprocessing(
        config=cfg,
        manifest_source=manifest_df,
        output_dir=full_dir,
        save_outputs=True,
        verify_completeness=True,
    )
    assert full_res.summary["n_recordings_processed"] == 6
    assert len(full_res.manifest_df) == 6
    assert len(full_res.channel_qc_df) == 6 * 14
    assert len(full_res.generated_npz_files) == 6
    assert full_res.summary["output_verification"]["all_outputs_verified"] is True

    # 3. Explicitly test verify_phase3b_outputs and tamper detection
    verified_info = verify_phase3b_outputs(
        output_dir=full_dir,
        expected_manifest_df=manifest_df,
        preprocessing_config=cfg.preprocessing,
    )
    assert verified_info["all_outputs_verified"] is True

    # Tamper with one .npz file and ensure verify_phase3b_outputs raises PreprocessingError
    tamper_npz = Path(full_res.generated_npz_files[0])
    orig_bytes = tamper_npz.read_bytes()
    tamper_npz.write_bytes(b"corrupted_npz_bytes")
    with pytest.raises(PreprocessingError, match="SHA-256 mismatch|Failed to load"):
        verify_phase3b_outputs(full_dir, manifest_df, cfg.preprocessing)
    tamper_npz.write_bytes(orig_bytes)

    # 4. Verify earlier-phase files and raw EDFs remain byte-for-byte identical
    raw_files_after = {p.as_posix(): _sha256(p) for p in data_root.rglob("*.edf")}
    p1_files_after = {p.as_posix(): _sha256(p) for p in p1_dir.glob("*") if p.is_file()}
    p2_files_after = {p.as_posix(): _sha256(p) for p in p2_dir.glob("*") if p.is_file()}
    p3a_files_after = {p.as_posix(): _sha256(p) for p in p3a_dir.glob("*") if p.is_file()}

    assert raw_files_before == raw_files_after
    assert p1_files_before == p1_files_after
    assert p2_files_before == p2_files_after
    assert p3a_files_before == p3a_files_after
