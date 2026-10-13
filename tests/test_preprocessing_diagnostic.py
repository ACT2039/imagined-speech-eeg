"""Unit tests for Phase 3A read-only preprocessing diagnostic helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import signal as sp_signal

from imagined_speech.config import DEFAULT_PRIMARY_EEG_MONTAGE, load_config
from imagined_speech.data.edf_audit import run_dataset_audit
from imagined_speech.data.preprocessing_diagnostic import (
    DEFAULT_CANDIDATE_MAX_PTP_UV,
    DEFAULT_CANDIDATE_MIN_STD_UV,
    DEFAULT_MAX_FLAGGED_CHANNELS_POLICIES,
    apply_candidate_mean_sub_and_bandpass,
    apply_common_average_reference,
    audit_prefixes_and_edf_metadata,
    evaluate_threshold_grid,
    run_phase3a_diagnostic,
    simulate_windowing_provenance,
)
from imagined_speech.data.signal_quality import run_signal_quality_analysis
from imagined_speech.exceptions import SignalQualityError
from tests.conftest import write_synthetic_edf


@pytest.mark.parametrize("duration_sec", [11, 12, 15, 22])
def test_apply_candidate_mean_sub_and_bandpass_across_dataset_durations(
    duration_sec: int,
) -> None:
    """Verify 768-sample (6.0 s) symmetric padding across 11-22 s recordings (including shortest 11 s)."""
    sfreq = 128.0
    n_samples = int(sfreq * duration_sec)
    t = np.arange(n_samples, dtype=np.float64) / sfreq

    # 4230 uV constant offset + 15 uV 10 Hz alpha wave + 5 uV 58 Hz out-of-band noise
    alpha_10hz = 15.0 * np.sin(2.0 * np.pi * 10.0 * t)
    hf_58hz = 5.0 * np.sin(2.0 * np.pi * 58.0 * t)
    raw_signal = 4230.0 + alpha_10hz + hf_58hz
    signals_2d = np.vstack([raw_signal, raw_signal * 0.5])
    original_copy = signals_2d.copy()

    filtered_2d, means = apply_candidate_mean_sub_and_bandpass(
        signals=signals_2d,
        sfreq_hz=sfreq,
        low_hz=0.5,
        high_hz=45.0,
        order=4,
    )

    # Caller array must not be modified
    np.testing.assert_array_equal(signals_2d, original_copy)
    assert filtered_2d.shape == signals_2d.shape
    assert np.isclose(means[0], 4230.0, atol=1e-2)
    assert abs(float(np.mean(filtered_2d[0]))) < 0.01

    # Verify exact match against explicit 768-sample (6.0 s = 3 cycles at 0.5 Hz) even padding
    expected_padlen = int(round(3.0 * sfreq / 0.5))
    assert expected_padlen == 768
    assert n_samples - 1 >= expected_padlen
    sos = sp_signal.butter(4, [0.5, 45.0], btype="bandpass", fs=sfreq, output="sos")
    expected_ch0 = sp_signal.sosfiltfilt(
        sos,
        raw_signal - np.mean(raw_signal),
        padtype="even",
        padlen=768,
    )
    np.testing.assert_allclose(filtered_2d[0], expected_ch0, rtol=1e-13, atol=1e-13)

    # Check interior samples
    interior = slice(128, n_samples - 128)
    assert abs(float(np.mean(filtered_2d[0, interior]))) < 0.01
    residual = filtered_2d[0, interior] - alpha_10hz[interior]
    assert float(np.std(residual)) < 0.05

    # Also verify 1D support
    filtered_1d, mean_1d = apply_candidate_mean_sub_and_bandpass(raw_signal, sfreq_hz=sfreq)
    assert filtered_1d.ndim == 1
    assert np.isclose(float(mean_1d), 4230.0, atol=1e-2)
    np.testing.assert_allclose(filtered_1d, filtered_2d[0], rtol=1e-12, atol=1e-12)


def test_apply_candidate_mean_sub_and_bandpass_edge_cases_and_non_finite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify non-finite inputs, short signal capping, invalid parameters, and filtering errors."""
    sfreq = 128.0
    # 512 samples (4.0 s): shorter than 768 samples, so effective_padlen caps at n_samples - 1 = 511
    t = np.arange(512, dtype=np.float64) / sfreq
    clean_row = 4230.0 + 10.0 * np.sin(2.0 * np.pi * 8.0 * t)
    nan_row = clean_row.copy()
    nan_row[100] = np.nan
    pos_inf_row = clean_row.copy()
    pos_inf_row[200] = np.inf
    neg_inf_row = clean_row.copy()
    neg_inf_row[300] = -np.inf
    arr = np.vstack([clean_row, nan_row, pos_inf_row, neg_inf_row])

    filtered, means = apply_candidate_mean_sub_and_bandpass(arr, sfreq_hz=sfreq)
    assert np.all(np.isfinite(filtered[0]))
    assert np.isfinite(means[0])
    for bad_idx in (1, 2, 3):
        assert np.all(np.isnan(filtered[bad_idx]))
        assert np.isnan(means[bad_idx])

    # All-non-finite input should return all-NaN without calling sosfiltfilt
    all_nan_filt, all_nan_means = apply_candidate_mean_sub_and_bandpass(nan_row, sfreq_hz=sfreq)
    assert np.all(np.isnan(all_nan_filt))
    assert np.isnan(all_nan_means)

    with pytest.raises(ValueError, match="sfreq_hz must be a positive finite number"):
        apply_candidate_mean_sub_and_bandpass(clean_row, sfreq_hz=0.0)
    with pytest.raises(ValueError, match="sfreq_hz must be a positive finite number"):
        apply_candidate_mean_sub_and_bandpass(clean_row, sfreq_hz=float("nan"))
    with pytest.raises(ValueError, match="Invalid band-pass cutoffs"):
        apply_candidate_mean_sub_and_bandpass(clean_row, sfreq_hz=128.0, low_hz=45.0, high_hz=10.0)
    with pytest.raises(ValueError, match="Invalid band-pass cutoffs"):
        apply_candidate_mean_sub_and_bandpass(clean_row, sfreq_hz=128.0, low_hz=0.5, high_hz=64.0)
    with pytest.raises(ValueError, match="Butterworth order must be an integer >= 1"):
        apply_candidate_mean_sub_and_bandpass(clean_row, sfreq_hz=128.0, order=0)
    with pytest.raises(ValueError, match="signals must be 1D or 2D"):
        apply_candidate_mean_sub_and_bandpass(np.ones((2, 2, 100)), sfreq_hz=128.0)
    with pytest.raises(ValueError, match="at least 1 channel"):
        apply_candidate_mean_sub_and_bandpass(np.empty((0, 100)), sfreq_hz=128.0)
    with pytest.raises(ValueError, match="too short for zero-phase filtering"):
        apply_candidate_mean_sub_and_bandpass(np.ones(27), sfreq_hz=128.0)

    # Simulate unexpected runtime failure inside scipy.signal.sosfiltfilt
    def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("synthetic sosfiltfilt failure")

    monkeypatch.setattr(sp_signal, "sosfiltfilt", _boom)
    with pytest.raises(SignalQualityError, match="Zero-phase sosfiltfilt failed"):
        apply_candidate_mean_sub_and_bandpass(clean_row, sfreq_hz=128.0)


def test_apply_common_average_reference_and_spike_propagation() -> None:
    """Verify CAR subtracts instantaneous channel mean and quantifies spike cross-talk."""
    rng = np.random.default_rng(123)
    n_ch, n_samp = 14, 512
    base = rng.normal(0.0, 5.0, size=(n_ch, n_samp))
    # Inject a 1400 uV transient spike on channel 0 only
    spiked = base.copy()
    spiked[0, 250] += 1400.0

    car_signals = apply_common_average_reference(spiked)
    # Instantaneous mean across channels after CAR must be ~0 at every sample
    np.testing.assert_allclose(np.mean(car_signals, axis=0), 0.0, atol=1e-11)
    # Channel 1..13 receive -1400/14 = -100 uV deflection at sample 250
    for ch_idx in range(1, n_ch):
        assert np.isclose(
            car_signals[ch_idx, 250] - (base[ch_idx, 250] - np.mean(base[:, 250])),
            -100.0,
            atol=1e-9,
        )

    with pytest.raises(ValueError, match="requires a 2D array"):
        apply_common_average_reference(np.ones(100))


def test_evaluate_threshold_grid_counts_and_label_balance() -> None:
    """Verify pre-filter and post-filter threshold grid calculations and unflagged retention."""
    rows = []
    # 3 recordings across Char, Digit, Image, each with 2 channels for controlled testing
    specs = [
        # Recording 1 (Char, label 'a'): 0 flagged channels under all moderate rules
        ("Char/P1_a.edf", "Char", "a", "AF3", 12.0, 120.0, 11.5, 110.0),
        ("Char/P1_a.edf", "Char", "a", "F7", 14.0, 140.0, 13.0, 130.0),
        # Recording 2 (Digit, label '1'): 1 channel has pre_ptp=1200 uV (resolved post-filter to 400 uV)
        ("Digit/P1_1.edf", "Digit", "1", "AF3", 15.0, 1200.0, 12.0, 400.0),
        ("Digit/P1_1.edf", "Digit", "1", "F7", 13.0, 130.0, 12.5, 125.0),
        # Recording 3 (Image, label 'up'): 1 channel has low std (0.1 uV pre and post), 1 channel has 1800 uV ptp
        ("Image/P1_up.edf", "Image", "up", "AF3", 0.1, 5.0, 0.1, 4.5),
        ("Image/P1_up.edf", "Image", "up", "F7", 20.0, 1800.0, 19.0, 1750.0),
    ]
    for rel_path, task, lbl, ch, pre_s, pre_p, post_s, post_p in specs:
        rows.append(
            {
                "relative_source_path": rel_path,
                "task_category": task,
                "normalized_label": lbl,
                "raw_channel_name": ch,
                "pre_has_nan_or_inf": False,
                "pre_std_uV": pre_s,
                "pre_ptp_uV": pre_p,
                "post_has_nan_or_inf": False,
                "post_std_uV": post_s,
                "post_ptp_uV": post_p,
            }
        )
    df = pd.DataFrame(rows)

    summary_df, label_df = evaluate_threshold_grid(df)
    expected_n_rules = (
        2
        * len(DEFAULT_CANDIDATE_MIN_STD_UV)
        * len(DEFAULT_CANDIDATE_MAX_PTP_UV)
        * len(DEFAULT_MAX_FLAGGED_CHANNELS_POLICIES)
    )
    assert len(summary_df) == expected_n_rules
    assert set(summary_df["signal_stage"]) == {"pre_filter", "post_filter"}

    # Check rule: min_std=0.5, max_ptp=1000, allow_0_flagged_ch
    pre_rule = summary_df[
        (summary_df["signal_stage"] == "pre_filter")
        & (summary_df["min_std_uV"] == 0.5)
        & (summary_df["max_ptp_uV"] == 1000.0)
        & (summary_df["max_flagged_channels_allowed"] == 0)
    ].iloc[0]
    post_rule = summary_df[
        (summary_df["signal_stage"] == "post_filter")
        & (summary_df["min_std_uV"] == 0.5)
        & (summary_df["max_ptp_uV"] == 1000.0)
        & (summary_df["max_flagged_channels_allowed"] == 0)
    ].iloc[0]

    # Pre-filter: Digit/P1_1.edf (ptp=1200) and Image/P1_up.edf (std=0.1, ptp=1800) are flagged -> 1 unflagged
    assert int(pre_rule["n_unflagged_recordings"]) == 1
    assert int(pre_rule["n_flagged_low_std_channels"]) == 1
    assert int(pre_rule["n_flagged_high_ptp_channels"]) == 2

    # Post-filter: Digit/P1_1.edf resolved to ptp=400 -> 2 unflagged recordings (Char and Digit)
    assert int(post_rule["n_unflagged_recordings"]) == 2
    assert int(post_rule["n_unflagged_char"]) == 1
    assert int(post_rule["n_unflagged_digit"]) == 1
    assert int(post_rule["n_unflagged_image"]) == 0
    assert int(post_rule["n_labels_with_zero_unflagged"]) == 1
    assert len(label_df) == expected_n_rules * 3


def test_simulate_windowing_and_prefix_metadata_audit() -> None:
    """Verify full-recording window simulation and unverified prefix metadata cross-tabulation."""
    manifest_df = pd.DataFrame(
        [
            {
                "relative_source_path": "Char/S01_a.edf",
                "task_category": "Char",
                "normalized_label": "a",
                "unverified_candidate_prefix": "S01",
                "edf_patient_id_raw": "X X X X",
                "edf_recording_id_raw": "Startdate X",
                "edf_start_date": "01.01.26",
                "sfreq_hz": 128.0,
                "n_samples": 128 * 11,
                "duration_sec": 11.0,
            },
            {
                "relative_source_path": "Digit/S01_1.edf",
                "task_category": "Digit",
                "normalized_label": "1",
                "unverified_candidate_prefix": "S01",
                "edf_patient_id_raw": "X X X X",
                "edf_recording_id_raw": "Startdate X",
                "edf_start_date": "01.01.26",
                "sfreq_hz": 128.0,
                "n_samples": 128 * 22,
                "duration_sec": 22.0,
            },
        ]
    )
    ch_stats_df = pd.DataFrame(
        [
            {
                "relative_source_path": "Char/S01_a.edf",
                "unverified_candidate_prefix": "S01",
                "pre_std_uV": 0.4,
                "pre_ptp_uV": 50.0,
                "post_std_uV": 0.35,
                "post_ptp_uV": 45.0,
            },
            {
                "relative_source_path": "Digit/S01_1.edf",
                "unverified_candidate_prefix": "S01",
                "pre_std_uV": 15.0,
                "pre_ptp_uV": 1200.0,
                "post_std_uV": 12.0,
                "post_ptp_uV": 300.0,
            },
        ]
    )

    prefix_df, meta_summary = audit_prefixes_and_edf_metadata(manifest_df, ch_stats_df)
    assert len(prefix_df) == 1
    assert prefix_df.iloc[0]["identity_verification_status"] == "unverified_filename_prefix"
    assert int(prefix_df.iloc[0]["n_pre_low_std_1uV_channels"]) == 1
    assert int(prefix_df.iloc[0]["n_pre_high_ptp_1000uV_channels"]) == 1
    assert int(prefix_df.iloc[0]["n_post_high_ptp_1000uV_channels"]) == 0
    assert meta_summary["n_distinct_unverified_prefixes"] == 1

    win_df = simulate_windowing_provenance(manifest_df, window_schemes=((4.0, 2.0),))
    all_row = win_df[win_df["task_scope"] == "ALL"].iloc[0]
    # 11s with W=4s, S=2s -> windows at [0..4, 2..6, 4..8, 6..10] = 4 windows (1s = 128 trailing samples)
    # 22s with W=4s, S=2s -> windows at [0..4, ..., 18..22] = 10 windows (0 trailing samples)
    assert int(all_row["min_windows_per_recording"]) == 4
    assert int(all_row["max_windows_per_recording"]) == 10
    assert int(all_row["total_windows"]) == 14
    assert int(all_row["n_recordings_with_trailing_unwindowed_samples"]) == 1


def test_run_phase3a_diagnostic_end_to_end_preserves_phase1_and_phase2(tmp_path: Path) -> None:
    """Verify Phase 3A smoke and full runs create all expected artifacts without touching Phase 1/2."""
    data_root = tmp_path / "raw"
    p1_dir = tmp_path / "phase1_audit"
    p2_dir = tmp_path / "phase2_signal_quality"
    p3a_smoke_dir = tmp_path / "phase3a_smoke"
    p3a_full_dir = tmp_path / "phase3a_diagnostic"

    eeg_names = list(DEFAULT_PRIMARY_EEG_MONTAGE)
    rng = np.random.default_rng(42)

    for task, lbl in (("Char", "A"), ("Digit", "1"), ("Image", "Up")):
        t_dir = data_root / task
        t_dir.mkdir(parents=True, exist_ok=True)
        for prefix in ("S01", "S02"):
            signals = 4230.0 + 14.0 * rng.standard_normal((len(eeg_names), 128 * 12))
            write_synthetic_edf(
                t_dir / f"{prefix}_{task}_{lbl}.edf",
                channel_names=eeg_names,
                signals=signals,
                sfreq=128,
            )

    cfg_p1 = load_config(data_root=data_root, output_dir=p1_dir, seed=42)
    run_dataset_audit(cfg_p1, save_outputs=True)
    cfg_p2 = load_config(data_root=data_root, output_dir=p2_dir, recordings_per_task=1, seed=42)
    run_signal_quality_analysis(
        cfg_p2,
        manifest_source=p1_dir / "manifest.csv",
        output_dir=p2_dir,
        save_outputs=True,
    )

    p1_files_before = {p.name: p.read_bytes() for p in p1_dir.glob("*") if p.is_file()}
    p2_files_before = {p.name: p.read_bytes() for p in p2_dir.glob("*") if p.is_file()}

    # 1. Smoke test on 1 recording per task
    smoke_res = run_phase3a_diagnostic(
        config=cfg_p1,
        manifest_source=p1_dir / "manifest.csv",
        output_dir=p3a_smoke_dir,
        selected_relative_paths=[
            "Char/S01_Char_A.edf",
            "Digit/S01_Digit_1.edf",
            "Image/S01_Image_Up.edf",
        ],
        save_outputs=True,
    )
    assert smoke_res.summary["n_recordings_inspected"] == 3
    assert len(smoke_res.channel_stats_df) == 3 * 14

    # 2. Full diagnostic across all discovered recordings
    full_res = run_phase3a_diagnostic(
        config=cfg_p1,
        manifest_source=p1_dir / "manifest.csv",
        output_dir=p3a_full_dir,
        save_outputs=True,
    )
    assert full_res.summary["n_recordings_inspected"] == 6
    assert len(full_res.channel_stats_df) == 6 * 14
    assert len(full_res.generated_files) == 8
    assert len(full_res.generated_figures) == 4
    for fpath in full_res.generated_files + full_res.generated_figures:
        p = Path(fpath)
        assert p.is_file() and p.stat().st_size > 0

    # Confirm Phase 1 and Phase 2 outputs are byte-for-byte unchanged
    p1_files_after = {p.name: p.read_bytes() for p in p1_dir.glob("*") if p.is_file()}
    p2_files_after = {p.name: p.read_bytes() for p in p2_dir.glob("*") if p.is_file()}
    assert p1_files_before == p1_files_after
    assert p2_files_before == p2_files_after
