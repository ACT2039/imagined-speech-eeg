"""Unit tests for Phase 3B EEG Preprocessing Quality Review."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from imagined_speech.config import DEFAULT_PRIMARY_EEG_MONTAGE
from imagined_speech.data.preprocessing import (
    save_preprocessed_recording_npz,
)
from imagined_speech.data.preprocessing_review import (
    Phase3BReviewResult,
    analyze_extreme_concentration,
    build_phase3b_review_markdown_report,
    compute_directory_sha256_snapshot,
    identify_extreme_traces,
    plot_representative_extreme_trace,
    run_phase3b_preprocessing_review,
    verify_directory_checksums,
)
from imagined_speech.exceptions import PreprocessingError


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _create_synthetic_qc_dataframe() -> pd.DataFrame:
    """Create a realistic synthetic ChannelPreprocessedQCRecord DataFrame for testing."""
    records: list[dict[str, object]] = []
    tasks = ["Char", "Digit", "Image"]
    prefixes = ["A", "B", "C"]

    # 3 recordings, each with 14 channels = 42 traces
    for rec_idx in range(3):
        task = tasks[rec_idx]
        prefix = prefixes[rec_idx]
        rel_path = f"{task}/rec_{rec_idx + 1}.edf"
        label = f"label_{rec_idx + 1}"

        for ch_idx, ch_name in enumerate(DEFAULT_PRIMARY_EEG_MONTAGE):
            # Create synthetic distributions with known extremes
            if rec_idx == 0 and ch_name == "F3":
                ptp = 1850.5  # Top PTP
                std = 45.0
                ratio = 1.2
                mean = 1.2e-5
            elif rec_idx == 1 and ch_name == "O1":
                ptp = 450.0
                std = 18.0
                ratio = 4.85  # Top boundary ratio
                mean = -3.4e-5
            elif rec_idx == 2 and ch_name == "P7":
                ptp = 0.85
                std = 0.05  # Lowest std
                ratio = 1.05
                mean = 1.5e-4  # Largest residual mean
            else:
                ptp = 60.0 + float(ch_idx * 5)
                std = 5.0 + float(ch_idx)
                ratio = 1.0 + float(ch_idx) * 0.05
                mean = float(ch_idx) * 1e-6

            records.append(
                {
                    "relative_source_path": rel_path,
                    "task_category": task,
                    "normalized_label": label,
                    "unverified_candidate_prefix": prefix,
                    "channel_name": ch_name,
                    "canonical_channel_order_index": ch_idx,
                    "pre_filter_mean_uV": 4235.0,
                    "pre_filter_std_uV": std * 1.5,
                    "pre_filter_ptp_uV": ptp * 1.5,
                    "subtracted_dc_mean_uV": 4235.0,
                    "post_filter_mean_uV": mean,
                    "post_filter_std_uV": std,
                    "post_filter_ptp_uV": ptp,
                    "post_filter_boundary_1s_std_uV": std * ratio,
                    "post_filter_interior_std_uV": std,
                    "post_filter_boundary_to_interior_std_ratio": ratio,
                    "effective_pad_samples": 768,
                    "pad_was_capped": False,
                    "screening_flag_low_std": std < 0.5,
                    "screening_flag_high_ptp": ptp > 1000.0,
                    "screening_flag_any": (std < 0.5) or (ptp > 1000.0),
                }
            )

    return pd.DataFrame(records)


def test_identify_extreme_traces() -> None:
    """Verify ranking and identification of extreme traces."""
    qc_df = _create_synthetic_qc_dataframe()
    extremes = identify_extreme_traces(qc_df, top_n=5, lowest_std_n=5)

    assert "top_ptp" in extremes
    assert "top_boundary_ratio" in extremes
    assert "lowest_std" in extremes
    assert "largest_residual_mean" in extremes

    # 1. Top PTP should identify F3 in rec 1 (1850.5 uV)
    top_ptp = extremes["top_ptp"]
    assert len(top_ptp) == 5
    assert top_ptp.iloc[0]["channel_name"] == "F3"
    assert top_ptp.iloc[0]["post_filter_ptp_uV"] == pytest.approx(1850.5)
    # Monotonically non-increasing
    assert (top_ptp["post_filter_ptp_uV"].diff().dropna() <= 0).all()

    # 2. Top boundary ratio should identify O1 in rec 2 (4.85x)
    top_b = extremes["top_boundary_ratio"]
    assert len(top_b) == 5
    assert top_b.iloc[0]["channel_name"] == "O1"
    assert top_b.iloc[0]["post_filter_boundary_to_interior_std_ratio"] == pytest.approx(4.85)
    assert (top_b["post_filter_boundary_to_interior_std_ratio"].diff().dropna() <= 0).all()

    # 3. Lowest standard deviation should identify P7 in rec 3 (0.05 uV)
    lowest_std = extremes["lowest_std"]
    assert len(lowest_std) == 5
    assert lowest_std.iloc[0]["channel_name"] == "P7"
    assert lowest_std.iloc[0]["post_filter_std_uV"] == pytest.approx(0.05)
    # Monotonically non-decreasing
    assert (lowest_std["post_filter_std_uV"].diff().dropna() >= 0).all()

    # 4. Largest residual mean should identify P7 in rec 3 (1.5e-4 uV)
    largest_mean = extremes["largest_residual_mean"]
    assert len(largest_mean) == 5
    assert largest_mean.iloc[0]["channel_name"] == "P7"
    assert largest_mean.iloc[0]["abs_post_filter_mean_uV"] == pytest.approx(1.5e-4)

    # 5. Missing columns raises PreprocessingError
    bad_df = qc_df.drop(columns=["post_filter_ptp_uV"])
    with pytest.raises(PreprocessingError, match="missing required columns"):
        identify_extreme_traces(bad_df)


def test_analyze_extreme_concentration() -> None:
    """Verify concentration analysis across tasks, channels, and unverified candidate prefixes."""
    qc_df = _create_synthetic_qc_dataframe()
    manifest_df = pd.DataFrame(
        [
            {"relative_source_path": "Char/rec_1.edf", "task_category": "Char"},
            {"relative_source_path": "Digit/rec_2.edf", "task_category": "Digit"},
            {"relative_source_path": "Image/rec_3.edf", "task_category": "Image"},
        ]
    )

    conc = analyze_extreme_concentration(
        channel_qc_df=qc_df,
        manifest_df=manifest_df,
        ptp_threshold_uV=1000.0,
        std_threshold_uV=0.5,
        boundary_ratio_threshold=2.0,
    )

    assert "concentration_by_task" in conc
    assert "concentration_by_channel" in conc
    assert "concentration_by_prefix" in conc

    # Task concentration
    t_df = conc["concentration_by_task"]
    assert len(t_df) == 3
    char_row = t_df[t_df["task_category"] == "Char"].iloc[0]
    assert char_row["n_high_ptp_traces"] == 1  # F3 was > 1000
    assert char_row["n_channel_traces"] == 14

    # Channel concentration follows canonical 14 montage
    ch_df = conc["concentration_by_channel"]
    assert tuple(ch_df["channel_name"]) == DEFAULT_PRIMARY_EEG_MONTAGE
    f3_row = ch_df[ch_df["channel_name"] == "F3"].iloc[0]
    assert f3_row["n_high_ptp_traces"] == 1

    # Prefix concentration explicitly labels verification status as unverified
    p_df = conc["concentration_by_prefix"]
    assert (p_df["identity_verification_status"] == "unverified_filename_prefix").all()
    # Sorted by high PTP percentage descending
    assert p_df.iloc[0]["unverified_candidate_prefix"] == "A"


def test_plot_representative_extreme_trace(tmp_path: Path) -> None:
    """Verify plotting function renders waveform, axes in uV and seconds, and generates PNG."""
    sfreq = 128.0
    duration_sec = 12.0
    n_samples = int(sfreq * duration_sec)
    t = np.arange(n_samples, dtype=np.float64) / sfreq
    signal = 25.0 * np.sin(2.0 * np.pi * 10.0 * t)

    out_png = tmp_path / "test_trace.png"
    result_path = plot_representative_extreme_trace(
        signal_1d=signal,
        sfreq_hz=sfreq,
        channel_name="AF3",
        recording_title="Char / test_rec",
        output_png_path=out_png,
        boundary_window_sec=1.0,
        metric_label="PTP = 50.0 µV",
    )

    assert Path(result_path).is_file()
    assert Path(result_path).stat().st_size > 1000


def test_compute_directory_sha256_snapshot_and_verify_checksums(tmp_path: Path) -> None:
    """Verify directory snapshot hashing and checksum comparison with limitation reporting."""
    test_dir = tmp_path / "stage_files"
    test_dir.mkdir(parents=True)
    f1 = test_dir / "data1.txt"
    f2 = test_dir / "subdir" / "data2.txt"
    f2.parent.mkdir(parents=True)
    f1.write_text("content 1", encoding="utf-8")
    f2.write_text("content 2", encoding="utf-8")

    # 1. Snapshot creation
    snaps = compute_directory_sha256_snapshot(test_dir)
    assert len(snaps) == 2
    assert "data1.txt" in snaps
    assert "subdir/data2.txt" in snaps

    # 2. Verified unmodified
    ver = verify_directory_checksums(snaps, test_dir, stage_name="test_stage")
    assert ver["status"] == "verified_unmodified"
    assert ver["is_verified"] is True
    assert ver["n_files_checked"] == 2

    # 3. Missing baseline reports limitation explicitly (not claiming verification)
    unavail = verify_directory_checksums({}, test_dir, stage_name="raw_edfs")
    assert unavail["status"] == "baseline_snapshot_unavailable"
    assert unavail["is_verified"] is None
    assert "unavailable" in unavail["detail"].lower()

    unavail_none = verify_directory_checksums(None, None, stage_name="phase1")
    assert unavail_none["status"] == "baseline_snapshot_unavailable"
    assert unavail_none["is_verified"] is None

    # 4. Checksum mismatch reports integrity violation
    f1.write_text("modified content", encoding="utf-8")
    mismatch = verify_directory_checksums(snaps, test_dir, stage_name="test_stage")
    assert mismatch["status"] == "integrity_violation"
    assert mismatch["is_verified"] is False
    assert "data1.txt" in mismatch["modified_files"]


def test_build_phase3b_review_markdown_report() -> None:
    """Verify markdown report generation produces well-formed content with safeguards."""
    qc_df = _create_synthetic_qc_dataframe()
    manifest_df = pd.DataFrame(
        [
            {"relative_source_path": "Char/rec_1.edf", "task_category": "Char"},
            {"relative_source_path": "Digit/rec_2.edf", "task_category": "Digit"},
            {"relative_source_path": "Image/rec_3.edf", "task_category": "Image"},
        ]
    )
    extremes = identify_extreme_traces(qc_df)
    conc = analyze_extreme_concentration(qc_df, manifest_df)

    summary = {
        "n_recordings_inspected": 3,
        "n_channel_traces_audited": 42,
        "screening_thresholds": {
            "qc_screening_min_std_uV": 0.5,
            "qc_screening_max_ptp_uV": 1000.0,
        },
        "integrity_summary": {
            "all_source_recordings_retained": True,
            "raw_edfs": {
                "status": "verified_unmodified",
                "is_verified": True,
                "n_files_checked": 690,
                "detail": "All 690 files verified byte-for-byte identical via SHA-256.",
            },
            "phase1_audit": {
                "status": "baseline_snapshot_unavailable",
                "is_verified": None,
                "n_files_checked": 0,
                "detail": "Baseline snapshot was not available.",
            },
        },
    }

    report = build_phase3b_review_markdown_report(summary, extremes, conc)

    assert "# Phase 3B EEG Preprocessing: Focused Quality Review Report" in report
    assert "Top Extreme Post-Filter Peak-to-Peak (PTP) Amplitudes" in report
    assert "Top Boundary-to-Interior Standard-Deviation Ratios" in report
    assert "Five Lowest Post-Filter Standard Deviations" in report
    assert "Largest Residual Channel Means" in report
    assert "unverified metadata" in report.lower()
    assert "100.0%" in report and "retained in outputs" in report
    assert "µV" in report
    assert "1-second" in report
    assert "Verified unmodified" in report
    assert "Limitation: baseline snapshot unavailable" in report


def test_run_phase3b_preprocessing_review_end_to_end(tmp_path: Path) -> None:
    """End-to-end integration test of review runner on synthetic preprocessed dataset."""
    preprocessed_dir = tmp_path / "preprocessed"
    signals_dir = preprocessed_dir / "signals"
    signals_dir.mkdir(parents=True, exist_ok=True)
    output_dir = tmp_path / "review_output"

    sfreq = 128.0
    duration_sec = 12.0
    n_samples = int(sfreq * duration_sec)
    t = np.arange(n_samples, dtype=np.float64) / sfreq

    # Create synthetic preprocessed .npz files
    manifest_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []

    for idx, task in enumerate(["Char", "Digit", "Image"], start=1):
        rel_path = f"{task}/rec_{idx}.edf"
        signals = np.vstack([
            np.sin(2.0 * np.pi * (idx + ch_i) * t) * 10.0
            for ch_i in range(14)
        ])
        npz_target = signals_dir / f"{task}/rec_{idx}.npz"
        npz_target.parent.mkdir(parents=True, exist_ok=True)

        save_preprocessed_recording_npz(
            npz_path=npz_target,
            eeg_signals_uV=signals,
            subtracted_channel_means_uV=np.full(14, 4230.0),
            channel_names=DEFAULT_PRIMARY_EEG_MONTAGE,
            sfreq_hz=sfreq,
            n_samples=n_samples,
            duration_sec=duration_sec,
            metadata={
                "relative_source_path": rel_path,
                "task_category": task,
                "normalized_label": f"label_{idx}",
                "unverified_candidate_prefix": f"Prefix_{idx}",
                "duration_sec": duration_sec,
                "n_samples": n_samples,
                "filter_low_hz": 0.5,
                "filter_high_hz": 45.0,
                "filter_order": 4,
                "filter_pad_mode": "symmetric",
                "effective_pad_samples": 768,
                "pad_was_capped": False,
                "reference_mode": "none",
            },
        )

        manifest_rows.append(
            {
                "relative_source_path": rel_path,
                "task_category": task,
                "normalized_label": f"label_{idx}",
                "unverified_candidate_prefix": f"Prefix_{idx}",
                "preprocessed_npz_path": str(npz_target),
                "recording_retained_in_outputs": True,
                "preprocessed_n_samples": n_samples,
                "preprocessed_duration_sec": duration_sec,
                "n_eeg_channels": 14,
                "preprocessed_sfreq_hz": sfreq,
                "filter_low_hz": 0.5,
                "filter_high_hz": 45.0,
                "reference_mode": "none",
                "qc_flagged_channel_count": 0,
                "preprocessed_sha256": "dummy_sha",
            }
        )

        for ch_idx, ch_name in enumerate(DEFAULT_PRIMARY_EEG_MONTAGE):
            qc_rows.append(
                {
                    "relative_source_path": rel_path,
                    "task_category": task,
                    "normalized_label": f"label_{idx}",
                    "unverified_candidate_prefix": f"Prefix_{idx}",
                    "channel_name": ch_name,
                    "canonical_channel_order_index": ch_idx,
                    "pre_filter_mean_uV": 4230.0,
                    "pre_filter_std_uV": 15.0,
                    "pre_filter_ptp_uV": 50.0,
                    "subtracted_dc_mean_uV": 4230.0,
                    "post_filter_mean_uV": 1e-6,
                    "post_filter_std_uV": 7.07,
                    "post_filter_ptp_uV": 20.0,
                    "post_filter_boundary_1s_std_uV": 7.07,
                    "post_filter_interior_std_uV": 7.07,
                    "post_filter_boundary_to_interior_std_ratio": 1.0,
                    "effective_pad_samples": 768,
                    "pad_was_capped": False,
                    "screening_flag_low_std": False,
                    "screening_flag_high_ptp": False,
                    "screening_flag_any": False,
                }
            )

    pd.DataFrame(manifest_rows).to_csv(preprocessed_dir / "preprocessed_manifest.csv", index=False)
    pd.DataFrame(qc_rows).to_csv(preprocessed_dir / "channel_qc_metrics.csv", index=False)

    p3b_summary = {
        "preprocessing_stage": "Phase 3B EEG Preprocessing",
        "preprocessing_config": {
            "qc_screening_min_std_uV": 0.5,
            "qc_screening_max_ptp_uV": 1000.0,
        },
    }
    (preprocessed_dir / "phase3b_preprocessing_summary.json").write_text(
        json.dumps(p3b_summary, indent=2), encoding="utf-8"
    )

    # Compute checksums of input files before review
    checksums_before = {
        p: _sha256(p)
        for p in preprocessed_dir.rglob("*")
        if p.is_file()
    }

    # Execute review
    result: Phase3BReviewResult = run_phase3b_preprocessing_review(
        preprocessed_dir=preprocessed_dir,
        output_dir=output_dir,
        save_outputs=True,
        generate_figures=True,
    )

    # Verify output structure and files
    assert isinstance(result, Phase3BReviewResult)
    expected_tables = [
        "top_extreme_ptp_channels.csv",
        "top_boundary_ratio_channels.csv",
        "lowest_std_channels.csv",
        "largest_residual_mean_channels.csv",
        "concentration_by_task.csv",
        "concentration_by_channel.csv",
        "concentration_by_prefix.csv",
    ]
    for tbl in expected_tables:
        tbl_path = output_dir / tbl
        assert tbl_path.is_file(), f"Expected table {tbl} was not created"
        assert tbl_path.stat().st_size > 0

    report_path = output_dir / "phase3b_review_report.md"
    assert report_path.is_file()
    assert report_path.stat().st_size > 0

    summary_path = output_dir / "phase3b_review_summary.json"
    assert summary_path.is_file()
    loaded_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert loaded_summary["n_recordings_inspected"] == 3
    assert loaded_summary["n_channel_traces_audited"] == 42
    assert loaded_summary["integrity_summary"]["all_source_recordings_retained"] is True
    assert loaded_summary["integrity_summary"]["phase3b_preprocessing"]["status"] == "verified_unmodified"
    assert loaded_summary["integrity_summary"]["phase3b_preprocessing"]["is_verified"] is True
    assert loaded_summary["integrity_summary"]["raw_edfs"]["status"] == "baseline_snapshot_unavailable"
    assert loaded_summary["integrity_summary"]["raw_edfs"]["is_verified"] is None
    assert "diagnostic indicator" in loaded_summary["screening_thresholds"]["boundary_ratio_nature"].lower()
    assert "unverified candidate metadata" in loaded_summary["identity_verification_note"].lower()

    # Verify figures
    figures_dir = output_dir / "figures"
    expected_figs = [
        "representative_extreme_ptp_trace.png",
        "representative_extreme_boundary_ratio_trace.png",
        "representative_lowest_std_trace.png",
        "extreme_concentration_breakdown.png",
    ]
    for fig_name in expected_figs:
        fig_p = figures_dir / fig_name
        assert fig_p.is_file(), f"Expected figure {fig_name} was not created"
        assert fig_p.stat().st_size > 0

    # Verify read-only safeguard: input files remain unmodified
    checksums_after = {
        p: _sha256(p)
        for p in preprocessed_dir.rglob("*")
        if p.is_file()
    }
    assert checksums_before == checksums_after
