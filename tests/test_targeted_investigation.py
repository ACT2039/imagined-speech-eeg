"""Unit tests for Phase 3B Targeted Signal Investigation."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from imagined_speech.config import DEFAULT_PRIMARY_EEG_MONTAGE
from imagined_speech.data.preprocessing import (
    save_preprocessed_recording_npz,
)
from imagined_speech.data.targeted_investigation import (
    DEFAULT_INVESTIGATION_CASES,
    CaseInvestigationResult,
    TargetedInvestigationReport,
    build_targeted_investigation_markdown_report,
    compute_trace_statistics,
    compute_zip_edf_sha256_snapshot,
    investigate_calibration_metadata,
    investigate_end_excursion,
    investigate_trace_quantization,
    plot_case_investigation,
    run_phase3b_targeted_investigation,
    verify_raw_edfs_against_zip_or_snapshot,
)
from tests.conftest import write_synthetic_edf


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_compute_trace_statistics() -> None:
    """Verify trace statistics calculation on clean and non-finite signals."""
    sfreq = 128.0
    t = np.arange(128 * 10, dtype=np.float64) / sfreq
    signal = 50.0 * np.sin(2.0 * np.pi * 10.0 * t) + 10.0

    stats = compute_trace_statistics(signal, sfreq)
    assert stats.min_uV == pytest.approx(-40.0, abs=1e-2)
    assert stats.max_uV == pytest.approx(60.0, abs=1e-2)
    assert stats.mean_uV == pytest.approx(10.0, abs=1e-2)
    assert stats.ptp_uV == pytest.approx(100.0, abs=1e-2)
    assert stats.std_uV == pytest.approx(50.0 / np.sqrt(2), abs=0.5)
    assert stats.n_samples == 1280
    assert stats.duration_sec == 10.0
    assert stats.finite_count == 1280
    assert stats.non_finite_count == 0

    # Signal with NaNs and Infs
    signal_with_nans = signal.copy()
    signal_with_nans[10:20] = np.nan
    signal_with_nans[50] = np.inf
    stats_nan = compute_trace_statistics(signal_with_nans, sfreq)
    assert stats_nan.finite_count == 1280 - 11
    assert stats_nan.non_finite_count == 11

    # Signal with all NaNs
    all_nans = np.full(128, np.nan)
    stats_all_nan = compute_trace_statistics(all_nans, sfreq)
    assert stats_all_nan.finite_count == 0
    assert stats_all_nan.non_finite_count == 128
    assert np.isnan(stats_all_nan.mean_uV)


def test_investigate_calibration_metadata(tmp_path: Path) -> None:
    """Verify EDF header calibration inspection and resolution calculation."""
    sfreq = 128
    n_samples = sfreq * 12
    t = np.arange(n_samples, dtype=np.float64) / sfreq

    signals = np.zeros((14, n_samples))
    for i in range(14):
        signals[i] = 4230.0 + 10.0 * np.sin(2.0 * np.pi * (i + 1) * t)

    edf_path = write_synthetic_edf(
        file_path=tmp_path / "Digit" / "test_rec.edf",
        channel_names=list(DEFAULT_PRIMARY_EEG_MONTAGE),
        signals=signals,
        sfreq=sfreq,
    )

    cal = investigate_calibration_metadata(edf_path, "O2")
    assert cal["channel_name"] == "O2"
    assert cal["physical_dimension"] == "uV"
    assert cal["physical_min_uV"] is not None
    assert cal["physical_max_uV"] is not None
    assert cal["digital_min"] is not None
    assert cal["digital_max"] is not None
    assert cal["resolution_uV_per_count"] is not None
    assert cal["sfreq_hz"] == 128.0


def test_investigate_end_excursion_case2() -> None:
    """Verify quantification of timing and magnitude of end-of-recording excursion (Case 2: P8)."""
    sfreq = 128.0
    duration_sec = 12.0
    n_samples = int(sfreq * duration_sec)

    # Quiet interior: std ~ 5 uV
    rng = np.random.default_rng(42)
    filt_signal = rng.normal(0.0, 5.0, size=n_samples)

    # Inject massive excursion in final 0.8 seconds (samples N - 100 to N)
    excursion_start = n_samples - 100
    filt_signal[excursion_start:] += np.linspace(0.0, 800.0, 100)

    # Raw signal has identical baseline offset + excursion
    raw_signal = filt_signal + 4230.0

    res = investigate_end_excursion(
        raw_trace=raw_signal,
        filtered_trace=filt_signal,
        sfreq_hz=sfreq,
        boundary_window_sec=1.0,
    )

    assert res["filtered_end_to_interior_std_ratio"] > 10.0
    assert res["raw_end_to_interior_std_ratio"] > 10.0
    assert res["raw_has_end_excursion"] is True
    assert res["excursion_time_from_recording_end_sec"] <= 1.0
    assert res["peak_magnitude_uV"] > 700.0
    assert "Empirical physical excursion" in res["interpretation"]


def test_investigate_trace_quantization_case3() -> None:
    """Verify classification of constant, quantized, and attenuated signals (Case 3: T8)."""
    n_samples = 128 * 12

    # 1. Constant flatline trace
    constant_trace = np.full(n_samples, 4235.5)
    res_const = investigate_trace_quantization(constant_trace, resolution_uV=0.502)
    assert res_const["is_constant"] is True
    assert res_const["quantization_classification"] == "constant_flatline"

    # 2. Quantized discrete trace with only 3 discrete steps
    quant_trace = np.array([4230.0, 4230.5, 4231.0] * (n_samples // 3))
    res_quant = investigate_trace_quantization(quant_trace, resolution_uV=0.502)
    assert res_quant["is_constant"] is False
    assert res_quant["n_unique_values"] == 3
    assert res_quant["quantization_classification"] == "quantized_discrete_steps"
    assert res_quant["min_step_uV"] == pytest.approx(0.5, abs=1e-3)

    # 3. Continuous attenuated physiological activity (std ~0.25 uV, 500+ unique values)
    rng = np.random.default_rng(123)
    attenuated_trace = 4230.0 + rng.normal(0.0, 0.25, size=n_samples)
    res_att = investigate_trace_quantization(attenuated_trace, resolution_uV=0.502)
    assert res_att["is_constant"] is False
    assert res_att["n_unique_values"] > 100
    assert res_att["quantization_classification"] == "low_amplitude_attenuated"


def test_verify_raw_edfs_against_zip_or_snapshot(tmp_path: Path) -> None:
    """Verify checksum comparison of extracted raw EDFs against original ZIP archive and snapshots."""
    data_root = tmp_path / "raw_data"
    data_root.mkdir()
    edf1 = data_root / "Digit" / "rec1.edf"
    edf2 = data_root / "Char" / "rec2.edf"
    edf1.parent.mkdir(parents=True)
    edf2.parent.mkdir(parents=True)

    edf1.write_bytes(b"EDF_BINARY_CONTENT_1")
    edf2.write_bytes(b"EDF_BINARY_CONTENT_2")

    # 1. Create a matching ZIP archive
    zip_p = tmp_path / "dataset.zip"
    with zipfile.ZipFile(zip_p, "w") as zf:
        zf.writestr("Imagined_speech_EEG_edf/Digit/rec1.edf", b"EDF_BINARY_CONTENT_1")
        zf.writestr("Imagined_speech_EEG_edf/Char/rec2.edf", b"EDF_BINARY_CONTENT_2")

    # Match against ZIP archive
    res_zip = verify_raw_edfs_against_zip_or_snapshot(
        data_root=data_root,
        zip_path=zip_p,
        relative_paths=["Digit/rec1.edf", "Char/rec2.edf"],
    )
    assert res_zip["status"] == "verified_against_zip_archive"
    assert res_zip["is_verified"] is True
    assert res_zip["n_files_checked"] == 2

    # 2. Match against snapshot
    snapshot = {
        "Digit/rec1.edf": _sha256(edf1),
        "Char/rec2.edf": _sha256(edf2),
    }
    res_snap = verify_raw_edfs_against_zip_or_snapshot(
        data_root=data_root,
        zip_path=None,
        baseline_snapshot=snapshot,
        relative_paths=["Digit/rec1.edf", "Char/rec2.edf"],
    )
    assert res_snap["status"] == "verified_against_snapshot"
    assert res_snap["is_verified"] is True

    # 3. Mismatch in ZIP
    edf1.write_bytes(b"TAMPERED_EDF_CONTENT")
    res_mismatch = verify_raw_edfs_against_zip_or_snapshot(
        data_root=data_root,
        zip_path=zip_p,
        relative_paths=["Digit/rec1.edf"],
    )
    assert res_mismatch["status"] == "integrity_violation"
    assert res_mismatch["is_verified"] is False

    # 4. Neither ZIP nor snapshot available -> reports limitation, NEVER verified
    res_unavail = verify_raw_edfs_against_zip_or_snapshot(
        data_root=data_root,
        zip_path=None,
        baseline_snapshot=None,
    )
    assert res_unavail["status"] == "baseline_snapshot_unavailable"
    assert res_unavail["is_verified"] is None


def test_plot_case_investigation(tmp_path: Path) -> None:
    """Verify multi-panel case investigation plot generation with labeled axes."""
    sfreq = 128.0
    duration_sec = 12.0
    n_samples = int(sfreq * duration_sec)
    t = np.arange(n_samples, dtype=np.float64) / sfreq
    raw = 4230.0 + 50.0 * np.sin(2.0 * np.pi * 5.0 * t)
    filt = 50.0 * np.sin(2.0 * np.pi * 5.0 * t)

    case = DEFAULT_INVESTIGATION_CASES[0]
    out_png = tmp_path / "test_case1.png"

    path_ret = plot_case_investigation(
        raw_trace=raw,
        filtered_trace=filt,
        sfreq_hz=sfreq,
        case=case,
        output_png_path=out_png,
        case_result={"raw_ptp_uV": 100.0, "filtered_ptp_uV": 100.0},
    )

    assert Path(path_ret).is_file()
    assert Path(path_ret).stat().st_size > 1000


def test_build_targeted_investigation_markdown_report() -> None:
    """Verify targeted investigation report generation formatting and safeguards."""
    case = DEFAULT_INVESTIGATION_CASES[0]
    raw_stats = compute_trace_statistics(np.linspace(100, 7800, 1280), 128.0)
    filt_stats = compute_trace_statistics(np.linspace(-3850, 3850, 1280), 128.0)
    cal = {
        "channel_name": "O2",
        "raw_label": "O2",
        "channel_index": 7,
        "physical_dimension": "uV",
        "physical_min_uV": -8224.0,
        "physical_max_uV": 8224.0,
        "physical_span_uV": 16448.0,
        "digital_min": -32768,
        "digital_max": 32767,
        "resolution_uV_per_count": 0.502,
        "transducer_type": "EEG",
        "prefiltering": "none",
        "sfreq_hz": 128.0,
    }
    spec_metrics = {
        "amplitude_origin": "present_in_raw_recording",
        "ptp_amplification_ratio": 0.99,
        "raw_ptp_uV": 7700.0,
        "physical_span_uV": 16448.0,
        "is_within_sensor_physical_span": True,
        "finding_summary": "Excursion already present in raw recording.",
    }

    res = CaseInvestigationResult(
        case=case,
        raw_stats=raw_stats,
        filtered_stats=filt_stats,
        subtracted_dc_uV=4230.0,
        calibration=cal,
        specific_metrics=spec_metrics,
    )

    zip_ver = {
        "status": "verified_against_zip_archive",
        "n_files_checked": 3,
        "detail": "All 3 checked EDF files verified identical.",
    }

    report = build_targeted_investigation_markdown_report([res], zip_ver)

    assert "# Phase 3B Targeted Signal Investigation Report" in report
    assert "Digit/name6_6.edf" in report
    assert "O2" in report
    assert "present_in_raw_recording" in report
    assert "Verified against original ZIP archive" in report
    assert "100% Data Retention" in report


def test_run_phase3b_targeted_investigation_end_to_end(tmp_path: Path) -> None:
    """End-to-end integration test of targeted investigation on synthetic dataset."""
    data_root = tmp_path / "data_root"
    preprocessed_dir = tmp_path / "preprocessed"
    output_dir = tmp_path / "investigation_out"
    signals_dir = preprocessed_dir / "signals"
    signals_dir.mkdir(parents=True, exist_ok=True)

    sfreq = 128
    duration_sec = 12
    n_samples = sfreq * duration_sec
    t = np.arange(n_samples, dtype=np.float64) / sfreq

    # Create 3 synthetic EDF recordings matching DEFAULT_INVESTIGATION_CASES
    custom_cases = list(DEFAULT_INVESTIGATION_CASES)

    for case in custom_cases:
        edf_file = data_root / case.relative_source_path
        edf_file.parent.mkdir(parents=True, exist_ok=True)

        raw_signals = np.full((14, n_samples), 4230.0)
        ch_idx = DEFAULT_PRIMARY_EEG_MONTAGE.index(case.channel_name)

        if case.channel_name == "O2":
            # High PTP excursion
            raw_signals[ch_idx] += 3800.0 * np.sin(2.0 * np.pi * 1.0 * t)
        elif case.channel_name == "P8":
            # End excursion
            raw_signals[ch_idx] += np.random.default_rng(42).normal(0.0, 5.0, size=n_samples)
            raw_signals[ch_idx, -100:] += np.linspace(0.0, 500.0, 100)
        elif case.channel_name == "T8":
            # Quantized / low amplitude
            raw_signals[ch_idx] += np.array([0.0, 0.5, -0.5] * (n_samples // 3))

        write_synthetic_edf(
            file_path=edf_file,
            channel_names=list(DEFAULT_PRIMARY_EEG_MONTAGE),
            signals=raw_signals,
            sfreq=sfreq,
        )

        # Preprocessed filtered signals (mean-subtracted)
        filtered_signals = raw_signals - 4230.0
        npz_target = signals_dir / Path(case.relative_source_path).with_suffix(".npz")
        npz_target.parent.mkdir(parents=True, exist_ok=True)

        save_preprocessed_recording_npz(
            npz_path=npz_target,
            eeg_signals_uV=filtered_signals,
            subtracted_channel_means_uV=np.full(14, 4230.0),
            channel_names=DEFAULT_PRIMARY_EEG_MONTAGE,
            sfreq_hz=float(sfreq),
            n_samples=n_samples,
            duration_sec=float(duration_sec),
            metadata={
                "relative_source_path": case.relative_source_path,
                "task_category": case.task_category,
                "filter_low_hz": 0.5,
                "filter_high_hz": 45.0,
                "filter_order": 4,
                "effective_pad_samples": 768,
                "pad_was_capped": False,
            },
        )

    # Snapshot preprocessed dir before run
    p3b_before = {p: _sha256(p) for p in preprocessed_dir.rglob("*") if p.is_file()}
    raw_before = {p: _sha256(p) for p in data_root.rglob("*") if p.is_file()}

    # Run targeted investigation
    report: TargetedInvestigationReport = run_phase3b_targeted_investigation(
        data_root=data_root,
        preprocessed_dir=preprocessed_dir,
        output_dir=output_dir,
        cases=custom_cases,
        generate_figures=True,
    )

    assert isinstance(report, TargetedInvestigationReport)
    assert len(report.cases_results) == 3

    # Check generated files
    csv_file = output_dir / "targeted_trace_statistics.csv"
    assert csv_file.is_file()
    df = report.statistics_df
    assert len(df) == 3
    assert "raw_ptp_uV" in df.columns
    assert "filtered_ptp_uV" in df.columns
    assert "physical_span_uV" in df.columns

    md_file = output_dir / "phase3b_targeted_investigation_report.md"
    assert md_file.is_file()
    assert md_file.stat().st_size > 0

    json_file = output_dir / "phase3b_targeted_investigation_summary.json"
    assert json_file.is_file()
    summary_data = json.loads(json_file.read_text(encoding="utf-8"))
    assert summary_data["cases_inspected"] == 3

    # Check figures
    figs_dir = output_dir / "figures"
    assert len(list(figs_dir.glob("*.png"))) == 3

    # Verify read-only invariant: input files untouched
    p3b_after = {p: _sha256(p) for p in preprocessed_dir.rglob("*") if p.is_file()}
    raw_after = {p: _sha256(p) for p in data_root.rglob("*") if p.is_file()}
    assert p3b_before == p3b_after
    assert raw_before == raw_after


def test_compute_zip_edf_sha256_snapshot_macosx_and_appledouble_exclusion(tmp_path: Path) -> None:
    """Verify that compute_zip_edf_sha256_snapshot excludes __MACOSX metadata and AppleDouble files."""
    zip_path = tmp_path / "test_macosx.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        # Real EDF files under root folder
        zf.writestr("Imagined_speech_EEG_edf/Char/rec_char.edf", b"RAW_CHAR_BYTES")
        zf.writestr("Imagined_speech_EEG_edf/Digit/rec_digit.edf", b"RAW_DIGIT_BYTES")
        zf.writestr("Imagined_speech_EEG_edf/Image/rec_image.edf", b"RAW_IMAGE_BYTES")
        # macOS AppleDouble metadata entries (should be excluded)
        zf.writestr("__MACOSX/Imagined_speech_EEG_edf/Char/._rec_char.edf", b"MAC_METADATA_1")
        zf.writestr("__MACOSX/Imagined_speech_EEG_edf/Digit/._rec_digit.edf", b"MAC_METADATA_2")
        zf.writestr("__MACOSX/._Imagined_speech_EEG_edf", b"MAC_ROOT_META")
        zf.writestr("Imagined_speech_EEG_edf/Char/._orphan_meta.edf", b"MAC_INLINE_META")
        # Non-EDF entries (should be excluded)
        zf.writestr("Imagined_speech_EEG_edf/readme.txt", b"README")

    snapshot = compute_zip_edf_sha256_snapshot(zip_path)

    # Exactly 3 real files, root folder stripped
    assert len(snapshot) == 3
    assert set(snapshot.keys()) == {
        "Char/rec_char.edf",
        "Digit/rec_digit.edf",
        "Image/rec_image.edf",
    }
    assert snapshot["Char/rec_char.edf"] == hashlib.sha256(b"RAW_CHAR_BYTES").hexdigest()
    assert snapshot["Digit/rec_digit.edf"] == hashlib.sha256(b"RAW_DIGIT_BYTES").hexdigest()
    assert snapshot["Image/rec_image.edf"] == hashlib.sha256(b"RAW_IMAGE_BYTES").hexdigest()


def test_verify_raw_edfs_against_zip_archive_structure_and_mismatch_detection(tmp_path: Path) -> None:
    """Verify full-dataset checksum audit excludes __MACOSX, normalizes roots, and detects modifications."""
    data_root = tmp_path / "extracted_data"
    data_root.mkdir()
    (data_root / "Char").mkdir()
    (data_root / "Digit").mkdir()
    (data_root / "Image").mkdir()

    f_char = data_root / "Char" / "rec_char.edf"
    f_digit = data_root / "Digit" / "rec_digit.edf"
    f_image = data_root / "Image" / "rec_image.edf"

    f_char.write_bytes(b"CHAR_SIGNAL_V1")
    f_digit.write_bytes(b"DIGIT_SIGNAL_V1")
    f_image.write_bytes(b"IMAGE_SIGNAL_V1")

    # Create ZIP with archive root + __MACOSX metadata
    zip_path = tmp_path / "source_dataset.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("Imagined_speech_EEG_edf/Char/rec_char.edf", b"CHAR_SIGNAL_V1")
        zf.writestr("Imagined_speech_EEG_edf/Digit/rec_digit.edf", b"DIGIT_SIGNAL_V1")
        zf.writestr("Imagined_speech_EEG_edf/Image/rec_image.edf", b"IMAGE_SIGNAL_V1")
        # AppleDouble metadata entries
        zf.writestr("__MACOSX/Imagined_speech_EEG_edf/Char/._rec_char.edf", b"MAC_META_CHAR")
        zf.writestr("__MACOSX/Imagined_speech_EEG_edf/Digit/._rec_digit.edf", b"MAC_META_DIGIT")
        zf.writestr("Imagined_speech_EEG_edf/Image/._rec_image.edf", b"MAC_INLINE_IMAGE")

    # 1. Full dataset check: should pass with 3 verified files, 0 missing, 0 added
    res_clean = verify_raw_edfs_against_zip_or_snapshot(data_root=data_root, zip_path=zip_path)
    assert res_clean["status"] == "verified_against_zip_archive"
    assert res_clean["is_verified"] is True
    assert res_clean["n_files_checked"] == 3
    assert res_clean["n_files_verified"] == 3
    assert res_clean["missing_files"] == []
    assert res_clean["added_files"] == []
    assert res_clean["modified_files"] == []

    # 2. Tampered file: modify rec_digit.edf
    f_digit.write_bytes(b"CORRUPTED_DIGIT_SIGNAL")
    res_tampered = verify_raw_edfs_against_zip_or_snapshot(data_root=data_root, zip_path=zip_path)
    assert res_tampered["status"] == "integrity_violation"
    assert res_tampered["is_verified"] is False
    assert "Digit/rec_digit.edf" in res_tampered["modified_files"]
    assert "1 modified" in res_tampered["detail"]
    f_digit.write_bytes(b"DIGIT_SIGNAL_V1")  # restore

    # 3. Missing file: delete rec_image.edf
    f_image.unlink()
    res_missing = verify_raw_edfs_against_zip_or_snapshot(data_root=data_root, zip_path=zip_path)
    assert res_missing["status"] == "integrity_violation"
    assert res_missing["is_verified"] is False
    assert "Image/rec_image.edf" in res_missing["missing_files"]
    assert "1 missing" in res_missing["detail"]
    f_image.write_bytes(b"IMAGE_SIGNAL_V1")  # restore

    # 4. Added file: add unexpected recording on disk
    f_extra = data_root / "Char" / "unexpected.edf"
    f_extra.write_bytes(b"UNEXPECTED_EXTRA_RECORDING")
    res_added = verify_raw_edfs_against_zip_or_snapshot(data_root=data_root, zip_path=zip_path)
    assert res_added["status"] == "integrity_violation"
    assert res_added["is_verified"] is False
    assert "Char/unexpected.edf" in res_added["added_files"]
    assert "1 added" in res_added["detail"]
