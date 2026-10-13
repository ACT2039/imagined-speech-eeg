"""Phase 3B: Reusable EEG Preprocessing Pipeline and Post-Filter QC Screening.

Approved Phase 3B Implementation Decisions & Safeguards
-------------------------------------------------------
1. Selects the 14 candidate EEG channels in the canonical montage order
   (``AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4``) and excludes
   auxiliary channels from EEG signal arrays while recording excluded channel names
   in provenance metadata.
2. Applies per-channel mean subtraction followed by a configurable 4th-order
   Butterworth 0.5-45.0 Hz zero-phase band-pass filter at 128 Hz (``sosfiltfilt``).
3. Retains symmetric reflection padding (``padtype="even"``, target 3 cycles at
   ``low_hz`` = 768 samples = 6.0 s at 128 Hz / 0.5 Hz), explicitly validates
   boundary vs. interior signal behavior, and records complete padding provenance
   in per-recording and dataset metadata.
4. Defaults to no additional spatial rereferencing (``reference_mode="none"``).
   Common Average Reference (``reference_mode="car"``) remains an explicitly
   configurable experimental option and is never enabled by default.
5. Computes post-filter channel-level QC metrics and preserves exploratory screening
   flags (default ``post_filter_std_uV < 0.5`` and ``post_filter_ptp_uV > 1000.0``).
   All flags are explicitly labeled as exploratory screening flags, never as verified
   physiological or hardware artifacts.
6. Preserves all 690 recordings and all 14 EEG channels per recording without
   automatic rejection, channel removal, interpolation, or silent sample replacement.
7. Preserves full recording provenance (label, task category, relative source path,
   unverified candidate prefix, duration, sample count, sampling frequency, channel
   order, filter/padding settings, and QC results).
8. Never modifies raw ``.edf`` files, Phase 1 reports, Phase 2 reports, or Phase 3A
   diagnostic reports.
9. Does not infer imagined-speech onset/offset sub-intervals, and does not perform
   windowing, train/val/test splitting, cross-recording normalization, feature
   extraction, or model training.
10. Treats filename prefixes strictly as ``unverified_candidate_prefix`` (never as
    verified participant identifiers).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy import signal as sp_signal

from imagined_speech.config import (
    PreprocessingConfig,
    ProjectConfig,
)
from imagined_speech.data.edf_audit import (
    RawEDFChannelHeader,
    RawEDFFixedHeader,
    clean_channel_name,
    read_raw_edf_header,
)
from imagined_speech.data.preprocessing_diagnostic import apply_common_average_reference
from imagined_speech.data.signal_quality import (
    _extract_native_unit_signals,
    load_and_validate_manifest,
)
from imagined_speech.exceptions import PreprocessingError, SignalQualityError

logger = logging.getLogger("imagined_speech.data.preprocessing")

_PROTECTED_EARLIER_PHASE_ARTIFACTS: tuple[str, ...] = (
    "manifest.csv",
    "channel_audit.csv",
    "audit_summary.json",
    "signal_statistics.csv",
    "signal_quality_summary.json",
    "signal_quality_report.md",
    "phase3a_diagnostic_summary.json",
    "phase3a_diagnostic_report.md",
    "pre_vs_post_filter_channel_stats.csv",
)

SCREENING_NOTE_TEXT: str = (
    "Exploratory screening flag only; not a verified physiological or hardware artifact; "
    "channel signal preserved without modification."
)


@dataclass(frozen=True)
class ExtractedEEGSignals:
    """Validated 14-channel EEG signals in native physical units (uV) in canonical montage order."""

    eeg_signals_uV: np.ndarray
    channel_names: tuple[str, ...]
    raw_channel_labels: tuple[str, ...]
    source_channel_indices: tuple[int, ...]
    excluded_auxiliary_channels: tuple[str, ...]
    physical_units: tuple[str, ...]
    sfreq_hz: float
    n_samples: int
    duration_sec: float
    fixed_header: RawEDFFixedHeader
    channel_headers: tuple[RawEDFChannelHeader, ...]


@dataclass(frozen=True)
class FilterAndPaddingMetadata:
    """Complete provenance record of the signal-conditioning filter, padding, and reference settings."""

    apply_mean_subtraction: bool
    apply_bandpass_filter: bool
    bandpass_low_hz: float
    bandpass_high_hz: float
    butterworth_order: int
    sfreq_hz: float
    n_samples: int
    filter_padtype: str | None
    filter_pad_cycles: float
    filter_padlen_formula: str
    default_padlen_samples: int
    target_padlen_samples: int
    effective_padlen_samples: int
    effective_padlen_sec: float
    padlen_was_capped: bool
    reference_mode: str
    car_applied: bool


@dataclass(frozen=True)
class ChannelBoundaryValidation:
    """Explicit validation metrics comparing boundary segments against interior samples for one channel.

    Diagnostic Scope
    ----------------
    What boundary-versus-interior metrics establish:
    - Quantifies whether edge reflection produced transient variance or peak inflation in the
      initial/final boundary window relative to the interior signal.
    - Confirms that boundary samples remain finite and continuous with interior samples.

    What boundary-versus-interior metrics DO NOT establish:
    - An elevated boundary-to-interior ratio is NOT proof of filter failure or corrupted data.
      Genuine physiological activity (e.g., blinks, saccades, head movements, or task onset effects)
      occurring near recording onset or offset can naturally alter boundary variance.
    - A ratio near 1.0 does NOT prove the absence of artifacts across the entire recording.
    - These metrics are preserved strictly as diagnostic indicators for downstream review
      and windowing/trimming evaluation, and do NOT trigger automated recording rejection
      or channel modification.
    """

    channel_index: int
    channel_name: str
    boundary_window_sec: float
    boundary_window_samples: int
    interior_samples: int
    boundary_all_finite: bool
    onset_window_mean_uV: float | None
    offset_window_mean_uV: float | None
    boundary_std_uV: float | None
    interior_std_uV: float | None
    boundary_to_interior_std_ratio: float | None
    boundary_max_abs_uV: float | None
    interior_max_abs_uV: float | None
    boundary_to_interior_max_abs_ratio: float | None


@dataclass(frozen=True)
class PreprocessedArrayResult:
    """Output of in-memory EEG array preprocessing and boundary behavior validation."""

    preprocessed_signals_uV: np.ndarray
    subtracted_channel_means_uV: np.ndarray
    filter_metadata: FilterAndPaddingMetadata
    boundary_validations: tuple[ChannelBoundaryValidation, ...]


@dataclass(frozen=True)
class ChannelPreprocessedQCRecord:
    """Per-channel pre-filter, post-filter, boundary validation, and exploratory screening record."""

    relative_source_path: str
    original_filename: str
    task_category: str
    raw_label: str | None
    normalized_label: str | None
    unverified_candidate_prefix: str | None
    identity_verification_status: str
    sfreq_hz: float
    n_samples: int
    duration_sec: float
    canonical_channel_order_index: int
    source_edf_channel_index: int
    channel_name: str
    raw_edf_channel_label: str
    physical_unit: str
    reference_mode: str
    pre_filter_has_nan_or_inf: bool
    subtracted_dc_mean_uV: float | None
    pre_filter_mean_uV: float | None
    pre_filter_std_uV: float | None
    pre_filter_min_uV: float | None
    pre_filter_max_uV: float | None
    pre_filter_ptp_uV: float | None
    post_filter_has_nan_or_inf: bool
    post_filter_mean_uV: float | None
    post_filter_std_uV: float | None
    post_filter_median_uV: float | None
    post_filter_iqr_uV: float | None
    post_filter_min_uV: float | None
    post_filter_max_uV: float | None
    post_filter_ptp_uV: float | None
    post_filter_max_abs_uV: float | None
    boundary_window_sec: float
    boundary_window_samples: int
    post_filter_boundary_1s_std_uV: float | None
    post_filter_interior_std_uV: float | None
    post_filter_boundary_to_interior_std_ratio: float | None
    post_filter_boundary_1s_max_abs_uV: float | None
    post_filter_interior_max_abs_uV: float | None
    qc_screening_min_std_uV: float
    qc_screening_max_ptp_uV: float
    screening_flag_non_finite: bool
    screening_flag_low_std: bool
    screening_flag_high_ptp: bool
    screening_flag_any: bool
    screening_flag_reasons: str
    screening_status: str
    screening_note: str


@dataclass(frozen=True)
class RecordingPreprocessedManifestRecord:
    """Per-recording Phase 3B provenance, filter/padding metadata, and QC screening summary row."""

    relative_source_path: str
    original_filename: str
    preprocessed_npz_relpath: str
    preprocessed_npz_sha256: str | None
    task_category: str
    raw_label: str | None
    normalized_label: str | None
    unverified_candidate_prefix: str | None
    identity_verification_status: str
    edf_patient_id_raw: str | None
    edf_recording_id_raw: str | None
    edf_start_date: str | None
    edf_start_time: str | None
    sfreq_hz: float
    n_samples: int
    duration_sec: float
    n_eeg_channels: int
    channel_names_json: str
    excluded_auxiliary_channels_json: str
    physical_unit: str
    apply_mean_subtraction: bool
    apply_bandpass_filter: bool
    bandpass_low_hz: float
    bandpass_high_hz: float
    butterworth_order: int
    filter_padtype: str | None
    filter_pad_cycles: float
    filter_padlen_formula: str
    default_padlen_samples: int
    target_padlen_samples: int
    effective_padlen_samples: int
    effective_padlen_sec: float
    padlen_was_capped: bool
    reference_mode: str
    car_applied: bool
    qc_screening_min_std_uV: float
    qc_screening_max_ptp_uV: float
    n_screening_flagged_channels: int
    n_low_std_channels: int
    n_high_ptp_channels: int
    n_non_finite_channels: int
    screening_flagged_channels_json: str
    low_std_channels_json: str
    high_ptp_channels_json: str
    screening_status: str
    is_unflagged_allow_0: bool
    is_unflagged_allow_1: bool
    is_unflagged_allow_2: bool
    recording_retained_in_outputs: bool
    pre_filter_median_mean_uV: float | None
    pre_filter_median_std_uV: float | None
    pre_filter_max_ptp_uV: float | None
    post_filter_max_abs_mean_uV: float | None
    post_filter_median_std_uV: float | None
    post_filter_min_std_uV: float | None
    post_filter_max_ptp_uV: float | None
    post_filter_median_boundary_to_interior_std_ratio: float | None
    post_filter_max_boundary_to_interior_std_ratio: float | None


@dataclass(frozen=True)
class LoadedPreprocessedRecording:
    """Container returned when loading a saved per-recording Phase 3B ``.npz`` archive."""

    npz_path: Path
    eeg_signals_uV: np.ndarray
    subtracted_channel_means_uV: np.ndarray
    channel_names: tuple[str, ...]
    sfreq_hz: float
    n_samples: int
    duration_sec: float
    metadata: dict[str, Any]


@dataclass
class Phase3BPreprocessingResult:
    """Complete outputs of the Phase 3B preprocessing pipeline."""

    manifest_df: pd.DataFrame
    channel_qc_df: pd.DataFrame
    summary: dict[str, Any] = field(default_factory=dict)
    generated_files: list[str] = field(default_factory=list)
    generated_npz_files: list[str] = field(default_factory=list)
    generated_figures: list[str] = field(default_factory=list)


def _clean_optional_str(val: Any) -> str | None:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if pd.isna(val):
        return None
    text = str(val).strip()
    return text if text else None


def extract_and_validate_eeg_channels(
    edf_path: Path | str,
    preprocessing_config: PreprocessingConfig | None = None,
) -> ExtractedEEGSignals:
    """Read and validate the 14 candidate EEG channels in canonical montage order in native uV.

    Validates:
    - Binary EDF header structure and readability.
    - Sampling rate matches ``preprocessing_config.expected_sfreq_hz`` within tolerance.
    - All expected EEG channels exist with no duplicate channel names.
    - Channels are reordered into ``preprocessing_config.expected_eeg_channels`` order
      regardless of their storage order in the EDF file.
    - Auxiliary/non-EEG channels are excluded from ``eeg_signals_uV`` and recorded in
      ``excluded_auxiliary_channels``.
    - Each selected EEG channel has a valid microvolt physical dimension and valid
      physical/digital range bounds in the EDF header.
    """
    cfg = preprocessing_config or PreprocessingConfig()
    path = Path(edf_path)
    raw_header = read_raw_edf_header(path)
    hdr_by_idx = {ch.channel_index: ch for ch in raw_header.channels}

    try:
        raw_mne = mne.io.read_raw_edf(str(path), preload=True, verbose="ERROR")
    except Exception as exc:
        raise PreprocessingError(f"Failed to read EDF signal data from {path}: {exc}") from exc

    try:
        sfreq_hz = float(raw_mne.info["sfreq"])
        native_signals, native_units = _extract_native_unit_signals(raw_mne, hdr_by_idx)
    finally:
        raw_mne.close()

    if abs(sfreq_hz - cfg.expected_sfreq_hz) > cfg.sfreq_tolerance_hz:
        raise PreprocessingError(
            f"Recording {path} has sfreq_hz={sfreq_hz}, expected "
            f"{cfg.expected_sfreq_hz} +/- {cfg.sfreq_tolerance_hz} Hz."
        )

    expected_Lookup = {name.strip().upper(): name.strip() for name in cfg.expected_eeg_channels}
    allowed_units_lower = {u.strip().lower() for u in cfg.expected_physical_units}

    clean_to_indices: dict[str, list[int]] = {}
    for ch in raw_header.channels:
        cleaned_upper = clean_channel_name(ch.label).upper()
        clean_to_indices.setdefault(cleaned_upper, []).append(ch.channel_index)

    duplicate_eeg = [
        name
        for key, name in expected_Lookup.items()
        if len(clean_to_indices.get(key, [])) > 1
    ]
    if duplicate_eeg:
        raise PreprocessingError(
            f"Recording {path} contains duplicate candidate EEG channels: {duplicate_eeg}"
        )

    missing_eeg = [
        name for key, name in expected_Lookup.items() if key not in clean_to_indices
    ]
    if missing_eeg:
        raise PreprocessingError(
            f"Recording {path} is missing expected candidate EEG channels: {missing_eeg}"
        )

    ordered_indices: list[int] = []
    ordered_names: list[str] = []
    ordered_raw_labels: list[str] = []
    ordered_headers: list[RawEDFChannelHeader] = []
    ordered_units: list[str] = []

    for target_name in cfg.expected_eeg_channels:
        key = target_name.strip().upper()
        ch_idx = clean_to_indices[key][0]
        hdr_ch = raw_header.channels[ch_idx]

        unit_raw = (
            hdr_ch.physical_dimension.strip()
            if hdr_ch.physical_dimension and hdr_ch.physical_dimension.strip()
            else native_units[ch_idx].strip()
        )
        if unit_raw.lower() not in allowed_units_lower:
            raise PreprocessingError(
                f"Channel {hdr_ch.label!r} (index {ch_idx}) in {path} has unsupported "
                f"physical unit {unit_raw!r}; expected one of {list(cfg.expected_physical_units)}."
            )

        if (
            hdr_ch.physical_min is None
            or hdr_ch.physical_max is None
            or not (hdr_ch.physical_min < hdr_ch.physical_max)
        ):
            raise PreprocessingError(
                f"Channel {hdr_ch.label!r} in {path} has invalid EDF physical range "
                f"[{hdr_ch.physical_min}, {hdr_ch.physical_max}]."
            )
        if (
            hdr_ch.digital_min is None
            or hdr_ch.digital_max is None
            or not (hdr_ch.digital_min < hdr_ch.digital_max)
        ):
            raise PreprocessingError(
                f"Channel {hdr_ch.label!r} in {path} has invalid EDF digital range "
                f"[{hdr_ch.digital_min}, {hdr_ch.digital_max}]."
            )

        ordered_indices.append(ch_idx)
        ordered_names.append(clean_channel_name(hdr_ch.label))
        ordered_raw_labels.append(hdr_ch.label)
        ordered_headers.append(hdr_ch)
        ordered_units.append("uV")

    selected_idx_set = set(ordered_indices)
    excluded_aux = tuple(
        clean_channel_name(ch.label)
        for ch in raw_header.channels
        if ch.channel_index not in selected_idx_set
    )

    eeg_signals = np.asarray(native_signals[ordered_indices], dtype=np.float64).copy()
    n_samples = int(eeg_signals.shape[1])
    duration_sec = float(n_samples / sfreq_hz)

    return ExtractedEEGSignals(
        eeg_signals_uV=eeg_signals,
        channel_names=tuple(ordered_names),
        raw_channel_labels=tuple(ordered_raw_labels),
        source_channel_indices=tuple(ordered_indices),
        excluded_auxiliary_channels=excluded_aux,
        physical_units=tuple(ordered_units),
        sfreq_hz=sfreq_hz,
        n_samples=n_samples,
        duration_sec=duration_sec,
        fixed_header=raw_header,
        channel_headers=tuple(ordered_headers),
    )


def validate_boundary_behavior(
    signals_2d: np.ndarray,
    sfreq_hz: float,
    channel_names: Sequence[str] | None = None,
    boundary_window_sec: float = 1.0,
) -> tuple[ChannelBoundaryValidation, ...]:
    """Quantify boundary vs. interior signal behavior for every channel after filtering.

    Compares the initial and final ``boundary_window_sec`` segments (capped at ``n_samples // 4``
    for very short test signals) against the interior segment ``[b_samples : n_samples - b_samples]``.

    Diagnostic Scope & Interpretation
    ---------------------------------
    What these metrics establish:
    - Quantifies whether boundary reflection padding produced variance inflation or peak
      amplitude distortion in the initial/final boundary window relative to the interior.
    - Confirms that all boundary samples remain finite and continuous with interior samples.

    What these metrics CANNOT establish:
    - An elevated boundary-to-interior ratio does NOT prove filter failure or corrupt data.
      Genuine physiological activity (e.g., blinks, saccades, movement, or task onset effects)
      occurring near recording boundaries naturally elevates boundary variance.
    - A ratio near 1.0 does NOT prove the absence of artifacts across the whole recording.
    - These metrics are preserved for downstream diagnostic monitoring and epoch-windowing
      evaluations, and do NOT trigger automated recording rejection or channel modification.
    """
    arr = np.asarray(signals_2d, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise PreprocessingError(
            f"validate_boundary_behavior expects a 2D array with >= 1 channel, got {arr.shape}."
        )
    if not np.isfinite(sfreq_hz) or sfreq_hz <= 0:
        raise PreprocessingError(f"sfreq_hz must be positive and finite, got {sfreq_hz!r}.")
    if not np.isfinite(boundary_window_sec) or boundary_window_sec <= 0:
        raise PreprocessingError(
            f"boundary_window_sec must be positive and finite, got {boundary_window_sec!r}."
        )

    n_channels, n_samples = arr.shape
    if n_samples < 4:
        raise PreprocessingError(
            f"Signal length n_samples={n_samples} is too short for boundary validation (requires >= 4)."
        )

    target_b_samples = max(1, int(round(boundary_window_sec * sfreq_hz)))
    b_samples = min(n_samples // 4, target_b_samples)
    eff_window_sec = float(b_samples / sfreq_hz)
    interior_len = int(n_samples - 2 * b_samples)

    names = (
        tuple(str(n) for n in channel_names)
        if channel_names is not None
        else tuple(f"CH{i}" for i in range(n_channels))
    )
    if len(names) != n_channels:
        raise PreprocessingError(
            f"Length of channel_names ({len(names)}) does not match n_channels ({n_channels})."
        )

    results: list[ChannelBoundaryValidation] = []
    for idx in range(n_channels):
        row = arr[idx]
        onset_seg = row[:b_samples]
        offset_seg = row[n_samples - b_samples :]
        boundary_seg = np.concatenate([onset_seg, offset_seg])
        interior_seg = row[b_samples : n_samples - b_samples]

        b_finite = bool(np.all(np.isfinite(boundary_seg)))
        all_finite = bool(b_finite and np.all(np.isfinite(interior_seg)))

        if all_finite:
            onset_mean = float(np.mean(onset_seg))
            offset_mean = float(np.mean(offset_seg))
            b_std = float(np.std(boundary_seg))
            i_std = float(np.std(interior_seg))
            std_ratio = float(b_std / i_std) if i_std > 1.0e-12 else 1.0
            b_max_abs = float(np.max(np.abs(boundary_seg)))
            i_max_abs = float(np.max(np.abs(interior_seg)))
            max_abs_ratio = float(b_max_abs / i_max_abs) if i_max_abs > 1.0e-12 else 1.0
        else:
            onset_mean = None
            offset_mean = None
            b_std = None
            i_std = None
            std_ratio = None
            b_max_abs = None
            i_max_abs = None
            max_abs_ratio = None

        results.append(
            ChannelBoundaryValidation(
                channel_index=idx,
                channel_name=names[idx],
                boundary_window_sec=eff_window_sec,
                boundary_window_samples=b_samples,
                interior_samples=interior_len,
                boundary_all_finite=b_finite,
                onset_window_mean_uV=onset_mean,
                offset_window_mean_uV=offset_mean,
                boundary_std_uV=b_std,
                interior_std_uV=i_std,
                boundary_to_interior_std_ratio=std_ratio,
                boundary_max_abs_uV=b_max_abs,
                interior_max_abs_uV=i_max_abs,
                boundary_to_interior_max_abs_ratio=max_abs_ratio,
            )
        )

    return tuple(results)


def preprocess_eeg_array(
    signals_uV: np.ndarray,
    sfreq_hz: float = 128.0,
    preprocessing_config: PreprocessingConfig | None = None,
    channel_names: Sequence[str] | None = None,
) -> PreprocessedArrayResult:
    """Apply Phase 3B signal conditioning to a 1D or 2D EEG array on an in-memory float64 copy.

    Processing order:
    1. Per-channel arithmetic mean subtraction (when ``apply_mean_subtraction=True``).
    2. Zero-phase Butterworth band-pass filter via ``scipy.signal.sosfiltfilt``
       (when ``apply_bandpass_filter=True``), recording full symmetric padding provenance.
    3. Spatial reference mode: ``"none"`` by default (no rereferencing); ``"car"``
       applies Common Average Reference only when explicitly configured.
    4. Boundary vs. interior signal behavior validation.

    Channels containing non-finite inputs (``NaN`` or ``Inf``) are preserved as ``NaN``
    traces without silent value replacement so they are explicitly flagged by QC.
    """
    cfg = preprocessing_config or PreprocessingConfig(expected_sfreq_hz=sfreq_hz)
    if not np.isfinite(sfreq_hz) or sfreq_hz <= 0:
        raise PreprocessingError(f"sfreq_hz must be a positive finite number, got {sfreq_hz!r}.")
    nyquist = sfreq_hz / 2.0
    if cfg.apply_bandpass_filter and not (
        0.0 < cfg.bandpass_low_hz < cfg.bandpass_high_hz < nyquist
    ):
        raise PreprocessingError(
            f"Invalid bandpass cutoffs low_hz={cfg.bandpass_low_hz}, "
            f"high_hz={cfg.bandpass_high_hz} for Nyquist={nyquist} Hz."
        )

    arr = np.asarray(signals_uV, dtype=np.float64).copy()
    was_1d = arr.ndim == 1
    if was_1d:
        arr = arr[np.newaxis, :]
    elif arr.ndim != 2:
        raise PreprocessingError(f"signals_uV must be 1D or 2D, got shape {arr.shape}.")

    n_channels, n_samples = arr.shape
    if n_channels == 0:
        raise PreprocessingError("signals_uV must contain at least 1 channel (got 0 channels).")

    try:
        sos = sp_signal.butter(
            N=cfg.butterworth_order,
            Wn=[cfg.bandpass_low_hz, cfg.bandpass_high_hz],
            btype="bandpass",
            fs=sfreq_hz,
            output="sos",
        )
    except Exception as exc:
        raise PreprocessingError(
            f"Failed to construct Butterworth filter (order={cfg.butterworth_order}, "
            f"low_hz={cfg.bandpass_low_hz}, high_hz={cfg.bandpass_high_hz}, "
            f"sfreq_hz={sfreq_hz}): {exc}"
        ) from exc

    default_padlen = 3 * (2 * len(sos) + 1)
    if cfg.apply_bandpass_filter and n_samples <= default_padlen:
        raise PreprocessingError(
            f"Signal length n_samples={n_samples} is too short for zero-phase filtering "
            f"(requires > {default_padlen} samples)."
        )

    target_padlen = int(round(cfg.filter_pad_cycles * sfreq_hz / cfg.bandpass_low_hz))
    if cfg.apply_bandpass_filter and cfg.filter_padtype is not None:
        effective_padlen = min(n_samples - 1, max(default_padlen, target_padlen))
        padlen_was_capped = bool(effective_padlen < target_padlen)
        padlen_formula = (
            f"min(n_samples - 1, max(default_padlen, round({cfg.filter_pad_cycles:g} * sfreq_hz / low_hz)))"
        )
    else:
        effective_padlen = 0
        padlen_was_capped = False
        padlen_formula = "0 (padding disabled)"

    output = np.full_like(arr, np.nan, dtype=np.float64)
    subtracted_means = np.full(n_channels, np.nan, dtype=np.float64)

    finite_row_mask = np.all(np.isfinite(arr), axis=1)
    if np.any(finite_row_mask):
        valid_rows = arr[finite_row_mask]
        if cfg.apply_mean_subtraction:
            row_means = np.mean(valid_rows, axis=1, keepdims=True)
            centered = valid_rows - row_means
            subtracted_means[finite_row_mask] = row_means[:, 0]
        else:
            centered = valid_rows.copy()
            subtracted_means[finite_row_mask] = 0.0

        if cfg.apply_bandpass_filter:
            try:
                filtered_valid = sp_signal.sosfiltfilt(
                    sos,
                    centered,
                    axis=1,
                    padtype=cfg.filter_padtype,
                    padlen=effective_padlen if cfg.filter_padtype is not None else None,
                )
            except Exception as exc:
                raise PreprocessingError(
                    f"Zero-phase sosfiltfilt failed on signal of shape {valid_rows.shape} "
                    f"with padtype={cfg.filter_padtype!r}, padlen={effective_padlen}: {exc}"
                ) from exc
        else:
            filtered_valid = centered

        if not np.all(np.isfinite(filtered_valid)):
            raise PreprocessingError(
                "Band-pass filtering introduced non-finite values on finite input channels."
            )

        output[finite_row_mask] = filtered_valid

    car_applied = False
    if cfg.reference_mode == "car":
        if n_channels < 2:
            raise PreprocessingError(
                "reference_mode='car' requires at least 2 channels, got 1 channel."
            )
        if np.all(finite_row_mask):
            output = apply_common_average_reference(output)
            car_applied = True
        elif int(np.sum(finite_row_mask)) >= 2:
            output[finite_row_mask] = apply_common_average_reference(output[finite_row_mask])
            car_applied = True

    boundary_validations = validate_boundary_behavior(
        signals_2d=output,
        sfreq_hz=sfreq_hz,
        channel_names=channel_names,
        boundary_window_sec=cfg.boundary_validation_window_sec,
    )

    filter_meta = FilterAndPaddingMetadata(
        apply_mean_subtraction=cfg.apply_mean_subtraction,
        apply_bandpass_filter=cfg.apply_bandpass_filter,
        bandpass_low_hz=float(cfg.bandpass_low_hz),
        bandpass_high_hz=float(cfg.bandpass_high_hz),
        butterworth_order=int(cfg.butterworth_order),
        sfreq_hz=float(sfreq_hz),
        n_samples=int(n_samples),
        filter_padtype=cfg.filter_padtype,
        filter_pad_cycles=float(cfg.filter_pad_cycles),
        filter_padlen_formula=padlen_formula,
        default_padlen_samples=int(default_padlen),
        target_padlen_samples=int(target_padlen),
        effective_padlen_samples=int(effective_padlen),
        effective_padlen_sec=round(float(effective_padlen / sfreq_hz), 6),
        padlen_was_capped=padlen_was_capped,
        reference_mode=cfg.reference_mode,
        car_applied=car_applied,
    )

    if was_1d:
        return PreprocessedArrayResult(
            preprocessed_signals_uV=output[0],
            subtracted_channel_means_uV=subtracted_means[:1],
            filter_metadata=filter_meta,
            boundary_validations=boundary_validations,
        )

    return PreprocessedArrayResult(
        preprocessed_signals_uV=output,
        subtracted_channel_means_uV=subtracted_means,
        filter_metadata=filter_meta,
        boundary_validations=boundary_validations,
    )


def compute_channel_qc_records(
    *,
    raw_signals_uV: np.ndarray,
    preprocessed_result: PreprocessedArrayResult,
    channel_names: Sequence[str],
    raw_channel_labels: Sequence[str],
    source_channel_indices: Sequence[int],
    relative_source_path: str,
    original_filename: str,
    task_category: str,
    raw_label: str | None,
    normalized_label: str | None,
    unverified_candidate_prefix: str | None,
    sfreq_hz: float,
    preprocessing_config: PreprocessingConfig | None = None,
) -> tuple[ChannelPreprocessedQCRecord, ...]:
    """Compute per-channel pre-filter, post-filter, boundary, and exploratory screening QC records.

    All thresholds produce exploratory screening flags only; no channels or samples are dropped
    or altered.
    """
    cfg = preprocessing_config or PreprocessingConfig(expected_sfreq_hz=sfreq_hz)
    raw_2d = np.asarray(raw_signals_uV, dtype=np.float64)
    post_2d = np.asarray(preprocessed_result.preprocessed_signals_uV, dtype=np.float64)
    if post_2d.ndim == 1:
        post_2d = post_2d[np.newaxis, :]
    if raw_2d.shape != post_2d.shape:
        raise PreprocessingError(
            f"Shape mismatch between raw_signals_uV {raw_2d.shape} and "
            f"preprocessed_signals_uV {post_2d.shape}."
        )

    n_channels, n_samples = raw_2d.shape
    duration_sec = float(n_samples / sfreq_hz)
    min_std_thr = float(cfg.qc_screening_min_std_uV)
    max_ptp_thr = float(cfg.qc_screening_max_ptp_uV)

    records: list[ChannelPreprocessedQCRecord] = []
    for idx in range(n_channels):
        raw_1d = raw_2d[idx]
        post_1d = post_2d[idx]
        b_val = preprocessed_result.boundary_validations[idx]
        sub_mean = preprocessed_result.subtracted_channel_means_uV[idx]

        pre_non_finite = bool(not np.all(np.isfinite(raw_1d)))
        post_non_finite = bool(not np.all(np.isfinite(post_1d)))

        if not pre_non_finite:
            pre_mean = float(np.mean(raw_1d))
            pre_std = float(np.std(raw_1d))
            pre_min = float(np.min(raw_1d))
            pre_max = float(np.max(raw_1d))
            pre_ptp = float(pre_max - pre_min)
            sub_mean_val = float(sub_mean) if np.isfinite(sub_mean) else pre_mean
        else:
            pre_mean = None
            pre_std = None
            pre_min = None
            pre_max = None
            pre_ptp = None
            sub_mean_val = None

        if not post_non_finite:
            post_mean = float(np.mean(post_1d))
            post_std = float(np.std(post_1d))
            q25, q50, q75 = np.percentile(post_1d, [25.0, 50.0, 75.0])
            post_median = float(q50)
            post_iqr = float(q75 - q25)
            post_min = float(np.min(post_1d))
            post_max = float(np.max(post_1d))
            post_ptp = float(post_max - post_min)
            post_max_abs = float(np.max(np.abs(post_1d)))
        else:
            post_mean = None
            post_std = None
            post_median = None
            post_iqr = None
            post_min = None
            post_max = None
            post_ptp = None
            post_max_abs = None

        flag_non_finite = bool(pre_non_finite or post_non_finite)
        flag_low_std = bool(
            not flag_non_finite and post_std is not None and post_std < min_std_thr
        )
        flag_high_ptp = bool(
            not flag_non_finite and post_ptp is not None and post_ptp > max_ptp_thr
        )
        flag_any = bool(flag_non_finite or flag_low_std or flag_high_ptp)

        reasons: list[str] = []
        if flag_non_finite:
            reasons.append("non_finite_samples")
        if flag_low_std:
            reasons.append(f"low_post_filter_std(<{min_std_thr:g}uV)")
        if flag_high_ptp:
            reasons.append(f"high_post_filter_ptp(>{max_ptp_thr:g}uV)")

        screening_status = (
            "flagged_for_screening" if flag_any else "unflagged_under_screening_rule"
        )

        records.append(
            ChannelPreprocessedQCRecord(
                relative_source_path=relative_source_path,
                original_filename=original_filename,
                task_category=task_category,
                raw_label=raw_label,
                normalized_label=normalized_label,
                unverified_candidate_prefix=unverified_candidate_prefix,
                identity_verification_status="unverified_filename_prefix",
                sfreq_hz=float(sfreq_hz),
                n_samples=int(n_samples),
                duration_sec=duration_sec,
                canonical_channel_order_index=int(idx),
                source_edf_channel_index=int(source_channel_indices[idx]),
                channel_name=str(channel_names[idx]),
                raw_edf_channel_label=str(raw_channel_labels[idx]),
                physical_unit="uV",
                reference_mode=preprocessed_result.filter_metadata.reference_mode,
                pre_filter_has_nan_or_inf=pre_non_finite,
                subtracted_dc_mean_uV=sub_mean_val,
                pre_filter_mean_uV=pre_mean,
                pre_filter_std_uV=pre_std,
                pre_filter_min_uV=pre_min,
                pre_filter_max_uV=pre_max,
                pre_filter_ptp_uV=pre_ptp,
                post_filter_has_nan_or_inf=post_non_finite,
                post_filter_mean_uV=post_mean,
                post_filter_std_uV=post_std,
                post_filter_median_uV=post_median,
                post_filter_iqr_uV=post_iqr,
                post_filter_min_uV=post_min,
                post_filter_max_uV=post_max,
                post_filter_ptp_uV=post_ptp,
                post_filter_max_abs_uV=post_max_abs,
                boundary_window_sec=b_val.boundary_window_sec,
                boundary_window_samples=b_val.boundary_window_samples,
                post_filter_boundary_1s_std_uV=b_val.boundary_std_uV,
                post_filter_interior_std_uV=b_val.interior_std_uV,
                post_filter_boundary_to_interior_std_ratio=b_val.boundary_to_interior_std_ratio,
                post_filter_boundary_1s_max_abs_uV=b_val.boundary_max_abs_uV,
                post_filter_interior_max_abs_uV=b_val.interior_max_abs_uV,
                qc_screening_min_std_uV=min_std_thr,
                qc_screening_max_ptp_uV=max_ptp_thr,
                screening_flag_non_finite=flag_non_finite,
                screening_flag_low_std=flag_low_std,
                screening_flag_high_ptp=flag_high_ptp,
                screening_flag_any=flag_any,
                screening_flag_reasons=";".join(reasons),
                screening_status=screening_status,
                screening_note=SCREENING_NOTE_TEXT,
            )
        )

    return tuple(records)


def save_preprocessed_recording_npz(
    npz_path: Path | str,
    *,
    eeg_signals_uV: np.ndarray,
    subtracted_channel_means_uV: np.ndarray,
    channel_names: Sequence[str],
    sfreq_hz: float,
    n_samples: int,
    duration_sec: float,
    metadata: dict[str, Any],
) -> str:
    """Write a compressed ``.npz`` archive for one preprocessed recording and return its SHA-256 hex digest."""
    target = Path(npz_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata_json_str = json.dumps(metadata, sort_keys=True)

    np.savez_compressed(
        target,
        eeg_signals_uV=np.asarray(eeg_signals_uV, dtype=np.float64),
        subtracted_channel_means_uV=np.asarray(subtracted_channel_means_uV, dtype=np.float64),
        channel_names=np.asarray(list(channel_names), dtype="<U32"),
        sfreq_hz=np.asarray(float(sfreq_hz), dtype=np.float64),
        n_samples=np.asarray(int(n_samples), dtype=np.int64),
        duration_sec=np.asarray(float(duration_sec), dtype=np.float64),
        metadata_json=np.asarray(metadata_json_str),
    )
    return hashlib.sha256(target.read_bytes()).hexdigest()


def load_preprocessed_recording(npz_path: Path | str) -> LoadedPreprocessedRecording:
    """Load and validate a saved Phase 3B per-recording ``.npz`` archive."""
    path = Path(npz_path)
    if not path.is_file():
        raise PreprocessingError(f"Preprocessed NPZ file not found: {path}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            required_keys = {
                "eeg_signals_uV",
                "subtracted_channel_means_uV",
                "channel_names",
                "sfreq_hz",
                "n_samples",
                "duration_sec",
                "metadata_json",
            }
            missing = required_keys - set(archive.files)
            if missing:
                raise PreprocessingError(
                    f"Preprocessed NPZ {path} is missing required keys: {sorted(missing)}"
                )
            eeg_signals = np.asarray(archive["eeg_signals_uV"], dtype=np.float64).copy()
            sub_means = np.asarray(archive["subtracted_channel_means_uV"], dtype=np.float64).copy()
            ch_names = tuple(str(x) for x in archive["channel_names"].tolist())
            sfreq_hz = float(archive["sfreq_hz"].item())
            n_samples = int(archive["n_samples"].item())
            duration_sec = float(archive["duration_sec"].item())
            meta_raw = str(archive["metadata_json"].item())
            metadata = json.loads(meta_raw)
    except PreprocessingError:
        raise
    except Exception as exc:
        raise PreprocessingError(f"Failed to load preprocessed NPZ {path}: {exc}") from exc

    return LoadedPreprocessedRecording(
        npz_path=path,
        eeg_signals_uV=eeg_signals,
        subtracted_channel_means_uV=sub_means,
        channel_names=ch_names,
        sfreq_hz=sfreq_hz,
        n_samples=n_samples,
        duration_sec=duration_sec,
        metadata=metadata,
    )


def _opt_stat(values: Sequence[float | None], func: Any) -> float | None:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite:
        return None
    return float(func(np.asarray(finite, dtype=np.float64)))


def preprocess_single_recording(
    edf_path: Path | str,
    manifest_row: dict[str, Any] | pd.Series,
    preprocessing_config: PreprocessingConfig | None = None,
    *,
    output_dir: Path | str | None = None,
    save_npz: bool = True,
) -> tuple[
    RecordingPreprocessedManifestRecord,
    tuple[ChannelPreprocessedQCRecord, ...],
    PreprocessedArrayResult,
]:
    """Run Phase 3B preprocessing and QC screening for a single EDF recording."""
    cfg = preprocessing_config or PreprocessingConfig()
    extracted = extract_and_validate_eeg_channels(
        edf_path=edf_path,
        preprocessing_config=cfg,
    )

    prep_res = preprocess_eeg_array(
        signals_uV=extracted.eeg_signals_uV,
        sfreq_hz=extracted.sfreq_hz,
        preprocessing_config=cfg,
        channel_names=extracted.channel_names,
    )

    rel_source_path = str(manifest_row["relative_source_path"]).replace("\\", "/")
    orig_filename = str(
        manifest_row.get("original_filename") or Path(edf_path).name
    )
    task_category = str(manifest_row["task_category"])
    raw_label = _clean_optional_str(manifest_row.get("raw_label"))
    normalized_label = _clean_optional_str(manifest_row.get("normalized_label"))
    prefix = _clean_optional_str(manifest_row.get("unverified_candidate_prefix"))

    edf_patient_id = _clean_optional_str(
        manifest_row.get("edf_patient_id_raw", extracted.fixed_header.local_patient_id)
    )
    edf_recording_id = _clean_optional_str(
        manifest_row.get("edf_recording_id_raw", extracted.fixed_header.local_recording_id)
    )
    edf_start_date = _clean_optional_str(
        manifest_row.get("edf_start_date", extracted.fixed_header.start_date)
    )
    edf_start_time = _clean_optional_str(
        manifest_row.get("edf_start_time", extracted.fixed_header.start_time)
    )

    qc_records = compute_channel_qc_records(
        raw_signals_uV=extracted.eeg_signals_uV,
        preprocessed_result=prep_res,
        channel_names=extracted.channel_names,
        raw_channel_labels=extracted.raw_channel_labels,
        source_channel_indices=extracted.source_channel_indices,
        relative_source_path=rel_source_path,
        original_filename=orig_filename,
        task_category=task_category,
        raw_label=raw_label,
        normalized_label=normalized_label,
        unverified_candidate_prefix=prefix,
        sfreq_hz=extracted.sfreq_hz,
        preprocessing_config=cfg,
    )

    flagged_ch_names = [r.channel_name for r in qc_records if r.screening_flag_any]
    low_std_ch_names = [r.channel_name for r in qc_records if r.screening_flag_low_std]
    high_ptp_ch_names = [r.channel_name for r in qc_records if r.screening_flag_high_ptp]
    n_non_finite = int(sum(1 for r in qc_records if r.screening_flag_non_finite))
    n_flagged = len(flagged_ch_names)

    rec_screening_status = (
        "unflagged_under_screening_rule"
        if n_flagged == 0
        else "has_screening_flagged_channels"
    )

    npz_relpath = (Path("signals") / Path(rel_source_path).with_suffix(".npz")).as_posix()
    f_meta = prep_res.filter_metadata

    npz_metadata: dict[str, Any] = {
        "relative_source_path": rel_source_path,
        "original_filename": orig_filename,
        "preprocessed_npz_relpath": npz_relpath,
        "task_category": task_category,
        "raw_label": raw_label,
        "normalized_label": normalized_label,
        "unverified_candidate_prefix": prefix,
        "identity_verification_status": "unverified_filename_prefix",
        "edf_patient_id_raw": edf_patient_id,
        "edf_recording_id_raw": edf_recording_id,
        "edf_start_date": edf_start_date,
        "edf_start_time": edf_start_time,
        "sfreq_hz": extracted.sfreq_hz,
        "n_samples": extracted.n_samples,
        "duration_sec": extracted.duration_sec,
        "channel_names": list(extracted.channel_names),
        "raw_channel_labels": list(extracted.raw_channel_labels),
        "source_channel_indices": list(extracted.source_channel_indices),
        "excluded_auxiliary_channels": list(extracted.excluded_auxiliary_channels),
        "physical_unit": "uV",
        "filter_and_padding_metadata": asdict(f_meta),
        "qc_screening_thresholds": {
            "qc_screening_min_std_uV": cfg.qc_screening_min_std_uV,
            "qc_screening_max_ptp_uV": cfg.qc_screening_max_ptp_uV,
            "terminology_note": SCREENING_NOTE_TEXT,
        },
        "qc_summary": {
            "n_screening_flagged_channels": n_flagged,
            "n_low_std_channels": len(low_std_ch_names),
            "n_high_ptp_channels": len(high_ptp_ch_names),
            "n_non_finite_channels": n_non_finite,
            "screening_flagged_channels": flagged_ch_names,
            "low_std_channels": low_std_ch_names,
            "high_ptp_channels": high_ptp_ch_names,
            "screening_status": rec_screening_status,
            "is_unflagged_allow_0": n_flagged == 0,
            "is_unflagged_allow_1": n_flagged <= 1,
            "is_unflagged_allow_2": n_flagged <= 2,
            "recording_retained_in_outputs": True,
        },
        "channel_qc_records": [asdict(r) for r in qc_records],
    }

    npz_sha256: str | None = None
    if save_npz:
        if output_dir is None:
            raise PreprocessingError("output_dir must be provided when save_npz=True.")
        full_npz_path = Path(output_dir) / npz_relpath
        npz_sha256 = save_preprocessed_recording_npz(
            full_npz_path,
            eeg_signals_uV=prep_res.preprocessed_signals_uV,
            subtracted_channel_means_uV=prep_res.subtracted_channel_means_uV,
            channel_names=extracted.channel_names,
            sfreq_hz=extracted.sfreq_hz,
            n_samples=extracted.n_samples,
            duration_sec=extracted.duration_sec,
            metadata=npz_metadata,
        )

    pre_means = [r.pre_filter_mean_uV for r in qc_records]
    pre_stds = [r.pre_filter_std_uV for r in qc_records]
    pre_ptps = [r.pre_filter_ptp_uV for r in qc_records]
    post_abs_means = [
        abs(r.post_filter_mean_uV) if r.post_filter_mean_uV is not None else None
        for r in qc_records
    ]
    post_stds = [r.post_filter_std_uV for r in qc_records]
    post_ptps = [r.post_filter_ptp_uV for r in qc_records]
    b_ratios = [r.post_filter_boundary_to_interior_std_ratio for r in qc_records]

    manifest_record = RecordingPreprocessedManifestRecord(
        relative_source_path=rel_source_path,
        original_filename=orig_filename,
        preprocessed_npz_relpath=npz_relpath,
        preprocessed_npz_sha256=npz_sha256,
        task_category=task_category,
        raw_label=raw_label,
        normalized_label=normalized_label,
        unverified_candidate_prefix=prefix,
        identity_verification_status="unverified_filename_prefix",
        edf_patient_id_raw=edf_patient_id,
        edf_recording_id_raw=edf_recording_id,
        edf_start_date=edf_start_date,
        edf_start_time=edf_start_time,
        sfreq_hz=extracted.sfreq_hz,
        n_samples=extracted.n_samples,
        duration_sec=extracted.duration_sec,
        n_eeg_channels=len(extracted.channel_names),
        channel_names_json=json.dumps(list(extracted.channel_names)),
        excluded_auxiliary_channels_json=json.dumps(list(extracted.excluded_auxiliary_channels)),
        physical_unit="uV",
        apply_mean_subtraction=f_meta.apply_mean_subtraction,
        apply_bandpass_filter=f_meta.apply_bandpass_filter,
        bandpass_low_hz=f_meta.bandpass_low_hz,
        bandpass_high_hz=f_meta.bandpass_high_hz,
        butterworth_order=f_meta.butterworth_order,
        filter_padtype=f_meta.filter_padtype,
        filter_pad_cycles=f_meta.filter_pad_cycles,
        filter_padlen_formula=f_meta.filter_padlen_formula,
        default_padlen_samples=f_meta.default_padlen_samples,
        target_padlen_samples=f_meta.target_padlen_samples,
        effective_padlen_samples=f_meta.effective_padlen_samples,
        effective_padlen_sec=f_meta.effective_padlen_sec,
        padlen_was_capped=f_meta.padlen_was_capped,
        reference_mode=f_meta.reference_mode,
        car_applied=f_meta.car_applied,
        qc_screening_min_std_uV=cfg.qc_screening_min_std_uV,
        qc_screening_max_ptp_uV=cfg.qc_screening_max_ptp_uV,
        n_screening_flagged_channels=n_flagged,
        n_low_std_channels=len(low_std_ch_names),
        n_high_ptp_channels=len(high_ptp_ch_names),
        n_non_finite_channels=n_non_finite,
        screening_flagged_channels_json=json.dumps(flagged_ch_names),
        low_std_channels_json=json.dumps(low_std_ch_names),
        high_ptp_channels_json=json.dumps(high_ptp_ch_names),
        screening_status=rec_screening_status,
        is_unflagged_allow_0=n_flagged == 0,
        is_unflagged_allow_1=n_flagged <= 1,
        is_unflagged_allow_2=n_flagged <= 2,
        recording_retained_in_outputs=True,
        pre_filter_median_mean_uV=_opt_stat(pre_means, np.median),
        pre_filter_median_std_uV=_opt_stat(pre_stds, np.median),
        pre_filter_max_ptp_uV=_opt_stat(pre_ptps, np.max),
        post_filter_max_abs_mean_uV=_opt_stat(post_abs_means, np.max),
        post_filter_median_std_uV=_opt_stat(post_stds, np.median),
        post_filter_min_std_uV=_opt_stat(post_stds, np.min),
        post_filter_max_ptp_uV=_opt_stat(post_ptps, np.max),
        post_filter_median_boundary_to_interior_std_ratio=_opt_stat(b_ratios, np.median),
        post_filter_max_boundary_to_interior_std_ratio=_opt_stat(b_ratios, np.max),
    )

    return manifest_record, qc_records, prep_res


def _validate_safe_output_directory(output_dir: Path, data_root: Path) -> None:
    """Prevent Phase 3B from writing inside raw data_root or overwriting Phase 1/2/3A directories."""
    out_resolved = output_dir.resolve()
    data_resolved = data_root.resolve()
    if out_resolved == data_resolved or out_resolved.is_relative_to(data_resolved):
        raise PreprocessingError(
            f"Refusing to write Phase 3B outputs inside raw data_root ({data_resolved})."
        )
    if out_resolved.is_dir():
        existing_protected = [
            name
            for name in _PROTECTED_EARLIER_PHASE_ARTIFACTS
            if (out_resolved / name).exists()
        ]
        if existing_protected:
            raise PreprocessingError(
                f"Output directory {out_resolved} contains earlier-phase artifacts "
                f"{existing_protected}. Phase 3B must write to a separate directory."
            )


def _quantiles_dict(values: np.ndarray) -> dict[str, float | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {k: None for k in ("min", "p5", "p25", "median", "p75", "p95", "max")}
    p5, p25, p50, p75, p95 = np.percentile(finite, [5.0, 25.0, 50.0, 75.0, 95.0])
    return {
        "min": round(float(np.min(finite)), 6),
        "p5": round(float(p5), 6),
        "p25": round(float(p25), 6),
        "median": round(float(p50), 6),
        "p75": round(float(p75), 6),
        "p95": round(float(p95), 6),
        "max": round(float(np.max(finite)), 6),
    }


def _generate_phase3b_figures(
    manifest_df: pd.DataFrame,
    channel_qc_df: pd.DataFrame,
    preprocessing_config: PreprocessingConfig,
    figures_dir: Path,
) -> list[str]:
    """Generate 3 summary figures for Phase 3B preprocessing and QC screening."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    # Figure 1: Post-filter standard deviation and peak-to-peak distributions with screening thresholds
    fig1_path = figures_dir / "post_filter_std_and_ptp_distributions.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    post_std = np.log10(
        np.clip(channel_qc_df["post_filter_std_uV"].dropna().to_numpy(dtype=float), 1e-6, None)
    )
    axes[0].hist(post_std, bins=50, color="#1f77b4", alpha=0.8)
    if preprocessing_config.qc_screening_min_std_uV > 0:
        axes[0].axvline(
            math.log10(preprocessing_config.qc_screening_min_std_uV),
            color="red",
            linestyle="--",
            linewidth=1.4,
            label=f"Screening min_std={preprocessing_config.qc_screening_min_std_uV:g} uV",
        )
        axes[0].legend(fontsize=8)
    axes[0].set_xlabel("log10(Post-Filter Std [uV])")
    axes[0].set_ylabel("Channel Trace Count")
    axes[0].set_title("Phase 3B Post-Filter Channel Standard Deviation")

    post_ptp = np.log10(
        np.clip(channel_qc_df["post_filter_ptp_uV"].dropna().to_numpy(dtype=float), 1e-6, None)
    )
    axes[1].hist(post_ptp, bins=50, color="#ff7f0e", alpha=0.8)
    axes[1].axvline(
        math.log10(preprocessing_config.qc_screening_max_ptp_uV),
        color="red",
        linestyle="--",
        linewidth=1.4,
        label=f"Screening max_ptp={preprocessing_config.qc_screening_max_ptp_uV:g} uV",
    )
    axes[1].legend(fontsize=8)
    axes[1].set_xlabel("log10(Post-Filter Peak-to-Peak [uV])")
    axes[1].set_ylabel("Channel Trace Count")
    axes[1].set_title("Phase 3B Post-Filter Channel Peak-to-Peak Range")
    fig.tight_layout()
    fig.savefig(fig1_path, dpi=150)
    plt.close(fig)
    generated.append(str(fig1_path))

    # Figure 2: Boundary vs. interior std ratio validation
    fig2_path = figures_dir / "boundary_vs_interior_std_ratio.png"
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    ratios = channel_qc_df["post_filter_boundary_to_interior_std_ratio"].dropna().to_numpy(dtype=float)
    if ratios.size > 0:
        ax.hist(ratios, bins=45, color="#2ca02c", alpha=0.8)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.2, label="Ratio = 1.0 (parity)")
    ax.set_xlabel("Boundary (First/Last 1s) to Interior Std Ratio")
    ax.set_ylabel("Channel Trace Count")
    ax.set_title("Symmetric-Padding Boundary vs. Interior Std Ratio")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig2_path, dpi=150)
    plt.close(fig)
    generated.append(str(fig2_path))

    # Figure 3: Exploratory screening flag counts by channel and task
    fig3_path = figures_dir / "screening_flags_by_channel_and_task.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ch_order = list(preprocessing_config.expected_eeg_channels)
    ch_flags = (
        channel_qc_df.groupby("channel_name")["screening_flag_any"]
        .sum()
        .reindex(ch_order, fill_value=0)
    )
    axes[0].bar(range(len(ch_order)), ch_flags.to_numpy(dtype=int), color="#9467bd", alpha=0.85)
    axes[0].set_xticks(range(len(ch_order)))
    axes[0].set_xticklabels(ch_order, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("Flagged Channel Trace Count")
    axes[0].set_title("Exploratory Screening Flags by EEG Channel (All Preserved)")

    task_names = sorted(manifest_df["task_category"].unique().tolist())
    unflagged_0 = [
        int(
            manifest_df[manifest_df["task_category"] == t]["is_unflagged_allow_0"].sum()
        )
        for t in task_names
    ]
    flagged_0 = [
        int(
            (~manifest_df[manifest_df["task_category"] == t]["is_unflagged_allow_0"]).sum()
        )
        for t in task_names
    ]
    x = np.arange(len(task_names))
    axes[1].bar(x, unflagged_0, label="Unflagged (0 flagged ch)", color="#2ca02c", alpha=0.85)
    axes[1].bar(
        x,
        flagged_0,
        bottom=unflagged_0,
        label=">=1 Screening-Flagged Ch (Retained)",
        color="#d62728",
        alpha=0.75,
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(task_names)
    axes[1].set_ylabel("Recording Count")
    axes[1].set_title("Recordings by Task and Screening Status (100% Retained)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig3_path, dpi=150)
    plt.close(fig)
    generated.append(str(fig3_path))

    return generated


def _build_phase3b_markdown_report(summary: dict[str, Any]) -> str:
    """Format a human-readable Markdown report for Phase 3B preprocessing."""
    cfg_s = summary["preprocessing_config"]
    pad_s = summary["padding_and_boundary_summary"]
    qc_s = summary["qc_screening_summary"]
    dist_s = summary["signal_distribution_summary"]

    lines = [
        "# Phase 3B: Reusable EEG Preprocessing Pipeline Report",
        "",
        "## 1. Approved Configuration & Scientific Safeguards",
        f"- **Recordings Processed & Retained**: `{summary['n_recordings_processed']}` / `{summary['n_recordings_expected']}` (`100.0%` retained; `{summary['n_channel_traces_processed']}` EEG channel traces)",
        f"- **Canonical 14-Channel EEG Montage**: `{', '.join(cfg_s['expected_eeg_channels'])}` (auxiliary channels excluded from signal arrays)",
        f"- **Physical Units**: `{cfg_s['physical_unit']}` (MNE SI scaling inverted to native microvolts; verified against EDF channel headers)",
        f"- **Signal Conditioning**: Per-channel mean subtraction (`{cfg_s['apply_mean_subtraction']}`) + `{cfg_s['butterworth_order']}`th-order Butterworth `{cfg_s['bandpass_low_hz']:g}-{cfg_s['bandpass_high_hz']:g} Hz` zero-phase band-pass filter at `{cfg_s['expected_sfreq_hz']:g} Hz` (`sosfiltfilt`)",
        f"- **Spatial Reference Mode**: `{cfg_s['reference_mode']}` (no additional rereferencing applied by default)",
        f"- **Exploratory Screening Rules**: `post_filter_std_uV < {cfg_s['qc_screening_min_std_uV']:g} uV` or `post_filter_ptp_uV > {cfg_s['qc_screening_max_ptp_uV']:g} uV` (labeled strictly as exploratory screening flags, not verified artifacts)",
        "- **Preservation Safeguards**: No recordings were rejected, no channels were removed or interpolated, no sample values were silently replaced, no speech onset/offset times were inferred, and filename prefixes remain `unverified_filename_prefix`.",
        "",
        "## 2. Filter Padding Provenance & Boundary Validation",
        f"- **Padding Type (`filter_padtype`)**: `{pad_s['filter_padtype']}`",
        f"- **Padding Formula**: `{pad_s['filter_padlen_formula']}`",
        f"- **Target Padding at 128 Hz**: `{pad_s['target_padlen_samples']} samples` (`{pad_s['target_padlen_sec']:.2f} s`)",
        f"- **Effective Padding Applied (min / max samples)**: `{pad_s['effective_padlen_samples_min']}` / `{pad_s['effective_padlen_samples_max']} samples`",
        f"- **Recordings with Capped Padding (`padlen_was_capped`)**: `{pad_s['n_recordings_padlen_capped']}`",
        f"- **Boundary Window Evaluated**: First `{pad_s['boundary_validation_window_sec']:g} s` and last `{pad_s['boundary_validation_window_sec']:g} s` vs. interior segment",
        f"- **Boundary All-Finite Traces**: `{pad_s['n_boundary_finite_traces']}` / `{summary['n_channel_traces_processed']}`",
        f"- **Boundary-to-Interior Std Ratio**: median `{pad_s['boundary_to_interior_std_ratio']['median']}`, p5 `{pad_s['boundary_to_interior_std_ratio']['p5']}`, p95 `{pad_s['boundary_to_interior_std_ratio']['p95']}`, max `{pad_s['boundary_to_interior_std_ratio']['max']}`",
        "",
        "## 3. Pre-Filter vs. Post-Filter Signal Distributions",
        f"- **Subtracted DC Mean (`uV`)**: median `{dist_s['subtracted_dc_mean_uV']['median']}`, min `{dist_s['subtracted_dc_mean_uV']['min']}`, max `{dist_s['subtracted_dc_mean_uV']['max']}`",
        f"- **Post-Filter Residual Mean (`uV`)**: median `{dist_s['post_filter_mean_uV']['median']}`, max abs `{dist_s['post_filter_max_abs_mean_uV']}`",
        f"- **Pre-Filter Std (`uV`)**: median `{dist_s['pre_filter_std_uV']['median']}`, p5 `{dist_s['pre_filter_std_uV']['p5']}`, p95 `{dist_s['pre_filter_std_uV']['p95']}`",
        f"- **Post-Filter Std (`uV`)**: median `{dist_s['post_filter_std_uV']['median']}`, p5 `{dist_s['post_filter_std_uV']['p5']}`, p95 `{dist_s['post_filter_std_uV']['p95']}`",
        f"- **Pre-Filter PTP (`uV`)**: median `{dist_s['pre_filter_ptp_uV']['median']}`, p95 `{dist_s['pre_filter_ptp_uV']['p95']}`, max `{dist_s['pre_filter_ptp_uV']['max']}`",
        f"- **Post-Filter PTP (`uV`)**: median `{dist_s['post_filter_ptp_uV']['median']}`, p95 `{dist_s['post_filter_ptp_uV']['p95']}`, max `{dist_s['post_filter_ptp_uV']['max']}`",
        "",
        "## 4. Exploratory Post-Filter QC Screening Summary (100% Retained)",
        f"- **Flagged Channel Traces (`screening_flag_any`)**: `{qc_s['n_flagged_channel_traces']}` / `{summary['n_channel_traces_processed']}` (`{qc_s['flagged_channel_traces_pct']:.2f}%`)",
        f"  - `screening_flag_low_std (< {cfg_s['qc_screening_min_std_uV']:g} uV)`: `{qc_s['n_low_std_channel_traces']}`",
        f"  - `screening_flag_high_ptp (> {cfg_s['qc_screening_max_ptp_uV']:g} uV)`: `{qc_s['n_high_ptp_channel_traces']}`",
        f"  - `screening_flag_non_finite`: `{qc_s['n_non_finite_channel_traces']}`",
        f"- **Unflagged Recordings (Allow 0 Flagged Channels)**: `{qc_s['n_unflagged_recordings_allow_0']}` / `{summary['n_recordings_processed']}` (`{qc_s['unflagged_recordings_allow_0_pct']:.2f}%`)",
        f"- **Unflagged Recordings (Allow <= 1 Flagged Channel)**: `{qc_s['n_unflagged_recordings_allow_1']}` / `{summary['n_recordings_processed']}` (`{qc_s['unflagged_recordings_allow_1_pct']:.2f}%`)",
        f"- **Unflagged Recordings (Allow <= 2 Flagged Channels)**: `{qc_s['n_unflagged_recordings_allow_2']}` / `{summary['n_recordings_processed']}` (`{qc_s['unflagged_recordings_allow_2_pct']:.2f}%`)",
        f"- **Recordings Saved to Output Archive**: `{qc_s['n_recordings_retained_in_outputs']}` (`100.0%`)",
        "",
    ]
    return "\n".join(lines)


def verify_phase3b_outputs(
    output_dir: Path | str,
    expected_manifest_df: pd.DataFrame,
    preprocessing_config: PreprocessingConfig | None = None,
) -> dict[str, Any]:
    """Verify 1-to-1 completeness and integrity of all Phase 3B processed outputs on disk.

    Checks that:
    1. ``preprocessed_manifest.csv``, ``channel_qc_metrics.csv``,
       ``phase3b_preprocessing_summary.json``, and ``phase3b_preprocessing_report.md``
       exist and are non-empty.
    2. Every expected recording from ``expected_manifest_df`` is present in
       ``preprocessed_manifest.csv`` with ``recording_retained_in_outputs == True``.
    3. Every expected recording has exactly 14 rows in ``channel_qc_metrics.csv``
       matching ``expected_eeg_channels`` in canonical order.
    4. Every recording's ``.npz`` file exists, is non-empty, matches its recorded
       SHA-256 digest, and loads with ``eeg_signals_uV.shape == (14, n_samples)``,
       matching channel names, sampling rate, and metadata.
    """
    cfg = preprocessing_config or PreprocessingConfig()
    target_dir = Path(output_dir)
    required_top_files = (
        "preprocessed_manifest.csv",
        "channel_qc_metrics.csv",
        "phase3b_preprocessing_summary.json",
        "phase3b_preprocessing_report.md",
    )
    for fname in required_top_files:
        fpath = target_dir / fname
        if not fpath.is_file() or fpath.stat().st_size == 0:
            raise PreprocessingError(
                f"Required Phase 3B output file is missing or empty: {fpath}"
            )

    out_manifest = pd.read_csv(target_dir / "preprocessed_manifest.csv")
    out_qc = pd.read_csv(target_dir / "channel_qc_metrics.csv")

    expected_rel_paths = [
        str(p).replace("\\", "/")
        for p in expected_manifest_df["relative_source_path"].tolist()
    ]
    actual_rel_paths = [
        str(p).replace("\\", "/")
        for p in out_manifest["relative_source_path"].tolist()
    ]

    if sorted(actual_rel_paths) != sorted(expected_rel_paths):
        missing = sorted(set(expected_rel_paths) - set(actual_rel_paths))
        extra = sorted(set(actual_rel_paths) - set(expected_rel_paths))
        raise PreprocessingError(
            f"Phase 3B manifest recording mismatch: missing={missing[:10]}, extra={extra[:10]}"
        )

    if not out_manifest["recording_retained_in_outputs"].astype(bool).all():
        raise PreprocessingError(
            "One or more recordings have recording_retained_in_outputs != True."
        )

    n_expected_channels = len(cfg.expected_eeg_channels)
    if len(out_qc) != len(expected_rel_paths) * n_expected_channels:
        raise PreprocessingError(
            f"Expected {len(expected_rel_paths) * n_expected_channels} rows in "
            f"channel_qc_metrics.csv, found {len(out_qc)}."
        )

    qc_counts = out_qc.groupby("relative_source_path").size().to_dict()
    for rel_path in expected_rel_paths:
        if qc_counts.get(rel_path, 0) != n_expected_channels:
            raise PreprocessingError(
                f"Recording {rel_path} has {qc_counts.get(rel_path, 0)} QC rows, "
                f"expected {n_expected_channels}."
            )

    verified_npz_count = 0
    for _, m_row in out_manifest.iterrows():
        rel_path = str(m_row["relative_source_path"])
        npz_rel = str(m_row["preprocessed_npz_relpath"])
        npz_path = target_dir / npz_rel
        if not npz_path.is_file() or npz_path.stat().st_size == 0:
            raise PreprocessingError(
                f"Missing or empty preprocessed NPZ file for {rel_path}: {npz_path}"
            )
        expected_sha = _clean_optional_str(m_row.get("preprocessed_npz_sha256"))
        actual_sha = hashlib.sha256(npz_path.read_bytes()).hexdigest()
        if expected_sha and actual_sha != expected_sha:
            raise PreprocessingError(
                f"SHA-256 mismatch for {npz_path}: expected {expected_sha}, got {actual_sha}"
            )

        loaded = load_preprocessed_recording(npz_path)
        expected_n_samples = int(m_row["n_samples"])
        if loaded.eeg_signals_uV.shape != (n_expected_channels, expected_n_samples):
            raise PreprocessingError(
                f"NPZ {npz_path} has shape {loaded.eeg_signals_uV.shape}, "
                f"expected {(n_expected_channels, expected_n_samples)}."
            )
        if loaded.channel_names != tuple(cfg.expected_eeg_channels):
            raise PreprocessingError(
                f"NPZ {npz_path} has channel order {loaded.channel_names}, "
                f"expected {cfg.expected_eeg_channels}."
            )
        if loaded.metadata.get("relative_source_path") != rel_path:
            raise PreprocessingError(
                f"NPZ {npz_path} metadata relative_source_path mismatch: "
                f"{loaded.metadata.get('relative_source_path')} != {rel_path}"
            )
        verified_npz_count += 1

    return {
        "n_expected_recordings": len(expected_rel_paths),
        "n_verified_manifest_rows": len(out_manifest),
        "n_verified_qc_rows": len(out_qc),
        "n_verified_npz_files": verified_npz_count,
        "all_outputs_verified": True,
    }


def run_phase3b_preprocessing(
    config: ProjectConfig,
    manifest_source: pd.DataFrame | Path | str,
    output_dir: Path | str | None = None,
    *,
    selected_relative_paths: Sequence[str] | None = None,
    save_outputs: bool = True,
    verify_completeness: bool = True,
) -> Phase3BPreprocessingResult:
    """Execute the reusable Phase 3B preprocessing pipeline across manifest recordings."""
    try:
        if isinstance(manifest_source, (Path, str)):
            manifest_df = load_and_validate_manifest(Path(manifest_source))
        else:
            manifest_df = load_and_validate_manifest(manifest_source.copy())
    except SignalQualityError as exc:
        raise PreprocessingError(str(exc)) from exc

    manifest_df = manifest_df.sort_values("relative_source_path").reset_index(drop=True)
    if selected_relative_paths is not None:
        sel_set = {str(p).replace("\\", "/") for p in selected_relative_paths}
        manifest_df = (
            manifest_df[
                manifest_df["relative_source_path"]
                .astype(str)
                .str.replace("\\", "/", regex=False)
                .isin(sel_set)
            ]
            .sort_values("relative_source_path")
            .reset_index(drop=True)
        )
        if manifest_df.empty:
            raise PreprocessingError(
                "No rows in manifest_source matched selected_relative_paths."
            )

    target_dir = Path(output_dir) if output_dir is not None else Path(config.output_dir)
    if save_outputs:
        _validate_safe_output_directory(target_dir, Path(config.data_root))
        target_dir.mkdir(parents=True, exist_ok=True)

    prep_cfg = config.preprocessing
    manifest_records: list[dict[str, Any]] = []
    all_qc_records: list[dict[str, Any]] = []
    generated_npz_files: list[str] = []

    for _, m_row in manifest_df.iterrows():
        rel_path = str(m_row["relative_source_path"]).replace("\\", "/")
        edf_path = (Path(config.data_root) / rel_path).resolve()
        rec_manifest, qc_tuple, _ = preprocess_single_recording(
            edf_path=edf_path,
            manifest_row=m_row,
            preprocessing_config=prep_cfg,
            output_dir=target_dir if save_outputs else None,
            save_npz=save_outputs,
        )
        manifest_records.append(asdict(rec_manifest))
        all_qc_records.extend(asdict(q) for q in qc_tuple)
        if save_outputs:
            generated_npz_files.append(str(target_dir / rec_manifest.preprocessed_npz_relpath))

    out_manifest_df = pd.DataFrame(manifest_records)
    out_qc_df = pd.DataFrame(all_qc_records)

    n_recs = len(out_manifest_df)
    n_ch_traces = len(out_qc_df)
    n_flagged_traces = int(out_qc_df["screening_flag_any"].sum()) if n_ch_traces > 0 else 0
    n_low_std_traces = int(out_qc_df["screening_flag_low_std"].sum()) if n_ch_traces > 0 else 0
    n_high_ptp_traces = int(out_qc_df["screening_flag_high_ptp"].sum()) if n_ch_traces > 0 else 0
    n_non_finite_traces = (
        int(out_qc_df["screening_flag_non_finite"].sum()) if n_ch_traces > 0 else 0
    )

    n_unf_0 = int(out_manifest_df["is_unflagged_allow_0"].sum()) if n_recs > 0 else 0
    n_unf_1 = int(out_manifest_df["is_unflagged_allow_1"].sum()) if n_recs > 0 else 0
    n_unf_2 = int(out_manifest_df["is_unflagged_allow_2"].sum()) if n_recs > 0 else 0
    n_retained = int(out_manifest_df["recording_retained_in_outputs"].sum()) if n_recs > 0 else 0

    target_pad_samples = int(
        round(prep_cfg.filter_pad_cycles * prep_cfg.expected_sfreq_hz / prep_cfg.bandpass_low_hz)
    )
    eff_pads = out_manifest_df["effective_padlen_samples"].to_numpy(dtype=int)

    summary_dict: dict[str, Any] = {
        "phase": "Phase 3B Reusable EEG Preprocessing Pipeline",
        "n_recordings_expected": int(len(manifest_df)),
        "n_recordings_processed": n_recs,
        "n_channel_traces_processed": n_ch_traces,
        "preprocessing_config": {
            "expected_eeg_channels": list(prep_cfg.expected_eeg_channels),
            "expected_sfreq_hz": prep_cfg.expected_sfreq_hz,
            "physical_unit": "uV",
            "apply_mean_subtraction": prep_cfg.apply_mean_subtraction,
            "apply_bandpass_filter": prep_cfg.apply_bandpass_filter,
            "bandpass_low_hz": prep_cfg.bandpass_low_hz,
            "bandpass_high_hz": prep_cfg.bandpass_high_hz,
            "butterworth_order": prep_cfg.butterworth_order,
            "filter_padtype": prep_cfg.filter_padtype,
            "filter_pad_cycles": prep_cfg.filter_pad_cycles,
            "reference_mode": prep_cfg.reference_mode,
            "qc_screening_min_std_uV": prep_cfg.qc_screening_min_std_uV,
            "qc_screening_max_ptp_uV": prep_cfg.qc_screening_max_ptp_uV,
            "boundary_validation_window_sec": prep_cfg.boundary_validation_window_sec,
        },
        "padding_and_boundary_summary": {
            "filter_padtype": prep_cfg.filter_padtype,
            "filter_padlen_formula": (
                str(out_manifest_df["filter_padlen_formula"].iloc[0])
                if not out_manifest_df.empty
                else ""
            ),
            "target_padlen_samples": target_pad_samples,
            "target_padlen_sec": round(target_pad_samples / prep_cfg.expected_sfreq_hz, 4),
            "effective_padlen_samples_min": int(np.min(eff_pads)) if eff_pads.size > 0 else 0,
            "effective_padlen_samples_max": int(np.max(eff_pads)) if eff_pads.size > 0 else 0,
            "n_recordings_padlen_capped": int(out_manifest_df["padlen_was_capped"].sum()),
            "boundary_validation_window_sec": prep_cfg.boundary_validation_window_sec,
            "n_boundary_finite_traces": int(
                out_qc_df["post_filter_boundary_1s_std_uV"].notna().sum()
            ),
            "boundary_to_interior_std_ratio": _quantiles_dict(
                out_qc_df["post_filter_boundary_to_interior_std_ratio"].to_numpy(dtype=float)
            ),
        },
        "signal_distribution_summary": {
            "subtracted_dc_mean_uV": _quantiles_dict(
                out_qc_df["subtracted_dc_mean_uV"].to_numpy(dtype=float)
            ),
            "post_filter_mean_uV": _quantiles_dict(
                out_qc_df["post_filter_mean_uV"].to_numpy(dtype=float)
            ),
            "post_filter_max_abs_mean_uV": (
                round(float(out_manifest_df["post_filter_max_abs_mean_uV"].max()), 8)
                if not out_manifest_df.empty
                else 0.0
            ),
            "pre_filter_std_uV": _quantiles_dict(
                out_qc_df["pre_filter_std_uV"].to_numpy(dtype=float)
            ),
            "post_filter_std_uV": _quantiles_dict(
                out_qc_df["post_filter_std_uV"].to_numpy(dtype=float)
            ),
            "pre_filter_ptp_uV": _quantiles_dict(
                out_qc_df["pre_filter_ptp_uV"].to_numpy(dtype=float)
            ),
            "post_filter_ptp_uV": _quantiles_dict(
                out_qc_df["post_filter_ptp_uV"].to_numpy(dtype=float)
            ),
        },
        "qc_screening_summary": {
            "terminology_note": SCREENING_NOTE_TEXT,
            "n_flagged_channel_traces": n_flagged_traces,
            "flagged_channel_traces_pct": (
                round(100.0 * n_flagged_traces / n_ch_traces, 2) if n_ch_traces > 0 else 0.0
            ),
            "n_low_std_channel_traces": n_low_std_traces,
            "n_high_ptp_channel_traces": n_high_ptp_traces,
            "n_non_finite_channel_traces": n_non_finite_traces,
            "n_unflagged_recordings_allow_0": n_unf_0,
            "unflagged_recordings_allow_0_pct": (
                round(100.0 * n_unf_0 / n_recs, 2) if n_recs > 0 else 0.0
            ),
            "n_unflagged_recordings_allow_1": n_unf_1,
            "unflagged_recordings_allow_1_pct": (
                round(100.0 * n_unf_1 / n_recs, 2) if n_recs > 0 else 0.0
            ),
            "n_unflagged_recordings_allow_2": n_unf_2,
            "unflagged_recordings_allow_2_pct": (
                round(100.0 * n_unf_2 / n_recs, 2) if n_recs > 0 else 0.0
            ),
            "n_recordings_retained_in_outputs": n_retained,
            "recordings_retained_pct": (
                round(100.0 * n_retained / n_recs, 2) if n_recs > 0 else 0.0
            ),
        },
    }

    generated_files: list[str] = []
    generated_figures: list[str] = []

    if save_outputs:
        manifest_csv_path = target_dir / "preprocessed_manifest.csv"
        out_manifest_df.to_csv(manifest_csv_path, index=False)
        generated_files.append(str(manifest_csv_path))

        qc_csv_path = target_dir / "channel_qc_metrics.csv"
        out_qc_df.to_csv(qc_csv_path, index=False)
        generated_files.append(str(qc_csv_path))

        summary_json_path = target_dir / "phase3b_preprocessing_summary.json"
        summary_json_path.write_text(json.dumps(summary_dict, indent=2), encoding="utf-8")
        generated_files.append(str(summary_json_path))

        report_md_path = target_dir / "phase3b_preprocessing_report.md"
        report_md_path.write_text(_build_phase3b_markdown_report(summary_dict), encoding="utf-8")
        generated_files.append(str(report_md_path))

        if prep_cfg.generate_figures:
            generated_figures = _generate_phase3b_figures(
                manifest_df=out_manifest_df,
                channel_qc_df=out_qc_df,
                preprocessing_config=prep_cfg,
                figures_dir=target_dir / "figures",
            )

        if verify_completeness:
            verification_info = verify_phase3b_outputs(
                output_dir=target_dir,
                expected_manifest_df=manifest_df,
                preprocessing_config=prep_cfg,
            )
            summary_dict["output_verification"] = verification_info
            summary_json_path.write_text(json.dumps(summary_dict, indent=2), encoding="utf-8")

    return Phase3BPreprocessingResult(
        manifest_df=out_manifest_df,
        channel_qc_df=out_qc_df,
        summary=summary_dict,
        generated_files=generated_files,
        generated_npz_files=generated_npz_files,
        generated_figures=generated_figures,
    )
