"""Phase 3A read-only preprocessing diagnostic helpers.

This module evaluates candidate preprocessing choices in memory without modifying
raw EDF files, Phase 1 outputs, or Phase 2 outputs, and without permanently
excluding any channels or recordings:

1. Pre-filter vs. post-filter (candidate per-channel mean subtraction + 4th-order
   Butterworth 0.5-45 Hz zero-phase band-pass at 128 Hz) signal distributions.
2. Exploratory threshold grid evaluation (min_std_uV in {0.001, 0.25, 0.5, 1.0},
   max_ptp_uV in {500, 1000, 1500, 2000}, and recording policies allowing 0, 1, or 2
   flagged channels), reported separately for pre-filter and post-filter stages.
3. Unverified filename-prefix and EDF header metadata cross-tabulation.
4. No re-referencing ('none') vs. Common Average Reference ('car') comparison on
   identical filtered signals to quantify potential cross-channel contamination.
5. Full-recording duration provenance and candidate sliding-window count simulation
   without inferring undocumented imagined-speech onset/offset intervals.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy import signal as sp_signal

from imagined_speech.config import DEFAULT_PRIMARY_EEG_MONTAGE, ProjectConfig
from imagined_speech.data.edf_audit import (
    RawEDFChannelHeader,
    RawEDFFixedHeader,
    clean_channel_name,
    read_raw_edf_header,
)
from imagined_speech.data.signal_quality import (
    _extract_native_unit_signals,
    load_and_validate_manifest,
)
from imagined_speech.exceptions import SignalQualityError

logger = logging.getLogger("imagined_speech.data.preprocessing_diagnostic")

DEFAULT_CANDIDATE_MIN_STD_UV: tuple[float, ...] = (0.001, 0.25, 0.5, 1.0)
DEFAULT_CANDIDATE_MAX_PTP_UV: tuple[float, ...] = (500.0, 1000.0, 1500.0, 2000.0)
DEFAULT_MAX_FLAGGED_CHANNELS_POLICIES: tuple[int, ...] = (0, 1, 2)
DEFAULT_CANDIDATE_BANDPASS_LOW_HZ: float = 0.5
DEFAULT_CANDIDATE_BANDPASS_HIGH_HZ: float = 45.0
DEFAULT_CANDIDATE_BUTTER_ORDER: int = 4
DEFAULT_CANDIDATE_WINDOW_SCHEMES: tuple[tuple[float, float], ...] = (
    (2.0, 1.0),
    (2.0, 2.0),
    (4.0, 2.0),
    (4.0, 4.0),
    (11.0, 11.0),
)


@dataclass
class Phase3ADiagnosticResult:
    """Container for all read-only Phase 3A diagnostic tables, summaries, and figures."""

    channel_stats_df: pd.DataFrame
    threshold_summary_df: pd.DataFrame
    threshold_label_balance_df: pd.DataFrame
    prefix_metadata_df: pd.DataFrame
    car_comparison_df: pd.DataFrame
    windowing_simulation_df: pd.DataFrame
    summary: dict[str, Any] = field(default_factory=dict)
    generated_files: list[str] = field(default_factory=list)
    generated_figures: list[str] = field(default_factory=list)


def apply_candidate_mean_sub_and_bandpass(
    signals: np.ndarray,
    sfreq_hz: float = 128.0,
    low_hz: float = DEFAULT_CANDIDATE_BANDPASS_LOW_HZ,
    high_hz: float = DEFAULT_CANDIDATE_BANDPASS_HIGH_HZ,
    order: int = DEFAULT_CANDIDATE_BUTTER_ORDER,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply candidate per-channel mean subtraction and zero-phase Butterworth band-pass filter.

    Operates strictly on an in-memory float64 copy of ``signals`` without mutating the
    caller's array. Channels containing non-finite values (NaN or Inf) produce NaN
    filtered traces so they remain explicitly flagged downstream.

    Boundary Padding Strategy
    -------------------------
    Uses symmetric reflection (``padtype="even"``) with:
    ``effective_padlen = min(n_samples - 1, max(default_padlen, int(round(3.0 * sfreq_hz / low_hz))))``
    where ``default_padlen = 3 * (2 * len(sos) + 1)`` (27 samples for a 4th-order band-pass
    filter with 4 second-order sections). At ``sfreq_hz = 128.0`` Hz and ``low_hz = 0.5`` Hz,
    ``int(round(3.0 * 128.0 / 0.5))`` equals **768 samples (6.0 seconds, or 3 full periods of
    the 0.5 Hz cutoff)**. Because all recordings in the dataset span 11-22 seconds
    (1408-2816 samples at 128 Hz, so ``n_samples - 1 >= 1407``), every recording in that range
    receives a uniform 768-sample (6.0 s) symmetric reflection pad.

    Parameters
    ----------
    signals : np.ndarray
        1D array ``(n_samples,)`` or 2D array ``(n_channels, n_samples)`` in native units (uV).
    sfreq_hz : float
        Sampling rate in Hz (expected 128.0 Hz).
    low_hz : float
        Low cutoff frequency in Hz (default 0.5 Hz).
    high_hz : float
        High cutoff frequency in Hz (default 45.0 Hz).
    order : int
        Butterworth filter order before forward-backward pass (default 4).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(filtered_signals, subtracted_channel_means)`` matching the input dimensionality.
    """
    if not np.isfinite(sfreq_hz) or sfreq_hz <= 0:
        raise ValueError(f"sfreq_hz must be a positive finite number, got {sfreq_hz!r}.")
    nyquist = sfreq_hz / 2.0
    if not (np.isfinite(low_hz) and np.isfinite(high_hz) and 0.0 < low_hz < high_hz < nyquist):
        raise ValueError(
            f"Invalid band-pass cutoffs low_hz={low_hz}, high_hz={high_hz} for Nyquist={nyquist} Hz."
        )
    if isinstance(order, bool) or not isinstance(order, int) or order < 1:
        raise ValueError(f"Butterworth order must be an integer >= 1, got {order!r}.")

    arr = np.asarray(signals, dtype=np.float64).copy()
    was_1d = arr.ndim == 1
    if was_1d:
        arr = arr[np.newaxis, :]
    elif arr.ndim != 2:
        raise ValueError(f"signals must be 1D or 2D, got shape {arr.shape}.")

    n_channels, n_samples = arr.shape
    if n_channels == 0:
        raise ValueError("signals must contain at least 1 channel (got 0 channels).")

    try:
        sos = sp_signal.butter(
            N=order,
            Wn=[low_hz, high_hz],
            btype="bandpass",
            fs=sfreq_hz,
            output="sos",
        )
    except Exception as exc:
        raise SignalQualityError(
            f"Failed to construct Butterworth band-pass filter (order={order}, "
            f"low_hz={low_hz}, high_hz={high_hz}, sfreq_hz={sfreq_hz}): {exc}"
        ) from exc

    default_padlen = 3 * (2 * len(sos) + 1)
    if n_samples <= default_padlen:
        raise ValueError(
            f"Signal length n_samples={n_samples} is too short for zero-phase filtering "
            f"(requires > {default_padlen} samples)."
        )
    target_cycle_padlen = int(round(3.0 * sfreq_hz / low_hz))
    effective_padlen = min(n_samples - 1, max(default_padlen, target_cycle_padlen))

    filtered = np.full_like(arr, np.nan, dtype=np.float64)
    means = np.full(n_channels, np.nan, dtype=np.float64)

    finite_row_mask = np.all(np.isfinite(arr), axis=1)
    if np.any(finite_row_mask):
        valid_rows = arr[finite_row_mask]
        row_means = np.mean(valid_rows, axis=1, keepdims=True)
        centered = valid_rows - row_means
        try:
            filtered_valid = sp_signal.sosfiltfilt(
                sos,
                centered,
                axis=1,
                padtype="even",
                padlen=effective_padlen,
            )
        except Exception as exc:
            raise SignalQualityError(
                f"Zero-phase sosfiltfilt failed on signal of shape {valid_rows.shape} "
                f"with padlen={effective_padlen}: {exc}"
            ) from exc
        filtered[finite_row_mask] = filtered_valid
        means[finite_row_mask] = row_means[:, 0]

    if was_1d:
        return filtered[0], means[0]
    return filtered, means


def apply_common_average_reference(signals_2d: np.ndarray) -> np.ndarray:
    """Subtract the instantaneous mean across channels at each sample on a derived copy.

    Parameters
    ----------
    signals_2d : np.ndarray
        2D array of shape ``(n_channels, n_samples)``.

    Returns
    -------
    np.ndarray
        Common-average-referenced copy of shape ``(n_channels, n_samples)``.
    """
    arr = np.asarray(signals_2d, dtype=np.float64).copy()
    if arr.ndim != 2 or arr.shape[0] < 2:
        raise ValueError(
            f"apply_common_average_reference requires a 2D array with >= 2 channels, got {arr.shape}."
        )
    common_avg = np.mean(arr, axis=0, keepdims=True)
    return arr - common_avg


def extract_candidate_eeg_signals_uV(
    edf_path: Path | str,
    expected_eeg_channels: Sequence[str] = DEFAULT_PRIMARY_EEG_MONTAGE,
) -> tuple[np.ndarray, list[str], float, RawEDFFixedHeader, list[RawEDFChannelHeader]]:
    """Read the 14 candidate EEG channels in native physical units (uV) in canonical montage order.

    Reuses ``read_raw_edf_header``, ``clean_channel_name``, and ``_extract_native_unit_signals``.
    """
    path = Path(edf_path)
    raw_header = read_raw_edf_header(path)
    hdr_by_idx = {ch.channel_index: ch for ch in raw_header.channels}

    raw_mne = mne.io.read_raw_edf(str(path), preload=True, verbose="ERROR")
    try:
        sfreq_hz = float(raw_mne.info["sfreq"])
        native_signals, _ = _extract_native_unit_signals(raw_mne, hdr_by_idx)
    finally:
        raw_mne.close()

    clean_to_idx: dict[str, int] = {}
    for ch in raw_header.channels:
        cleaned = clean_channel_name(ch.label).upper()
        if cleaned not in clean_to_idx:
            clean_to_idx[cleaned] = ch.channel_index

    ordered_indices: list[int] = []
    ordered_names: list[str] = []
    ordered_headers: list[RawEDFChannelHeader] = []
    missing: list[str] = []
    for target in expected_eeg_channels:
        key = target.strip().upper()
        if key not in clean_to_idx:
            missing.append(target)
        else:
            idx = clean_to_idx[key]
            ordered_indices.append(idx)
            ordered_names.append(clean_channel_name(raw_header.channels[idx].label))
            ordered_headers.append(raw_header.channels[idx])

    if missing:
        raise SignalQualityError(
            f"Recording {path} is missing expected candidate EEG channels: {missing}"
        )

    eeg_signals = np.asarray(native_signals[ordered_indices], dtype=np.float64)
    return eeg_signals, ordered_names, sfreq_hz, raw_header, ordered_headers


def _mean_offdiag_abs_correlation(signals_2d: np.ndarray) -> float | None:
    """Compute the mean absolute off-diagonal Pearson correlation across channels."""
    if signals_2d.ndim != 2 or signals_2d.shape[0] < 2:
        return None
    if not np.all(np.isfinite(signals_2d)):
        return None
    stds = np.std(signals_2d, axis=1)
    valid = stds > 1e-12
    if int(np.sum(valid)) < 2:
        return None
    corr = np.corrcoef(signals_2d[valid])
    n_valid = corr.shape[0]
    mask = ~np.eye(n_valid, dtype=bool)
    vals = np.abs(corr[mask])
    finite_vals = vals[np.isfinite(vals)]
    if finite_vals.size == 0:
        return None
    return float(np.mean(finite_vals))


def compute_recording_diagnostic_metrics(
    edf_path: Path | str,
    manifest_row: dict[str, Any] | pd.Series,
    expected_eeg_channels: Sequence[str] = DEFAULT_PRIMARY_EEG_MONTAGE,
    low_hz: float = DEFAULT_CANDIDATE_BANDPASS_LOW_HZ,
    high_hz: float = DEFAULT_CANDIDATE_BANDPASS_HIGH_HZ,
    butter_order: int = DEFAULT_CANDIDATE_BUTTER_ORDER,
    rail_rel_tol: float = 1.0e-3,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compute pre-filter, post-filter (0.5-45 Hz), and CAR diagnostic metrics for one recording."""
    eeg_raw, ch_names, sfreq_hz, _, ch_headers = extract_candidate_eeg_signals_uV(
        edf_path=edf_path,
        expected_eeg_channels=expected_eeg_channels,
    )
    n_channels, n_samples = eeg_raw.shape
    duration_sec = float(n_samples / sfreq_hz) if sfreq_hz > 0 else 0.0

    eeg_filt, _ = apply_candidate_mean_sub_and_bandpass(
        signals=eeg_raw,
        sfreq_hz=sfreq_hz,
        low_hz=low_hz,
        high_hz=high_hz,
        order=butter_order,
    )
    eeg_car = apply_common_average_reference(eeg_filt)

    rel_path = str(manifest_row["relative_source_path"])
    orig_fname = str(manifest_row.get("original_filename", Path(edf_path).name))
    task_cat = str(manifest_row["task_category"])
    raw_label = (
        None if pd.isna(manifest_row.get("raw_label")) else str(manifest_row.get("raw_label"))
    )
    norm_label = (
        None
        if pd.isna(manifest_row.get("normalized_label"))
        else str(manifest_row.get("normalized_label"))
    )
    prefix = (
        None
        if pd.isna(manifest_row.get("unverified_candidate_prefix"))
        else str(manifest_row.get("unverified_candidate_prefix"))
    )

    channel_rows: list[dict[str, Any]] = []
    for idx in range(n_channels):
        raw_1d = eeg_raw[idx]
        filt_1d = eeg_filt[idx]
        car_1d = eeg_car[idx]
        hdr_ch = ch_headers[idx]

        pre_non_finite = bool(not np.all(np.isfinite(raw_1d)))
        post_non_finite = bool(not np.all(np.isfinite(filt_1d)))

        if not pre_non_finite:
            pre_mean = float(np.mean(raw_1d))
            pre_std = float(np.std(raw_1d))
            q25, q50, q75 = np.percentile(raw_1d, [25.0, 50.0, 75.0])
            pre_median = float(q50)
            pre_iqr = float(q75 - q25)
            pre_min = float(np.min(raw_1d))
            pre_max = float(np.max(raw_1d))
            pre_ptp = float(pre_max - pre_min)
            pre_max_abs_centered = float(np.max(np.abs(raw_1d - pre_mean)))

            if (
                hdr_ch.physical_min is not None
                and hdr_ch.physical_max is not None
                and hdr_ch.physical_max > hdr_ch.physical_min
            ):
                span = float(hdr_ch.physical_max - hdr_ch.physical_min)
                tol = rail_rel_tol * span
                near_rail = (raw_1d <= float(hdr_ch.physical_min) + tol) | (
                    raw_1d >= float(hdr_ch.physical_max) - tol
                )
                pre_rail_frac = float(np.mean(near_rail))
            else:
                pre_rail_frac = 0.0
        else:
            pre_mean = pre_std = pre_median = pre_iqr = pre_min = pre_max = pre_ptp = (
                pre_max_abs_centered
            ) = pre_rail_frac = math.nan

        if not post_non_finite:
            post_mean = float(np.mean(filt_1d))
            post_std = float(np.std(filt_1d))
            fq25, fq50, fq75 = np.percentile(filt_1d, [25.0, 50.0, 75.0])
            post_median = float(fq50)
            post_iqr = float(fq75 - fq25)
            post_min = float(np.min(filt_1d))
            post_max = float(np.max(filt_1d))
            post_ptp = float(post_max - post_min)
            abs_filt = np.abs(filt_1d)
            post_max_abs = float(np.max(abs_filt))
            exc_100 = float(np.mean(abs_filt > 100.0))
            exc_250 = float(np.mean(abs_filt > 250.0))
            exc_500 = float(np.mean(abs_filt > 500.0))
            exc_1000 = float(np.mean(abs_filt > 1000.0))
            std_ratio = float(post_std / pre_std) if pre_std and pre_std > 1e-15 else 0.0
            ptp_ratio = float(post_ptp / pre_ptp) if pre_ptp and pre_ptp > 1e-15 else 0.0
        else:
            post_mean = post_std = post_median = post_iqr = post_min = post_max = post_ptp = (
                post_max_abs
            ) = exc_100 = exc_250 = exc_500 = exc_1000 = std_ratio = ptp_ratio = math.nan

        if np.all(np.isfinite(car_1d)):
            car_std = float(np.std(car_1d))
            car_ptp = float(np.ptp(car_1d))
            car_max_abs = float(np.max(np.abs(car_1d)))
            car_std_ratio = float(car_std / post_std) if post_std and post_std > 1e-15 else 0.0
            car_ptp_ratio = float(car_ptp / post_ptp) if post_ptp and post_ptp > 1e-15 else 0.0
        else:
            car_std = car_ptp = car_max_abs = car_std_ratio = car_ptp_ratio = math.nan

        channel_rows.append(
            {
                "relative_source_path": rel_path,
                "original_filename": orig_fname,
                "task_category": task_cat,
                "raw_label": raw_label,
                "normalized_label": norm_label,
                "unverified_candidate_prefix": prefix,
                "sfreq_hz": sfreq_hz,
                "n_samples": n_samples,
                "duration_sec": duration_sec,
                "channel_index": int(hdr_ch.channel_index),
                "raw_channel_name": ch_names[idx],
                "pre_has_nan_or_inf": pre_non_finite,
                "pre_mean_uV": pre_mean,
                "pre_std_uV": pre_std,
                "pre_median_uV": pre_median,
                "pre_iqr_uV": pre_iqr,
                "pre_min_uV": pre_min,
                "pre_max_uV": pre_max,
                "pre_ptp_uV": pre_ptp,
                "pre_max_abs_centered_uV": pre_max_abs_centered,
                "pre_rail_fraction": pre_rail_frac,
                "post_has_nan_or_inf": post_non_finite,
                "post_mean_uV": post_mean,
                "post_std_uV": post_std,
                "post_median_uV": post_median,
                "post_iqr_uV": post_iqr,
                "post_min_uV": post_min,
                "post_max_uV": post_max,
                "post_ptp_uV": post_ptp,
                "post_max_abs_uV": post_max_abs,
                "post_exceeds_100uV_frac": exc_100,
                "post_exceeds_250uV_frac": exc_250,
                "post_exceeds_500uV_frac": exc_500,
                "post_exceeds_1000uV_frac": exc_1000,
                "std_retention_ratio": std_ratio,
                "ptp_reduction_ratio": ptp_ratio,
                "car_post_std_uV": car_std,
                "car_post_ptp_uV": car_ptp,
                "car_post_max_abs_uV": car_max_abs,
                "car_vs_none_std_ratio": car_std_ratio,
                "car_vs_none_ptp_ratio": car_ptp_ratio,
            }
        )

    post_stds_none = np.array([r["post_std_uV"] for r in channel_rows], dtype=np.float64)
    post_ptps_none = np.array([r["post_ptp_uV"] for r in channel_rows], dtype=np.float64)
    post_stds_car = np.array([r["car_post_std_uV"] for r in channel_rows], dtype=np.float64)
    post_ptps_car = np.array([r["car_post_ptp_uV"] for r in channel_rows], dtype=np.float64)

    n_gt_1000 = int(np.sum(post_ptps_none > 1000.0)) if np.all(np.isfinite(post_ptps_none)) else 0
    n_gt_500 = int(np.sum(post_ptps_none > 500.0)) if np.all(np.isfinite(post_ptps_none)) else 0

    unflagged_1000_mask = np.isfinite(post_ptps_none) & (post_ptps_none <= 1000.0)
    if np.any(unflagged_1000_mask):
        unflagged_med_ptp_none = float(np.median(post_ptps_none[unflagged_1000_mask]))
        unflagged_med_ptp_car = float(np.median(post_ptps_car[unflagged_1000_mask]))
        unflagged_ptp_ratio = (
            float(unflagged_med_ptp_car / unflagged_med_ptp_none)
            if unflagged_med_ptp_none > 1e-12
            else 1.0
        )
    else:
        unflagged_med_ptp_none = math.nan
        unflagged_med_ptp_car = math.nan
        unflagged_ptp_ratio = math.nan

    car_row = {
        "relative_source_path": rel_path,
        "task_category": task_cat,
        "raw_label": raw_label,
        "normalized_label": norm_label,
        "unverified_candidate_prefix": prefix,
        "n_channels_post_ptp_gt_1000uV": n_gt_1000,
        "n_channels_post_ptp_gt_500uV": n_gt_500,
        "median_channel_std_none_uV": float(np.nanmedian(post_stds_none)),
        "median_channel_std_car_uV": float(np.nanmedian(post_stds_car)),
        "max_channel_ptp_none_uV": float(np.nanmax(post_ptps_none)),
        "max_channel_ptp_car_uV": float(np.nanmax(post_ptps_car)),
        "unflagged_channels_median_ptp_none_uV": unflagged_med_ptp_none,
        "unflagged_channels_median_ptp_car_uV": unflagged_med_ptp_car,
        "unflagged_channels_ptp_ratio_car_over_none": unflagged_ptp_ratio,
        "mean_abs_interchannel_corr_none": _mean_offdiag_abs_correlation(eeg_filt),
        "mean_abs_interchannel_corr_car": _mean_offdiag_abs_correlation(eeg_car),
    }

    return channel_rows, car_row


def evaluate_threshold_grid(
    channel_stats_df: pd.DataFrame,
    min_std_grid_uV: Sequence[float] = DEFAULT_CANDIDATE_MIN_STD_UV,
    max_ptp_grid_uV: Sequence[float] = DEFAULT_CANDIDATE_MAX_PTP_UV,
    max_flagged_channels_policies: Sequence[int] = DEFAULT_MAX_FLAGGED_CHANNELS_POLICIES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate exploratory screening threshold rules separately on pre-filter and post-filter stats.

    Safeguard: Channels and recordings that satisfy a rule are described strictly as
    ``unflagged`` under that candidate rule, never as verified clean data.
    """
    required_cols = {
        "relative_source_path",
        "task_category",
        "normalized_label",
        "raw_channel_name",
        "pre_has_nan_or_inf",
        "pre_std_uV",
        "pre_ptp_uV",
        "post_has_nan_or_inf",
        "post_std_uV",
        "post_ptp_uV",
    }
    missing = required_cols - set(channel_stats_df.columns)
    if missing:
        raise ValueError(f"channel_stats_df is missing required columns: {sorted(missing)}")

    rec_meta_df = (
        channel_stats_df[["relative_source_path", "task_category", "normalized_label"]]
        .drop_duplicates(subset=["relative_source_path"])
        .sort_values("relative_source_path")
        .reset_index(drop=True)
    )
    n_total_recs = len(rec_meta_df)
    n_total_ch_rows = len(channel_stats_df)

    task_totals = rec_meta_df["task_category"].value_counts().to_dict()
    label_totals_df = (
        rec_meta_df.groupby(["task_category", "normalized_label"], dropna=False)
        .size()
        .reset_index(name="n_total_recordings")
    )

    summary_rows: list[dict[str, Any]] = []
    label_balance_rows: list[dict[str, Any]] = []

    stages = (
        ("pre_filter", "pre_has_nan_or_inf", "pre_std_uV", "pre_ptp_uV"),
        ("post_filter", "post_has_nan_or_inf", "post_std_uV", "post_ptp_uV"),
    )

    for stage_name, nan_col, std_col, ptp_col in stages:
        non_finite_mask = channel_stats_df[nan_col].astype(bool) | ~np.isfinite(
            channel_stats_df[std_col]
        ) | ~np.isfinite(channel_stats_df[ptp_col])

        for min_std in min_std_grid_uV:
            low_std_mask = (~non_finite_mask) & (channel_stats_df[std_col] < float(min_std))
            for max_ptp in max_ptp_grid_uV:
                high_ptp_mask = (~non_finite_mask) & (channel_stats_df[ptp_col] > float(max_ptp))
                flagged_ch_mask = non_finite_mask | low_std_mask | high_ptp_mask

                n_flagged_ch = int(flagged_ch_mask.sum())
                n_low_std = int(low_std_mask.sum())
                n_high_ptp = int(high_ptp_mask.sum())
                n_non_finite = int(non_finite_mask.sum())

                flagged_per_rec = (
                    channel_stats_df.assign(_flagged=flagged_ch_mask.astype(int))
                    .groupby("relative_source_path")["_flagged"]
                    .sum()
                )
                rec_eval = rec_meta_df.copy()
                rec_eval["n_flagged_channels"] = (
                    rec_eval["relative_source_path"].map(flagged_per_rec).fillna(0).astype(int)
                )

                for max_allowed in max_flagged_channels_policies:
                    rule_id = (
                        f"{stage_name}__std_ge_{min_std:g}uV__"
                        f"ptp_le_{max_ptp:g}uV__allow_{max_allowed}_flagged_ch"
                    )
                    rec_eval["is_unflagged_under_rule"] = (
                        rec_eval["n_flagged_channels"] <= int(max_allowed)
                    )

                    n_unflagged_recs = int(rec_eval["is_unflagged_under_rule"].sum())
                    n_flagged_recs = n_total_recs - n_unflagged_recs
                    unflagged_pct = (
                        round(100.0 * n_unflagged_recs / n_total_recs, 2)
                        if n_total_recs > 0
                        else 0.0
                    )

                    task_unflagged = (
                        rec_eval[rec_eval["is_unflagged_under_rule"]]
                        .groupby("task_category")
                        .size()
                        .to_dict()
                    )

                    def _task_stats(t_name: str) -> tuple[int, float]:
                        tot = int(task_totals.get(t_name, 0))
                        unf = int(task_unflagged.get(t_name, 0))
                        pct = round(100.0 * unf / tot, 2) if tot > 0 else 0.0
                        return unf, pct

                    unf_char, pct_char = _task_stats("Char")
                    unf_digit, pct_digit = _task_stats("Digit")
                    unf_image, pct_image = _task_stats("Image")

                    lbl_unflagged = (
                        rec_eval[rec_eval["is_unflagged_under_rule"]]
                        .groupby(["task_category", "normalized_label"], dropna=False)
                        .size()
                        .reset_index(name="n_unflagged_recordings")
                    )
                    lbl_merged = label_totals_df.merge(
                        lbl_unflagged,
                        on=["task_category", "normalized_label"],
                        how="left",
                    )
                    lbl_merged["n_unflagged_recordings"] = (
                        lbl_merged["n_unflagged_recordings"].fillna(0).astype(int)
                    )
                    lbl_merged["n_flagged_recordings"] = (
                        lbl_merged["n_total_recordings"] - lbl_merged["n_unflagged_recordings"]
                    )
                    lbl_merged["unflagged_recording_pct"] = np.where(
                        lbl_merged["n_total_recordings"] > 0,
                        np.round(
                            100.0
                            * lbl_merged["n_unflagged_recordings"]
                            / lbl_merged["n_total_recordings"],
                            2,
                        ),
                        0.0,
                    )

                    unf_counts = lbl_merged["n_unflagged_recordings"].to_numpy(dtype=int)
                    n_zero_labels = int(np.sum(unf_counts == 0))
                    min_lbl = int(np.min(unf_counts)) if unf_counts.size > 0 else 0
                    med_lbl = float(np.median(unf_counts)) if unf_counts.size > 0 else 0.0
                    max_lbl = int(np.max(unf_counts)) if unf_counts.size > 0 else 0
                    imbalance_ratio = (
                        round(float(max_lbl / min_lbl), 4) if min_lbl > 0 else math.inf
                    )

                    summary_rows.append(
                        {
                            "rule_id": rule_id,
                            "signal_stage": stage_name,
                            "min_std_uV": float(min_std),
                            "max_ptp_uV": float(max_ptp),
                            "max_flagged_channels_allowed": int(max_allowed),
                            "n_total_channel_rows": n_total_ch_rows,
                            "n_flagged_channel_rows": n_flagged_ch,
                            "flagged_channel_pct": (
                                round(100.0 * n_flagged_ch / n_total_ch_rows, 2)
                                if n_total_ch_rows > 0
                                else 0.0
                            ),
                            "n_flagged_low_std_channels": n_low_std,
                            "n_flagged_high_ptp_channels": n_high_ptp,
                            "n_flagged_non_finite_channels": n_non_finite,
                            "n_total_recordings": n_total_recs,
                            "n_unflagged_recordings": n_unflagged_recs,
                            "n_flagged_recordings": n_flagged_recs,
                            "unflagged_recording_pct": unflagged_pct,
                            "n_unflagged_char": unf_char,
                            "unflagged_char_pct": pct_char,
                            "n_unflagged_digit": unf_digit,
                            "unflagged_digit_pct": pct_digit,
                            "n_unflagged_image": unf_image,
                            "unflagged_image_pct": pct_image,
                            "n_total_task_labels": int(len(lbl_merged)),
                            "n_labels_with_zero_unflagged": n_zero_labels,
                            "min_unflagged_per_label": min_lbl,
                            "median_unflagged_per_label": med_lbl,
                            "max_unflagged_per_label": max_lbl,
                            "max_to_min_label_ratio": imbalance_ratio,
                        }
                    )

                    for _, l_row in lbl_merged.iterrows():
                        label_balance_rows.append(
                            {
                                "rule_id": rule_id,
                                "signal_stage": stage_name,
                                "min_std_uV": float(min_std),
                                "max_ptp_uV": float(max_ptp),
                                "max_flagged_channels_allowed": int(max_allowed),
                                "task_category": str(l_row["task_category"]),
                                "normalized_label": str(l_row["normalized_label"]),
                                "n_total_recordings": int(l_row["n_total_recordings"]),
                                "n_unflagged_recordings": int(l_row["n_unflagged_recordings"]),
                                "n_flagged_recordings": int(l_row["n_flagged_recordings"]),
                                "unflagged_recording_pct": float(
                                    l_row["unflagged_recording_pct"]
                                ),
                            }
                        )

    return pd.DataFrame(summary_rows), pd.DataFrame(label_balance_rows)


def audit_prefixes_and_edf_metadata(
    manifest_df: pd.DataFrame,
    channel_stats_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Inspect unverified filename-prefix patterns and EDF header metadata without assuming participant identity."""
    m_df = manifest_df.copy()
    m_df["prefix_key"] = m_df["unverified_candidate_prefix"].fillna("<NONE>").astype(str)

    ch_df = channel_stats_df.copy()
    ch_df["prefix_key"] = ch_df["unverified_candidate_prefix"].fillna("<NONE>").astype(str)
    ch_df["pre_low_std_1uV"] = ch_df["pre_std_uV"] < 1.0
    ch_df["pre_high_ptp_1000uV"] = ch_df["pre_ptp_uV"] > 1000.0
    ch_df["post_low_std_1uV"] = ch_df["post_std_uV"] < 1.0
    ch_df["post_low_std_0_5uV"] = ch_df["post_std_uV"] < 0.5
    ch_df["post_high_ptp_1000uV"] = ch_df["post_ptp_uV"] > 1000.0

    prefix_ch_agg = (
        ch_df.groupby("prefix_key")
        .agg(
            n_pre_low_std_1uV_channels=("pre_low_std_1uV", "sum"),
            n_pre_high_ptp_1000uV_channels=("pre_high_ptp_1000uV", "sum"),
            n_post_low_std_1uV_channels=("post_low_std_1uV", "sum"),
            n_post_low_std_0_5uV_channels=("post_low_std_0_5uV", "sum"),
            n_post_high_ptp_1000uV_channels=("post_high_ptp_1000uV", "sum"),
        )
        .to_dict(orient="index")
    )

    rows: list[dict[str, Any]] = []
    for prefix_val, grp in m_df.groupby("prefix_key", sort=True):
        task_counts = grp["task_category"].value_counts().to_dict()
        char_grp = grp[grp["task_category"] == "Char"]
        digit_grp = grp[grp["task_category"] == "Digit"]
        image_grp = grp[grp["task_category"] == "Image"]

        patient_ids = (
            sorted({str(x).strip() for x in grp["edf_patient_id_raw"].dropna()})
            if "edf_patient_id_raw" in grp.columns
            else []
        )
        rec_ids = (
            sorted({str(x).strip() for x in grp["edf_recording_id_raw"].dropna()})
            if "edf_recording_id_raw" in grp.columns
            else []
        )
        start_dates = (
            sorted({str(x).strip() for x in grp["edf_start_date"].dropna()})
            if "edf_start_date" in grp.columns
            else []
        )
        durations = grp["duration_sec"].dropna().to_numpy(dtype=float) if "duration_sec" in grp else np.array([])
        ch_info = prefix_ch_agg.get(prefix_val, {})

        rows.append(
            {
                "unverified_candidate_prefix": prefix_val,
                "identity_verification_status": "unverified_filename_prefix",
                "n_recordings_total": int(len(grp)),
                "n_char_recordings": int(task_counts.get("Char", 0)),
                "n_digit_recordings": int(task_counts.get("Digit", 0)),
                "n_image_recordings": int(task_counts.get("Image", 0)),
                "n_distinct_char_labels": int(char_grp["normalized_label"].nunique()),
                "n_distinct_digit_labels": int(digit_grp["normalized_label"].nunique()),
                "n_distinct_image_labels": int(image_grp["normalized_label"].nunique()),
                "n_distinct_edf_patient_ids": len(patient_ids),
                "edf_patient_ids_sample": "; ".join(patient_ids[:3]),
                "n_distinct_edf_recording_ids": len(rec_ids),
                "edf_recording_ids_sample": "; ".join(rec_ids[:3]),
                "n_distinct_edf_start_dates": len(start_dates),
                "edf_start_dates_sample": "; ".join(start_dates[:5]),
                "min_duration_sec": float(np.min(durations)) if durations.size > 0 else math.nan,
                "max_duration_sec": float(np.max(durations)) if durations.size > 0 else math.nan,
                "n_pre_low_std_1uV_channels": int(ch_info.get("n_pre_low_std_1uV_channels", 0)),
                "n_pre_high_ptp_1000uV_channels": int(
                    ch_info.get("n_pre_high_ptp_1000uV_channels", 0)
                ),
                "n_post_low_std_1uV_channels": int(ch_info.get("n_post_low_std_1uV_channels", 0)),
                "n_post_low_std_0_5uV_channels": int(
                    ch_info.get("n_post_low_std_0_5uV_channels", 0)
                ),
                "n_post_high_ptp_1000uV_channels": int(
                    ch_info.get("n_post_high_ptp_1000uV_channels", 0)
                ),
            }
        )

    prefix_df = pd.DataFrame(rows).sort_values("unverified_candidate_prefix").reset_index(drop=True)

    distinct_patient_ids = (
        sorted({str(x).strip() for x in m_df["edf_patient_id_raw"].dropna()})
        if "edf_patient_id_raw" in m_df.columns
        else []
    )
    distinct_rec_ids = (
        sorted({str(x).strip() for x in m_df["edf_recording_id_raw"].dropna()})
        if "edf_recording_id_raw" in m_df.columns
        else []
    )
    distinct_dates = (
        sorted({str(x).strip() for x in m_df["edf_start_date"].dropna()})
        if "edf_start_date" in m_df.columns
        else []
    )

    metadata_summary = {
        "identity_verification_note": (
            "Filename prefixes are treated strictly as unverified candidate prefixes. "
            "They are not assumed to represent verified human participants."
        ),
        "n_distinct_unverified_prefixes": int(len(prefix_df)),
        "recordings_per_prefix_min": int(prefix_df["n_recordings_total"].min()) if not prefix_df.empty else 0,
        "recordings_per_prefix_median": float(prefix_df["n_recordings_total"].median()) if not prefix_df.empty else 0.0,
        "recordings_per_prefix_max": int(prefix_df["n_recordings_total"].max()) if not prefix_df.empty else 0,
        "n_distinct_edf_patient_id_raw": len(distinct_patient_ids),
        "distinct_edf_patient_id_raw_values": distinct_patient_ids[:20],
        "n_distinct_edf_recording_id_raw": len(distinct_rec_ids),
        "distinct_edf_recording_id_raw_values": distinct_rec_ids[:20],
        "n_distinct_edf_start_dates": len(distinct_dates),
        "distinct_edf_start_dates_sample": distinct_dates[:20],
    }

    return prefix_df, metadata_summary


def simulate_windowing_provenance(
    manifest_df: pd.DataFrame,
    window_schemes: Sequence[tuple[float, float]] = DEFAULT_CANDIDATE_WINDOW_SCHEMES,
    default_sfreq_hz: float = 128.0,
) -> pd.DataFrame:
    """Simulate full-recording fixed-length window counts without inferring speech onset/offset times."""
    rows: list[dict[str, Any]] = []
    m_df = manifest_df.copy()
    if "sfreq_hz" not in m_df.columns:
        m_df["sfreq_hz"] = default_sfreq_hz
    if "n_samples" not in m_df.columns and "duration_sec" in m_df.columns:
        m_df["n_samples"] = np.round(m_df["duration_sec"] * m_df["sfreq_hz"]).astype(int)

    for window_sec, stride_sec in window_schemes:
        scheme_id = f"win_{window_sec:g}s_stride_{stride_sec:g}s"
        win_counts: list[int] = []
        trailing_samples: list[int] = []
        task_list: list[str] = []

        for _, r in m_df.iterrows():
            sfreq = float(r.get("sfreq_hz") or default_sfreq_hz)
            n_samp = int(r.get("n_samples") or round(float(r.get("duration_sec", 0.0)) * sfreq))
            win_samp = int(round(window_sec * sfreq))
            stride_samp = int(round(stride_sec * sfreq))

            if n_samp < win_samp or win_samp <= 0 or stride_samp <= 0:
                n_win = 0
                rem = n_samp
            else:
                n_win = 1 + (n_samp - win_samp) // stride_samp
                covered = win_samp + (n_win - 1) * stride_samp
                rem = n_samp - covered

            win_counts.append(n_win)
            trailing_samples.append(rem)
            task_list.append(str(r["task_category"]))

        win_arr = np.array(win_counts, dtype=int)
        rem_arr = np.array(trailing_samples, dtype=int)

        for scope in ("ALL", "Char", "Digit", "Image"):
            if scope == "ALL":
                mask = np.ones(len(win_arr), dtype=bool)
            else:
                mask = np.array([t == scope for t in task_list], dtype=bool)
            if not np.any(mask):
                continue

            sub_w = win_arr[mask]
            sub_r = rem_arr[mask]
            s_min = int(np.min(sub_w))
            s_max = int(np.max(sub_w))
            rows.append(
                {
                    "scheme_id": scheme_id,
                    "window_sec": float(window_sec),
                    "stride_sec": float(stride_sec),
                    "task_scope": scope,
                    "n_recordings": int(np.sum(mask)),
                    "total_windows": int(np.sum(sub_w)),
                    "min_windows_per_recording": s_min,
                    "median_windows_per_recording": float(np.median(sub_w)),
                    "max_windows_per_recording": s_max,
                    "max_to_min_recording_window_weight_ratio": (
                        round(float(s_max / s_min), 4) if s_min > 0 else math.inf
                    ),
                    "n_recordings_with_trailing_unwindowed_samples": int(np.sum(sub_r > 0)),
                    "median_trailing_unwindowed_samples": float(np.median(sub_r)),
                    "max_trailing_unwindowed_samples": int(np.max(sub_r)),
                }
            )

    return pd.DataFrame(rows)


def _generate_phase3a_figures(
    channel_stats_df: pd.DataFrame,
    threshold_summary_df: pd.DataFrame,
    car_comparison_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    figures_dir: Path,
) -> list[str]:
    """Generate 4 diagnostic summary figures in figures_dir."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    # Figure 1: Pre-filter vs Post-filter log10(std) and log10(ptp)
    fig1_path = figures_dir / "pre_vs_post_filter_distributions.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    pre_std = np.log10(np.clip(channel_stats_df["pre_std_uV"].dropna().to_numpy(), 1e-6, None))
    post_std = np.log10(np.clip(channel_stats_df["post_std_uV"].dropna().to_numpy(), 1e-6, None))
    axes[0].hist(pre_std, bins=50, alpha=0.55, label="Pre-filter std", color="#1f77b4")
    axes[0].hist(post_std, bins=50, alpha=0.55, label="Post-filter (0.5-45 Hz) std", color="#ff7f0e")
    axes[0].axvline(0.0, color="red", linestyle="--", linewidth=1.2, label="1.0 uV reference")
    axes[0].set_xlabel("log10(Standard Deviation [uV])")
    axes[0].set_ylabel("Channel Trace Count")
    axes[0].set_title("Candidate EEG Channel Standard Deviation")
    axes[0].legend(fontsize=8)

    pre_ptp = np.log10(np.clip(channel_stats_df["pre_ptp_uV"].dropna().to_numpy(), 1e-6, None))
    post_ptp = np.log10(np.clip(channel_stats_df["post_ptp_uV"].dropna().to_numpy(), 1e-6, None))
    axes[1].hist(pre_ptp, bins=50, alpha=0.55, label="Pre-filter ptp", color="#1f77b4")
    axes[1].hist(post_ptp, bins=50, alpha=0.55, label="Post-filter (0.5-45 Hz) ptp", color="#ff7f0e")
    axes[1].axvline(3.0, color="red", linestyle="--", linewidth=1.2, label="1000 uV reference")
    axes[1].set_xlabel("log10(Peak-to-Peak Amplitude [uV])")
    axes[1].set_ylabel("Channel Trace Count")
    axes[1].set_title("Candidate EEG Channel Peak-to-Peak Range")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(fig1_path, dpi=150)
    plt.close(fig)
    generated.append(str(fig1_path))

    # Figure 2: Unflagged recording percentage across threshold rules (allow 0 flagged channels)
    fig2_path = figures_dir / "threshold_unflagged_retention_comparison.png"
    sub_thr = threshold_summary_df[
        (threshold_summary_df["max_flagged_channels_allowed"] == 0)
        & (threshold_summary_df["min_std_uV"].isin([0.5, 1.0]))
    ].copy()
    if not sub_thr.empty:
        fig, ax = plt.subplots(figsize=(10, 4.8))
        labels = [
            f"{r['signal_stage']}\nstd>={r['min_std_uV']:g}, ptp<={r['max_ptp_uV']:g}"
            for _, r in sub_thr.iterrows()
        ]
        x = np.arange(len(labels))
        ax.bar(x, sub_thr["unflagged_recording_pct"], color="#2ca02c", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Unflagged Recordings (%)")
        ax.set_ylim(0, 105)
        ax.set_title("Unflagged Recording Percentage (Strict Policy: 0 Flagged Channels Allowed)")
        fig.tight_layout()
        fig.savefig(fig2_path, dpi=150)
        plt.close(fig)
        generated.append(str(fig2_path))

    # Figure 3: CAR vs None ptp inflation on unflagged channels
    fig3_path = figures_dir / "car_vs_none_contamination_comparison.png"
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    zero_spike = car_comparison_df[
        car_comparison_df["n_channels_post_ptp_gt_1000uV"] == 0
    ]["unflagged_channels_ptp_ratio_car_over_none"].dropna()
    has_spike = car_comparison_df[
        car_comparison_df["n_channels_post_ptp_gt_1000uV"] > 0
    ]["unflagged_channels_ptp_ratio_car_over_none"].dropna()
    data_to_plot = [
        zero_spike.to_numpy() if len(zero_spike) > 0 else np.array([1.0]),
        has_spike.to_numpy() if len(has_spike) > 0 else np.array([1.0]),
    ]
    ax.boxplot(
        data_to_plot,
        tick_labels=[
            f"0 channels >1000 uV\n(n={len(zero_spike)})",
            f">=1 channel >1000 uV\n(n={len(has_spike)})",
        ],
    )
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0)
    ax.set_ylabel("Median Unflagged-Channel PTP Ratio (CAR / None)")
    ax.set_title("Cross-Channel Amplitude Ratio Under CAR vs. No Re-Referencing")
    fig.tight_layout()
    fig.savefig(fig3_path, dpi=150)
    plt.close(fig)
    generated.append(str(fig3_path))

    # Figure 4: Recording duration distribution across tasks
    fig4_path = figures_dir / "recording_duration_distribution.png"
    if "duration_sec" in manifest_df.columns:
        fig, ax = plt.subplots(figsize=(8, 4.2))
        for task_name in ("Char", "Digit", "Image"):
            durs = manifest_df[manifest_df["task_category"] == task_name]["duration_sec"].dropna()
            if not durs.empty:
                ax.hist(durs, bins=15, alpha=0.5, label=f"{task_name} (n={len(durs)})")
        ax.set_xlabel("Recording Duration (seconds)")
        ax.set_ylabel("Recording Count")
        ax.set_title("Full-Recording Duration Distribution by Task Category")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig4_path, dpi=150)
        plt.close(fig)
        generated.append(str(fig4_path))

    return generated


def _build_markdown_report(
    summary: dict[str, Any],
    threshold_summary_df: pd.DataFrame,
    windowing_df: pd.DataFrame,
) -> str:
    """Format a human-readable Markdown report for Phase 3A."""
    f_sum = summary["filter_comparison_summary"]
    m_sum = summary["prefix_and_edf_metadata_summary"]
    c_sum = summary["car_vs_none_summary"]

    lines = [
        "# Phase 3A: Read-Only Preprocessing Diagnostic Report",
        "",
        "## 1. Scope & Scientific Safeguards",
        "- **Read-Only Execution**: Raw `.edf` files, Phase 1 audit reports, and Phase 2 signal-quality reports were not modified.",
        "- **Candidate Filter Under Evaluation**: Per-channel mean subtraction followed by a 4th-order Butterworth `0.5-45.0 Hz` zero-phase band-pass filter at `128 Hz` (`sosfiltfilt`).",
        "- **Exploratory Screening Terminology**: Channels and recordings that satisfy a candidate threshold rule are reported strictly as **unflagged under that rule**, not as verified artifact-free EEG.",
        "- **Unverified Prefixes**: Filename prefixes (`unverified_candidate_prefix`) are not assumed to identify human participants.",
        "- **Full-Recording Provenance**: No imagined-speech onset/offset sub-windows were inferred or cropped.",
        "",
        "## 2. Pre-Filter vs. Post-Filter (0.5-45 Hz) Distribution Comparison",
        f"- **Recordings Inspected**: `{summary['n_recordings_inspected']}` (`{summary['n_channel_traces_inspected']}` candidate EEG channel traces)",
        f"- **Pre-Filter Non-Finite Traces**: `{f_sum['pre_non_finite_count']}` | **Post-Filter Non-Finite Traces**: `{f_sum['post_non_finite_count']}`",
        f"- **Pre-Filter Mean (uV)**: median `{f_sum['pre_mean_uV']['median']:.2f}`, min `{f_sum['pre_mean_uV']['min']:.2f}`, max `{f_sum['pre_mean_uV']['max']:.2f}`",
        f"- **Post-Filter Mean (uV)**: median `{f_sum['post_mean_uV']['median']:.4e}`",
        f"- **Pre-Filter Std (uV)**: min `{f_sum['pre_std_uV']['min']:.4f}`, p5 `{f_sum['pre_std_uV']['p5']:.2f}`, median `{f_sum['pre_std_uV']['median']:.2f}`, p95 `{f_sum['pre_std_uV']['p95']:.2f}`, max `{f_sum['pre_std_uV']['max']:.2f}`",
        f"- **Post-Filter Std (uV)**: min `{f_sum['post_std_uV']['min']:.4f}`, p5 `{f_sum['post_std_uV']['p5']:.2f}`, median `{f_sum['post_std_uV']['median']:.2f}`, p95 `{f_sum['post_std_uV']['p95']:.2f}`, max `{f_sum['post_std_uV']['max']:.2f}`",
        f"- **Pre-Filter PTP (uV)**: min `{f_sum['pre_ptp_uV']['min']:.2f}`, median `{f_sum['pre_ptp_uV']['median']:.2f}`, p95 `{f_sum['pre_ptp_uV']['p95']:.2f}`, max `{f_sum['pre_ptp_uV']['max']:.2f}`",
        f"- **Post-Filter PTP (uV)**: min `{f_sum['post_ptp_uV']['min']:.2f}`, median `{f_sum['post_ptp_uV']['median']:.2f}`, p95 `{f_sum['post_ptp_uV']['p95']:.2f}`, max `{f_sum['post_ptp_uV']['max']:.2f}`",
        f"- **Channel Traces with `std < 1.0 uV`**: Pre-filter `{f_sum['n_pre_std_lt_1uV']}` -> Post-filter `{f_sum['n_post_std_lt_1uV']}` (`std < 0.5 uV` post-filter: `{f_sum['n_post_std_lt_0_5uV']}`, `std < 0.001 uV` post-filter: `{f_sum['n_post_std_lt_0_001uV']}`)",
        f"- **Channel Traces with `ptp > 1000.0 uV`**: Pre-filter `{f_sum['n_pre_ptp_gt_1000uV']}` -> Post-filter `{f_sum['n_post_ptp_gt_1000uV']}` (`ptp > 500 uV` post-filter: `{f_sum['n_post_ptp_gt_500uV']}`, `ptp > 1500 uV` post-filter: `{f_sum['n_post_ptp_gt_1500uV']}`)",
        "",
        "## 3. Exploratory Threshold Grid Highlights",
        "Selected rows from `threshold_retention_summary.csv` (`max_flagged_channels_allowed` in `{0, 1, 2}`):",
        "",
        "| Signal Stage | Min Std (uV) | Max PTP (uV) | Max Flagged Ch | Flagged Ch Rows | Unflagged Recordings | Unflagged (%) | Char (%) | Digit (%) | Image (%) | Min Unflagged / Label |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    highlight_df = threshold_summary_df[
        threshold_summary_df["min_std_uV"].isin([0.001, 0.5, 1.0])
        & threshold_summary_df["max_ptp_uV"].isin([500.0, 1000.0, 1500.0])
        & threshold_summary_df["max_flagged_channels_allowed"].isin([0, 1, 2])
    ]
    for _, r in highlight_df.iterrows():
        lines.append(
            f"| `{r['signal_stage']}` | {r['min_std_uV']:g} | {r['max_ptp_uV']:g} | "
            f"{int(r['max_flagged_channels_allowed'])} | {int(r['n_flagged_channel_rows'])} | "
            f"{int(r['n_unflagged_recordings'])}/{int(r['n_total_recordings'])} | "
            f"{r['unflagged_recording_pct']:.1f}% | {r['unflagged_char_pct']:.1f}% | "
            f"{r['unflagged_digit_pct']:.1f}% | {r['unflagged_image_pct']:.1f}% | "
            f"{int(r['min_unflagged_per_label'])} |"
        )

    lines.extend(
        [
            "",
            "## 4. Filename-Prefix & EDF Metadata Findings",
            f"- **Distinct `unverified_candidate_prefix` values**: `{m_sum['n_distinct_unverified_prefixes']}`",
            f"- **Recordings per prefix (min / median / max)**: `{m_sum['recordings_per_prefix_min']}` / `{m_sum['recordings_per_prefix_median']:.1f}` / `{m_sum['recordings_per_prefix_max']}`",
            f"- **Distinct `edf_patient_id_raw` values**: `{m_sum['n_distinct_edf_patient_id_raw']}` (`{m_sum['distinct_edf_patient_id_raw_values']}`)",
            f"- **Distinct `edf_recording_id_raw` values**: `{m_sum['n_distinct_edf_recording_id_raw']}`",
            f"- **Distinct `edf_start_date` values**: `{m_sum['n_distinct_edf_start_dates']}`",
            f"- **Safeguard**: {m_sum['identity_verification_note']}",
            "",
            "## 5. No Re-Referencing (`none`) vs. Common Average Reference (`CAR`)",
            f"- **Recordings with 0 channels > 1000 uV post-filter**: `{c_sum['n_recordings_zero_high_ptp']}` (median CAR/None PTP ratio on unflagged channels: `{c_sum['median_car_over_none_ptp_ratio_when_zero_high_ptp']}`)",
            f"- **Recordings with >= 1 channel > 1000 uV post-filter**: `{c_sum['n_recordings_with_high_ptp']}` (median CAR/None PTP ratio on the remaining <= 1000 uV channels: `{c_sum['median_car_over_none_ptp_ratio_when_high_ptp_present']}`, max ratio: `{c_sum['max_car_over_none_ptp_ratio_when_high_ptp_present']}`)",
            f"- **Median Mean-Abs Inter-Channel Correlation**: `none` = `{c_sum['median_interchannel_corr_none']}`, `CAR` = `{c_sum['median_interchannel_corr_car']}`",
            "",
            "## 6. Full-Recording Duration Provenance & Windowing Implications",
            "Window-count simulation across full recordings (no inferred speech onset/offset cropping):",
            "",
            "| Scheme | Scope | Total Recordings | Total Windows | Min/Med/Max Windows per Rec | Duration Weight Ratio (Max/Min) | Recs with Trailing Samples |",
            "| :--- | :--- | ---: | ---: | :--- | ---: | ---: |",
        ]
    )

    for _, w_row in windowing_df[windowing_df["task_scope"] == "ALL"].iterrows():
        lines.append(
            f"| `{w_row['scheme_id']}` | `{w_row['task_scope']}` | {int(w_row['n_recordings'])} | "
            f"{int(w_row['total_windows'])} | "
            f"{int(w_row['min_windows_per_recording'])} / {w_row['median_windows_per_recording']:.1f} / {int(w_row['max_windows_per_recording'])} | "
            f"{w_row['max_to_min_recording_window_weight_ratio']:.2f}x | "
            f"{int(w_row['n_recordings_with_trailing_unwindowed_samples'])} |"
        )

    lines.append("")
    return "\n".join(lines)


def _quantiles_dict(values: np.ndarray) -> dict[str, float]:
    """Return min, p5, p25, median, p75, p95, max for finite values."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {k: math.nan for k in ("min", "p5", "p25", "median", "p75", "p95", "max")}
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


def run_phase3a_diagnostic(
    config: ProjectConfig,
    manifest_source: pd.DataFrame | Path | str,
    output_dir: Path | str | None = None,
    *,
    selected_relative_paths: Sequence[str] | None = None,
    min_std_grid_uV: Sequence[float] = DEFAULT_CANDIDATE_MIN_STD_UV,
    max_ptp_grid_uV: Sequence[float] = DEFAULT_CANDIDATE_MAX_PTP_UV,
    max_flagged_channels_policies: Sequence[int] = DEFAULT_MAX_FLAGGED_CHANNELS_POLICIES,
    low_hz: float = DEFAULT_CANDIDATE_BANDPASS_LOW_HZ,
    high_hz: float = DEFAULT_CANDIDATE_BANDPASS_HIGH_HZ,
    butter_order: int = DEFAULT_CANDIDATE_BUTTER_ORDER,
    save_outputs: bool = True,
) -> Phase3ADiagnosticResult:
    """Execute the read-only Phase 3A preprocessing diagnostic across manifest recordings."""
    if isinstance(manifest_source, (Path, str)):
        manifest_df = load_and_validate_manifest(Path(manifest_source))
    else:
        manifest_df = manifest_source.copy()

    manifest_df = manifest_df.sort_values("relative_source_path").reset_index(drop=True)
    if selected_relative_paths is not None:
        sel_set = {str(p) for p in selected_relative_paths}
        manifest_df = (
            manifest_df[manifest_df["relative_source_path"].astype(str).isin(sel_set)]
            .sort_values("relative_source_path")
            .reset_index(drop=True)
        )
        if manifest_df.empty:
            raise ValueError("No rows in manifest_df matched selected_relative_paths.")

    expected_eeg = tuple(config.signal_quality.expected_eeg_channels)
    all_channel_rows: list[dict[str, Any]] = []
    all_car_rows: list[dict[str, Any]] = []

    for _, m_row in manifest_df.iterrows():
        rel_path = str(m_row["relative_source_path"])
        edf_path = (Path(config.data_root) / rel_path).resolve()
        ch_rows, car_row = compute_recording_diagnostic_metrics(
            edf_path=edf_path,
            manifest_row=m_row,
            expected_eeg_channels=expected_eeg,
            low_hz=low_hz,
            high_hz=high_hz,
            butter_order=butter_order,
            rail_rel_tol=config.signal_quality.clipping_boundary_rel_tol,
        )
        all_channel_rows.extend(ch_rows)
        all_car_rows.append(car_row)

    channel_stats_df = pd.DataFrame(all_channel_rows)
    car_comparison_df = pd.DataFrame(all_car_rows)

    threshold_summary_df, threshold_label_balance_df = evaluate_threshold_grid(
        channel_stats_df=channel_stats_df,
        min_std_grid_uV=min_std_grid_uV,
        max_ptp_grid_uV=max_ptp_grid_uV,
        max_flagged_channels_policies=max_flagged_channels_policies,
    )
    prefix_metadata_df, prefix_meta_summary = audit_prefixes_and_edf_metadata(
        manifest_df=manifest_df,
        channel_stats_df=channel_stats_df,
    )
    windowing_simulation_df = simulate_windowing_provenance(manifest_df=manifest_df)

    # Compute fate of pre-filter low-std (<1 uV) and high-ptp (>1000 uV) traces after filtering
    pre_low_1 = channel_stats_df["pre_std_uV"] < 1.0
    pre_high_1000 = channel_stats_df["pre_ptp_uV"] > 1000.0
    post_low_1 = channel_stats_df["post_std_uV"] < 1.0
    post_low_05 = channel_stats_df["post_std_uV"] < 0.5
    post_low_025 = channel_stats_df["post_std_uV"] < 0.25
    post_low_0001 = channel_stats_df["post_std_uV"] < 0.001
    post_high_500 = channel_stats_df["post_ptp_uV"] > 500.0
    post_high_1000 = channel_stats_df["post_ptp_uV"] > 1000.0
    post_high_1500 = channel_stats_df["post_ptp_uV"] > 1500.0
    post_high_2000 = channel_stats_df["post_ptp_uV"] > 2000.0

    target_pad_samples = int(round(3.0 * 128.0 / low_hz))
    filter_summary = {
        "candidate_filter_description": (
            f"Per-channel mean subtraction + {butter_order}th-order Butterworth "
            f"{low_hz:g}-{high_hz:g} Hz zero-phase band-pass filter at 128 Hz "
            f"(sosfiltfilt, padtype='even', padlen={target_pad_samples} samples = "
            f"{target_pad_samples / 128.0:.1f} s for 11-22 s recordings)"
        ),
        "filter_padtype": "even",
        "filter_padlen_formula": "min(n_samples - 1, max(default_padlen, round(3.0 * sfreq_hz / low_hz)))",
        "filter_target_padlen_samples_at_128hz": target_pad_samples,
        "filter_target_padlen_duration_sec_at_128hz": round(target_pad_samples / 128.0, 4),
        "pre_non_finite_count": int(channel_stats_df["pre_has_nan_or_inf"].sum()),
        "post_non_finite_count": int(channel_stats_df["post_has_nan_or_inf"].sum()),
        "pre_mean_uV": _quantiles_dict(channel_stats_df["pre_mean_uV"].to_numpy(dtype=float)),
        "post_mean_uV": _quantiles_dict(channel_stats_df["post_mean_uV"].to_numpy(dtype=float)),
        "pre_std_uV": _quantiles_dict(channel_stats_df["pre_std_uV"].to_numpy(dtype=float)),
        "post_std_uV": _quantiles_dict(channel_stats_df["post_std_uV"].to_numpy(dtype=float)),
        "pre_ptp_uV": _quantiles_dict(channel_stats_df["pre_ptp_uV"].to_numpy(dtype=float)),
        "post_ptp_uV": _quantiles_dict(channel_stats_df["post_ptp_uV"].to_numpy(dtype=float)),
        "std_retention_ratio": _quantiles_dict(
            channel_stats_df["std_retention_ratio"].to_numpy(dtype=float)
        ),
        "ptp_reduction_ratio": _quantiles_dict(
            channel_stats_df["ptp_reduction_ratio"].to_numpy(dtype=float)
        ),
        "n_pre_std_lt_1uV": int(pre_low_1.sum()),
        "n_recordings_with_pre_std_lt_1uV": int(
            channel_stats_df.loc[pre_low_1, "relative_source_path"].nunique()
        ),
        "n_post_std_lt_1uV": int(post_low_1.sum()),
        "n_recordings_with_post_std_lt_1uV": int(
            channel_stats_df.loc[post_low_1, "relative_source_path"].nunique()
        ),
        "n_post_std_lt_0_5uV": int(post_low_05.sum()),
        "n_post_std_lt_0_25uV": int(post_low_025.sum()),
        "n_post_std_lt_0_001uV": int(post_low_0001.sum()),
        "n_pre_ptp_gt_1000uV": int(pre_high_1000.sum()),
        "n_recordings_with_pre_ptp_gt_1000uV": int(
            channel_stats_df.loc[pre_high_1000, "relative_source_path"].nunique()
        ),
        "n_post_ptp_gt_500uV": int(post_high_500.sum()),
        "n_post_ptp_gt_1000uV": int(post_high_1000.sum()),
        "n_recordings_with_post_ptp_gt_1000uV": int(
            channel_stats_df.loc[post_high_1000, "relative_source_path"].nunique()
        ),
        "n_post_ptp_gt_1500uV": int(post_high_1500.sum()),
        "n_post_ptp_gt_2000uV": int(post_high_2000.sum()),
        "n_pre_ptp_gt_1000uV_resolved_to_le_1000uV_post_filter": int(
            (pre_high_1000 & ~post_high_1000).sum()
        ),
    }

    zero_spike_df = car_comparison_df[car_comparison_df["n_channels_post_ptp_gt_1000uV"] == 0]
    has_spike_df = car_comparison_df[car_comparison_df["n_channels_post_ptp_gt_1000uV"] > 0]
    car_summary = {
        "n_recordings_zero_high_ptp": int(len(zero_spike_df)),
        "n_recordings_with_high_ptp": int(len(has_spike_df)),
        "median_car_over_none_ptp_ratio_when_zero_high_ptp": (
            round(float(zero_spike_df["unflagged_channels_ptp_ratio_car_over_none"].median()), 4)
            if not zero_spike_df.empty
            else None
        ),
        "median_car_over_none_ptp_ratio_when_high_ptp_present": (
            round(float(has_spike_df["unflagged_channels_ptp_ratio_car_over_none"].median()), 4)
            if not has_spike_df.empty
            else None
        ),
        "max_car_over_none_ptp_ratio_when_high_ptp_present": (
            round(float(has_spike_df["unflagged_channels_ptp_ratio_car_over_none"].max()), 4)
            if not has_spike_df.empty
            else None
        ),
        "median_interchannel_corr_none": (
            round(float(car_comparison_df["mean_abs_interchannel_corr_none"].dropna().median()), 4)
            if not car_comparison_df["mean_abs_interchannel_corr_none"].dropna().empty
            else None
        ),
        "median_interchannel_corr_car": (
            round(float(car_comparison_df["mean_abs_interchannel_corr_car"].dropna().median()), 4)
            if not car_comparison_df["mean_abs_interchannel_corr_car"].dropna().empty
            else None
        ),
    }

    summary_dict: dict[str, Any] = {
        "phase": "Phase 3A Read-Only Preprocessing Diagnostic",
        "n_recordings_inspected": int(len(manifest_df)),
        "n_channel_traces_inspected": int(len(channel_stats_df)),
        "filter_comparison_summary": filter_summary,
        "prefix_and_edf_metadata_summary": prefix_meta_summary,
        "car_vs_none_summary": car_summary,
    }

    generated_files: list[str] = []
    generated_figures: list[str] = []

    if save_outputs:
        target_dir = Path(output_dir) if output_dir is not None else Path(config.output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        csv_specs = (
            ("pre_vs_post_filter_channel_stats.csv", channel_stats_df),
            ("threshold_retention_summary.csv", threshold_summary_df),
            ("threshold_class_balance_by_label.csv", threshold_label_balance_df),
            ("prefix_and_header_metadata_audit.csv", prefix_metadata_df),
            ("reference_car_vs_none_comparison.csv", car_comparison_df),
            ("windowing_provenance_simulation.csv", windowing_simulation_df),
        )
        for fname, df_obj in csv_specs:
            f_path = target_dir / fname
            df_obj.to_csv(f_path, index=False)
            generated_files.append(str(f_path))

        json_path = target_dir / "phase3a_diagnostic_summary.json"
        json_path.write_text(json.dumps(summary_dict, indent=2), encoding="utf-8")
        generated_files.append(str(json_path))

        md_path = target_dir / "phase3a_diagnostic_report.md"
        md_path.write_text(
            _build_markdown_report(summary_dict, threshold_summary_df, windowing_simulation_df),
            encoding="utf-8",
        )
        generated_files.append(str(md_path))

        generated_figures = _generate_phase3a_figures(
            channel_stats_df=channel_stats_df,
            threshold_summary_df=threshold_summary_df,
            car_comparison_df=car_comparison_df,
            manifest_df=manifest_df,
            figures_dir=target_dir / "figures",
        )

    return Phase3ADiagnosticResult(
        channel_stats_df=channel_stats_df,
        threshold_summary_df=threshold_summary_df,
        threshold_label_balance_df=threshold_label_balance_df,
        prefix_metadata_df=prefix_metadata_df,
        car_comparison_df=car_comparison_df,
        windowing_simulation_df=windowing_simulation_df,
        summary=summary_dict,
        generated_files=generated_files,
        generated_figures=generated_figures,
    )
