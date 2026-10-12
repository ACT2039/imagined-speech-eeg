"""Phase 2: EEG Signal Quality Analysis and Visualization (Rules 6, 7, 8, 9, 10, 12, 13).

Scientific Safeguards
---------------------
1. Raw recordings are read in-memory and never modified on disk.
2. No bandpass filtering, notch filtering, resampling, ICA, baseline correction,
   normalization, or artifact removal is applied in this module.
3. Imagined-speech onset/offset intervals are NOT inferred without verified event markers
   or protocol documentation.
4. Physical units are preserved from the EDF channel headers; calibration is never guessed
   solely from amplitude magnitudes.
5. Measured signal statistics (`measured_*`) are strictly separated from heuristic diagnostic
   flags (`heuristic_*`). Heuristic flags indicate recordings/channels warranting review and
   do NOT constitute definitive proof of physiological or hardware artifacts.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from imagined_speech.config import EDFAuditConfig, ProjectConfig, SignalQualityConfig
from imagined_speech.data.edf_audit import (
    classify_channel,
    clean_channel_name,
    detect_counter_like_signal,
    read_raw_edf_header,
    run_dataset_audit,
)
from imagined_speech.exceptions import EDFValidationError, SignalQualityError

logger = logging.getLogger("imagined_speech.data.signal_quality")

REQUIRED_MANIFEST_COLUMNS: frozenset[str] = frozenset(
    {
        "original_filename",
        "relative_source_path",
        "task_category",
        "raw_label",
        "normalized_label",
        "unverified_candidate_prefix",
        "is_valid_edf",
    }
)

_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_\-]+")


@dataclass(frozen=True)
class SelectedRecordingInput:
    """Validated input specification for a recording selected for signal-quality inspection."""

    original_filename: str
    relative_source_path: str
    absolute_path: Path
    task_category: str
    raw_label: str | None
    normalized_label: str | None
    unverified_candidate_prefix: str | None
    selection_reason: str


@dataclass(frozen=True)
class InspectionErrorRecord:
    """Error record when a selected or discovered EDF recording cannot be analyzed."""

    relative_source_path: str
    task_category: str
    error_stage: str
    error_message: str


@dataclass(frozen=True)
class ChannelSignalStatistics:
    """Per-channel signal statistics separating measured values from heuristic flags."""

    relative_source_path: str
    original_filename: str
    task_category: str
    raw_label: str | None
    normalized_label: str | None
    unverified_candidate_prefix: str | None
    channel_index: int
    raw_channel_name: str
    cleaned_channel_name: str
    mne_assigned_type: str
    audited_channel_role: str
    is_candidate_eeg: bool
    physical_unit: str
    edf_physical_min: float | None
    edf_physical_max: float | None
    edf_transducer_type: str
    edf_prefiltering: str
    # Measured values in original physical units
    n_samples: int
    sfreq_hz: float
    duration_sec: float
    n_non_finite: int
    n_finite: int
    measured_min: float | None
    measured_max: float | None
    measured_mean: float | None
    measured_std: float | None
    measured_median: float | None
    measured_iqr: float | None
    measured_amplitude_range_ptp: float | None
    measured_max_constant_run_samples: int
    measured_max_constant_run_fraction: float
    measured_near_zero_diff_fraction: float
    measured_extreme_rail_fraction: float
    measured_unique_value_count: int
    measured_unique_value_ratio: float
    measured_mode_value_fraction: float
    measured_repeated_adjacent_fraction: float
    # Heuristic diagnostic flags (not definitive proof of artifacts)
    heuristic_has_non_finite: bool
    heuristic_is_flatline: bool
    heuristic_is_near_flatline: bool
    heuristic_potential_clipping: bool
    heuristic_suspicious_repeated_values: bool
    heuristic_is_counter_like: bool
    heuristic_quality_flags: str
    calibration_or_unit_note: str


@dataclass(frozen=True)
class RecordingQualityInspection:
    """Complete inspection result for a single EDF recording."""

    recording: SelectedRecordingInput
    is_readable: bool
    error_message: str | None
    sfreq_hz: float | None
    n_samples: int | None
    duration_sec: float | None
    edf_patient_id_raw: str | None
    edf_recording_id_raw: str | None
    present_channel_names: tuple[str, ...]
    candidate_eeg_channel_names: tuple[str, ...]
    non_eeg_channel_names: tuple[str, ...]
    unknown_channel_names: tuple[str, ...]
    missing_expected_eeg_channels: tuple[str, ...]
    channel_statistics: tuple[ChannelSignalStatistics, ...]
    generated_figures: tuple[str, ...] = ()


@dataclass
class SignalQualityAnalysisResult:
    """Outputs of the Phase 2 signal-quality analysis pipeline."""

    selected_recordings: list[SelectedRecordingInput] = field(default_factory=list)
    recording_inspections: list[RecordingQualityInspection] = field(default_factory=list)
    channel_statistics: list[ChannelSignalStatistics] = field(default_factory=list)
    errors: list[InspectionErrorRecord] = field(default_factory=list)
    generated_figures: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    report_markdown: str = ""


def _sanitize_token(text: str) -> str:
    cleaned = _SAFE_FILENAME_RE.sub("_", text.strip()).strip("_")
    return cleaned or "recording"


def _compute_max_constant_run(values_1d: np.ndarray) -> int:
    """Compute the maximum number of consecutive identical values in a 1D finite array."""
    n = int(values_1d.size)
    if n == 0:
        return 0
    if n == 1:
        return 1

    change_indices = np.flatnonzero(np.diff(values_1d) != 0.0) + 1
    boundaries = np.concatenate(([0], change_indices, [n]))
    run_lengths = np.diff(boundaries)
    return int(np.max(run_lengths))


def _compute_mode_fraction(values_1d: np.ndarray) -> tuple[int, float]:
    """Return `(unique_count, max_mode_fraction)` for a 1D finite array."""
    n = int(values_1d.size)
    if n == 0:
        return 0, 0.0
    unique_vals, counts = np.unique(values_1d, return_counts=True)
    return int(unique_vals.size), float(np.max(counts) / n)


def compute_channel_signal_statistics(
    signal_1d: np.ndarray,
    *,
    sfreq_hz: float,
    relative_source_path: str,
    original_filename: str,
    task_category: str,
    raw_label: str | None = None,
    normalized_label: str | None = None,
    unverified_candidate_prefix: str | None = None,
    channel_index: int = 0,
    raw_channel_name: str = "CH0",
    mne_assigned_type: str = "eeg",
    physical_unit: str = "uV",
    edf_physical_min: float | None = None,
    edf_physical_max: float | None = None,
    edf_transducer_type: str = "",
    edf_prefiltering: str = "",
    sq_config: SignalQualityConfig | None = None,
    audit_config: EDFAuditConfig | None = None,
) -> ChannelSignalStatistics:
    """Calculate measured statistics and heuristic quality indicators for one channel.

    Measured values are computed directly on the unmodified signal in its original
    EDF physical units. Heuristic quality flags are reported separately and never
    alter the underlying signal.
    """
    if sfreq_hz <= 0:
        raise SignalQualityError(f"sfreq_hz must be positive, got {sfreq_hz!r}.")

    sq_cfg = sq_config or SignalQualityConfig()
    aud_cfg = audit_config or EDFAuditConfig(flat_std_threshold=sq_cfg.flat_std_threshold)

    arr = np.asarray(signal_1d, dtype=float).ravel()
    n_samples = int(arr.size)
    duration_sec = float(n_samples / sfreq_hz)

    finite_mask = np.isfinite(arr)
    n_finite = int(np.sum(finite_mask))
    n_non_finite = int(n_samples - n_finite)
    finite_vals = arr[finite_mask]

    (
        audited_role,
        is_cand_eeg,
        _has_usable,
        _is_flat_audit,
        _is_counter_audit,
        _has_nan_audit,
        _n_uniq_audit,
        _audit_flags,
    ) = classify_channel(
        raw_channel_name=raw_channel_name,
        mne_assigned_type=mne_assigned_type,
        edf_phys_dim=physical_unit,
        signal_1d=finite_vals if n_finite > 0 else arr,
        config=aud_cfg,
    )

    cleaned_name = clean_channel_name(raw_channel_name)
    norm_unit = physical_unit.strip() if physical_unit and physical_unit.strip() else "unspecified"

    calibration_notes: list[str] = []
    if norm_unit == "unspecified":
        calibration_notes.append("missing_edf_physical_unit")
    elif is_cand_eeg and norm_unit.lower() not in {"uv", "µv", "mv", "v", "microvolt"}:
        calibration_notes.append(f"non_standard_eeg_unit({norm_unit})")

    if n_finite == 0:
        flags = ["all_samples_non_finite"]
        if audited_role == "unknown_needs_review":
            flags.append("unrecognized_channel_name")
        return ChannelSignalStatistics(
            relative_source_path=relative_source_path,
            original_filename=original_filename,
            task_category=task_category,
            raw_label=raw_label,
            normalized_label=normalized_label,
            unverified_candidate_prefix=unverified_candidate_prefix,
            channel_index=channel_index,
            raw_channel_name=raw_channel_name,
            cleaned_channel_name=cleaned_name,
            mne_assigned_type=mne_assigned_type,
            audited_channel_role=audited_role,
            is_candidate_eeg=is_cand_eeg,
            physical_unit=norm_unit,
            edf_physical_min=edf_physical_min,
            edf_physical_max=edf_physical_max,
            edf_transducer_type=edf_transducer_type,
            edf_prefiltering=edf_prefiltering,
            n_samples=n_samples,
            sfreq_hz=float(sfreq_hz),
            duration_sec=duration_sec,
            n_non_finite=n_non_finite,
            n_finite=0,
            measured_min=None,
            measured_max=None,
            measured_mean=None,
            measured_std=None,
            measured_median=None,
            measured_iqr=None,
            measured_amplitude_range_ptp=None,
            measured_max_constant_run_samples=0,
            measured_max_constant_run_fraction=0.0,
            measured_near_zero_diff_fraction=0.0,
            measured_extreme_rail_fraction=0.0,
            measured_unique_value_count=0,
            measured_unique_value_ratio=0.0,
            measured_mode_value_fraction=0.0,
            measured_repeated_adjacent_fraction=0.0,
            heuristic_has_non_finite=True,
            heuristic_is_flatline=False,
            heuristic_is_near_flatline=False,
            heuristic_potential_clipping=False,
            heuristic_suspicious_repeated_values=False,
            heuristic_is_counter_like=False,
            heuristic_quality_flags=";".join(flags),
            calibration_or_unit_note=";".join(calibration_notes),
        )

    measured_min = float(np.min(finite_vals))
    measured_max = float(np.max(finite_vals))
    measured_mean = float(np.mean(finite_vals))
    measured_std = float(np.std(finite_vals))
    q25, q50, q75 = np.percentile(finite_vals, [25.0, 50.0, 75.0])
    measured_median = float(q50)
    measured_iqr = float(q75 - q25)
    measured_ptp = float(measured_max - measured_min)

    if is_cand_eeg and abs(measured_mean) > 1000.0 and measured_std > 0:
        calibration_notes.append(
            f"large_dc_offset_present(mean={measured_mean:.1f}_{norm_unit})"
        )

    max_const_run = _compute_max_constant_run(finite_vals)
    max_const_run_frac = float(max_const_run / n_finite)

    if n_finite >= 2:
        diffs = np.abs(np.diff(finite_vals))
        near_zero_diff_frac = float(np.mean(diffs <= sq_cfg.near_flat_diff_tol))
        repeated_adj_frac = float(np.mean(diffs == 0.0))
    else:
        near_zero_diff_frac = 1.0
        repeated_adj_frac = 0.0

    unique_count, mode_frac = _compute_mode_fraction(finite_vals)
    unique_ratio = float(unique_count / n_finite)

    # Measure rail proximity against EDF physical min/max if available, else empirical min/max
    rail_frac = 0.0
    if measured_ptp > sq_cfg.flat_std_threshold:
        if (
            edf_physical_min is not None
            and edf_physical_max is not None
            and edf_physical_max > edf_physical_min
        ):
            hdr_span = edf_physical_max - edf_physical_min
            tol = sq_cfg.clipping_boundary_rel_tol * hdr_span
            near_hdr_rails = (finite_vals <= edf_physical_min + tol) | (
                finite_vals >= edf_physical_max - tol
            )
            hdr_rail_frac = float(np.mean(near_hdr_rails))
        else:
            hdr_rail_frac = 0.0

        emp_tol = sq_cfg.clipping_boundary_rel_tol * measured_ptp
        near_emp_rails = (finite_vals <= measured_min + emp_tol) | (
            finite_vals >= measured_max - emp_tol
        )
        emp_rail_frac = float(np.mean(near_emp_rails))
        # Empirical extremes only suggest clipping if many samples pile up at the exact ceiling/floor
        rail_frac = max(hdr_rail_frac, emp_rail_frac)

    # Heuristic indicators (explicitly separated from measured metrics)
    heur_has_non_finite = bool(n_non_finite > 0)
    heur_is_flat = bool(
        measured_std <= sq_cfg.flat_std_threshold
        or measured_ptp <= sq_cfg.flat_std_threshold
    )
    heur_is_near_flat = bool(
        not heur_is_flat
        and (
            measured_std <= sq_cfg.near_flat_std_threshold
            or max_const_run_frac >= sq_cfg.constant_run_fraction_threshold
            or near_zero_diff_frac >= sq_cfg.near_zero_diff_fraction_threshold
        )
    )
    heur_is_counter = bool(
        not heur_is_flat and detect_counter_like_signal(finite_vals)
    )

    # Potential clipping heuristic: non-flat, non-counter signal where a notable fraction of
    # samples saturate at the exact minimum/maximum rails
    heur_clipping = bool(
        not heur_is_flat
        and not heur_is_counter
        and n_finite >= 32
        and rail_frac >= sq_cfg.clipping_fraction_threshold
    )

    heur_repeated = bool(
        not heur_is_flat
        and not heur_is_counter
        and n_finite >= 32
        and (
            mode_frac >= sq_cfg.repeated_value_fraction_threshold
            or repeated_adj_frac >= sq_cfg.repeated_adjacent_fraction_threshold
            or unique_ratio <= sq_cfg.low_unique_ratio_threshold
        )
    )

    heur_flags: list[str] = []
    if heur_has_non_finite:
        heur_flags.append("non_finite_samples_present")
    if heur_is_flat:
        heur_flags.append("flatline")
    if heur_is_near_flat:
        heur_flags.append("near_flatline")
    if heur_clipping:
        heur_flags.append("potential_clipping_or_rail_saturation")
    if heur_repeated:
        heur_flags.append("suspicious_repeated_or_quantized_values")
    if heur_is_counter:
        heur_flags.append("counter_or_ramp_pattern")
    if audited_role == "unknown_needs_review":
        heur_flags.append("unrecognized_channel_name")

    return ChannelSignalStatistics(
        relative_source_path=relative_source_path,
        original_filename=original_filename,
        task_category=task_category,
        raw_label=raw_label,
        normalized_label=normalized_label,
        unverified_candidate_prefix=unverified_candidate_prefix,
        channel_index=channel_index,
        raw_channel_name=raw_channel_name,
        cleaned_channel_name=cleaned_name,
        mne_assigned_type=mne_assigned_type,
        audited_channel_role=audited_role,
        is_candidate_eeg=is_cand_eeg,
        physical_unit=norm_unit,
        edf_physical_min=edf_physical_min,
        edf_physical_max=edf_physical_max,
        edf_transducer_type=edf_transducer_type,
        edf_prefiltering=edf_prefiltering,
        n_samples=n_samples,
        sfreq_hz=float(sfreq_hz),
        duration_sec=duration_sec,
        n_non_finite=n_non_finite,
        n_finite=n_finite,
        measured_min=measured_min,
        measured_max=measured_max,
        measured_mean=measured_mean,
        measured_std=measured_std,
        measured_median=measured_median,
        measured_iqr=measured_iqr,
        measured_amplitude_range_ptp=measured_ptp,
        measured_max_constant_run_samples=max_const_run,
        measured_max_constant_run_fraction=max_const_run_frac,
        measured_near_zero_diff_fraction=near_zero_diff_frac,
        measured_extreme_rail_fraction=rail_frac,
        measured_unique_value_count=unique_count,
        measured_unique_value_ratio=unique_ratio,
        measured_mode_value_fraction=mode_frac,
        measured_repeated_adjacent_fraction=repeated_adj_frac,
        heuristic_has_non_finite=heur_has_non_finite,
        heuristic_is_flatline=heur_is_flat,
        heuristic_is_near_flatline=heur_is_near_flat,
        heuristic_potential_clipping=heur_clipping,
        heuristic_suspicious_repeated_values=heur_repeated,
        heuristic_is_counter_like=heur_is_counter,
        heuristic_quality_flags=";".join(heur_flags),
        calibration_or_unit_note=";".join(calibration_notes),
    )


def load_and_validate_manifest(manifest_source: Path | str | pd.DataFrame) -> pd.DataFrame:
    """Load and validate the Phase 1 dataset audit manifest.

    Parameters
    ----------
    manifest_source : Path | str | pd.DataFrame
        Path to ``manifest.csv`` or an in-memory DataFrame.

    Returns
    -------
    pd.DataFrame
        Validated manifest DataFrame.

    Raises
    ------
    SignalQualityError
        If the file does not exist, is empty, or is missing required columns.
    """
    if isinstance(manifest_source, pd.DataFrame):
        df = manifest_source.copy()
    else:
        path = Path(manifest_source)
        if not path.is_file():
            raise SignalQualityError(f"Audit manifest file not found: {path}")
        try:
            df = pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001
            raise SignalQualityError(
                f"Failed to read audit manifest CSV {path}: {exc}"
            ) from exc

    missing_cols = sorted(REQUIRED_MANIFEST_COLUMNS - set(df.columns))
    if missing_cols:
        raise SignalQualityError(
            f"Audit manifest is missing required columns: {missing_cols}"
        )
    if df.empty:
        raise SignalQualityError("Audit manifest contains zero recording rows.")

    return df


def _clean_optional_str(val: Any) -> str | None:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip()
    return s if s else None


def select_representative_recordings(
    manifest_df: pd.DataFrame,
    data_root: Path | str,
    *,
    expected_tasks: tuple[str, ...] = ("Char", "Digit", "Image"),
    recordings_per_task: int = 3,
    specific_recordings: tuple[str, ...] = (),
    seed: int = 42,
) -> list[SelectedRecordingInput]:
    """Deterministically select representative recordings across tasks, labels, and durations.

    Selection strategy:
    1. Any explicitly requested ``specific_recordings`` (matched by ``relative_source_path``
       or ``original_filename``) are included first.
    2. For each task in ``expected_tasks``, valid recordings are ordered deterministically by
       ``(duration_sec, normalized_label, relative_source_path)`` with seeded tie-breaking.
       Up to ``recordings_per_task`` recordings are chosen across evenly spaced index positions
       while preferring distinct ``normalized_label`` values and different durations.
    """
    if recordings_per_task <= 0:
        raise SignalQualityError("recordings_per_task must be a positive integer.")

    root = Path(data_root)
    if not root.exists() or not root.is_dir():
        raise SignalQualityError(f"Dataset root directory does not exist: {root}")

    df = load_and_validate_manifest(manifest_df)
    selected: list[SelectedRecordingInput] = []
    seen_rel_paths: set[str] = set()

    # 1. Include any explicitly requested files first
    for target in specific_recordings:
        norm_target = target.strip().replace("\\", "/")
        matches = df[
            (df["relative_source_path"].astype(str) == norm_target)
            | (df["original_filename"].astype(str) == norm_target)
        ]
        if matches.empty:
            raise SignalQualityError(
                f"Requested specific recording {target!r} was not found in the manifest."
            )
        for _, row in matches.iterrows():
            rel_path = str(row["relative_source_path"])
            if rel_path not in seen_rel_paths:
                seen_rel_paths.add(rel_path)
                selected.append(
                    SelectedRecordingInput(
                        original_filename=str(row["original_filename"]),
                        relative_source_path=rel_path,
                        absolute_path=root / rel_path,
                        task_category=str(row["task_category"]),
                        raw_label=_clean_optional_str(row.get("raw_label")),
                        normalized_label=_clean_optional_str(row.get("normalized_label")),
                        unverified_candidate_prefix=_clean_optional_str(
                            row.get("unverified_candidate_prefix")
                        ),
                        selection_reason="explicit_user_request",
                    )
                )

    # 2. Representative deterministic selection per task category
    rng = np.random.default_rng(seed)

    for task in expected_tasks:
        task_rows = df[
            (df["task_category"].astype(str).str.lower() == task.lower())
            & (df["is_valid_edf"].astype(bool))
        ].copy()

        if task_rows.empty:
            logger.warning("No valid EDF recordings found in manifest for task %r.", task)
            continue

        # Sort deterministically by relative_source_path first, then apply seeded permutation
        # before sorting by duration & label so ties are broken reproducibly by `seed`
        task_rows = task_rows.sort_values("relative_source_path", kind="mergesort").reset_index(
            drop=True
        )
        perm = rng.permutation(len(task_rows))
        task_rows = task_rows.iloc[perm].reset_index(drop=True)

        if "duration_sec" in task_rows.columns:
            task_rows["_sort_dur"] = pd.to_numeric(
                task_rows["duration_sec"], errors="coerce"
            ).fillna(0.0)
        else:
            task_rows["_sort_dur"] = 0.0

        task_rows["_sort_lbl"] = task_rows["normalized_label"].fillna("").astype(str)
        task_rows = task_rows.sort_values(
            by=["_sort_dur", "_sort_lbl", "relative_source_path"],
            kind="mergesort",
        ).reset_index(drop=True)

        n_avail = len(task_rows)
        n_target = min(recordings_per_task, n_avail)

        # Pick candidates spanning duration quantiles while maximizing distinct labels
        anchor_indices = np.linspace(0, n_avail - 1, num=n_target, dtype=int).tolist()
        chosen_indices: list[int] = []
        used_labels: set[str] = set()

        for anchor in anchor_indices:
            # Search outward from the quantile anchor for an unused label first
            best_idx: int | None = None
            for offset in range(n_avail):
                for candidate_idx in (anchor + offset, anchor - offset):
                    if 0 <= candidate_idx < n_avail and candidate_idx not in chosen_indices:
                        cand_rel = str(task_rows.loc[candidate_idx, "relative_source_path"])
                        if cand_rel in seen_rel_paths:
                            continue
                        cand_lbl = str(task_rows.loc[candidate_idx, "_sort_lbl"])
                        if cand_lbl not in used_labels:
                            best_idx = candidate_idx
                            break
                if best_idx is not None:
                    break

            # Fallback if all available labels have already been represented
            if best_idx is None:
                for offset in range(n_avail):
                    for candidate_idx in (anchor + offset, anchor - offset):
                        if 0 <= candidate_idx < n_avail and candidate_idx not in chosen_indices:
                            cand_rel = str(task_rows.loc[candidate_idx, "relative_source_path"])
                            if cand_rel not in seen_rel_paths:
                                best_idx = candidate_idx
                                break
                    if best_idx is not None:
                        break

            if best_idx is not None:
                chosen_indices.append(best_idx)
                used_labels.add(str(task_rows.loc[best_idx, "_sort_lbl"]))

        for rank_idx, row_idx in enumerate(chosen_indices):
            row = task_rows.loc[row_idx]
            rel_path = str(row["relative_source_path"])
            seen_rel_paths.add(rel_path)
            selected.append(
                SelectedRecordingInput(
                    original_filename=str(row["original_filename"]),
                    relative_source_path=rel_path,
                    absolute_path=root / rel_path,
                    task_category=str(row["task_category"]),
                    raw_label=_clean_optional_str(row.get("raw_label")),
                    normalized_label=_clean_optional_str(row.get("normalized_label")),
                    unverified_candidate_prefix=_clean_optional_str(
                        row.get("unverified_candidate_prefix")
                    ),
                    selection_reason=f"representative_task_sample(rank={rank_idx + 1}/{n_target})",
                )
            )

    return selected


_UNIT_TO_SI_SCALE: dict[str, float] = {
    "uv": 1.0e-6,
    "µv": 1.0e-6,
    "μv": 1.0e-6,
    "microvolt": 1.0e-6,
    "microvolts": 1.0e-6,
    "mv": 1.0e-3,
    "millivolt": 1.0e-3,
    "millivolts": 1.0e-3,
    "nv": 1.0e-9,
    "v": 1.0,
    "volt": 1.0,
    "volts": 1.0,
}


def _extract_native_unit_signals(
    raw_mne: mne.io.BaseRaw,
    raw_header_channels: dict[int, Any],
) -> tuple[np.ndarray, list[str]]:
    """Extract channel signals in their original EDF physical units.

    MNE's ``read_raw_edf`` scales channels with recognized voltage units (e.g. ``uV``,
    ``mV``) into SI Volts (`V`) using per-channel unit multipliers stored in
    ``raw_mne._raw_extras``. To preserve the original physical units stored in the EDF
    file (Section 4), this function divides out MNE's SI unit scaling factor and returns
    the native unit string for each channel.
    """
    data_si = raw_mne.get_data()
    n_channels = data_si.shape[0]
    native_signals = np.empty_like(data_si, dtype=float)
    orig_units_map: dict[str, str] = getattr(raw_mne, "_orig_units", {}) or {}

    raw_extras = getattr(raw_mne, "_raw_extras", None)
    extra_units: list[float] | None = None
    if isinstance(raw_extras, list) and raw_extras and isinstance(raw_extras[0], dict):
        u_list = raw_extras[0].get("units")
        if isinstance(u_list, (list, tuple, np.ndarray)):
            extra_units = [float(x) for x in u_list]

    unit_labels: list[str] = []
    for idx in range(n_channels):
        ch_name = raw_mne.ch_names[idx]
        hdr_ch = raw_header_channels.get(idx)
        hdr_unit = (
            hdr_ch.physical_dimension.strip()
            if hdr_ch and hdr_ch.physical_dimension
            else ""
        )
        orig_unit = hdr_unit or orig_units_map.get(ch_name, "").strip()

        si_scale = 1.0
        if extra_units is not None and idx < len(extra_units) and extra_units[idx] > 0:
            si_scale = float(extra_units[idx])
        elif orig_unit.lower() in _UNIT_TO_SI_SCALE:
            si_scale = _UNIT_TO_SI_SCALE[orig_unit.lower()]

        if si_scale != 1.0 and si_scale > 0.0:
            native_signals[idx] = data_si[idx] / si_scale
        else:
            native_signals[idx] = data_si[idx]

        unit_labels.append(orig_unit if orig_unit else "unspecified")

    return native_signals, unit_labels


def inspect_recording_signal_quality(
    recording: SelectedRecordingInput,
    sq_config: SignalQualityConfig | None = None,
    audit_config: EDFAuditConfig | None = None,
    figures_dir: Path | None = None,
) -> tuple[RecordingQualityInspection, np.ndarray | None, list[str]]:
    """Inspect a single EDF recording, compute channel statistics, and optionally plot figures.

    Returns
    -------
    tuple[RecordingQualityInspection, np.ndarray | None, list[str]]
        The inspection record, the native-unit signal matrix (for cross-task plotting),
        and the list of native unit strings per channel.
    """
    sq_cfg = sq_config or SignalQualityConfig()
    aud_cfg = audit_config or EDFAuditConfig(flat_std_threshold=sq_cfg.flat_std_threshold)

    path = recording.absolute_path
    if not path.exists() or not path.is_file():
        msg = f"EDF file does not exist on disk: {path}"
        return (
            RecordingQualityInspection(
                recording=recording,
                is_readable=False,
                error_message=msg,
                sfreq_hz=None,
                n_samples=None,
                duration_sec=None,
                edf_patient_id_raw=None,
                edf_recording_id_raw=None,
                present_channel_names=(),
                candidate_eeg_channel_names=(),
                non_eeg_channel_names=(),
                unknown_channel_names=(),
                missing_expected_eeg_channels=tuple(sq_cfg.expected_eeg_channels),
                channel_statistics=(),
                generated_figures=(),
            ),
            None,
            [],
        )

    try:
        raw_header = read_raw_edf_header(path)
    except EDFValidationError as exc:
        return (
            RecordingQualityInspection(
                recording=recording,
                is_readable=False,
                error_message=str(exc),
                sfreq_hz=None,
                n_samples=None,
                duration_sec=None,
                edf_patient_id_raw=None,
                edf_recording_id_raw=None,
                present_channel_names=(),
                candidate_eeg_channel_names=(),
                non_eeg_channel_names=(),
                unknown_channel_names=(),
                missing_expected_eeg_channels=tuple(sq_cfg.expected_eeg_channels),
                channel_statistics=(),
                generated_figures=(),
            ),
            None,
            [],
        )

    try:
        raw_mne = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
        sfreq_hz = float(raw_mne.info["sfreq"])
        n_samples = int(raw_mne.n_times)
        duration_sec = float(n_samples / sfreq_hz) if sfreq_hz > 0 else 0.0
        ch_names = list(raw_mne.ch_names)
        mne_types = list(raw_mne.get_channel_types())
        hdr_by_idx = {ch.channel_index: ch for ch in raw_header.channels}
        native_signals, native_units = _extract_native_unit_signals(raw_mne, hdr_by_idx)
        raw_mne.close()
    except Exception as exc:  # noqa: BLE001
        return (
            RecordingQualityInspection(
                recording=recording,
                is_readable=False,
                error_message=f"MNE failed to read EDF signal data: {exc}",
                sfreq_hz=None,
                n_samples=None,
                duration_sec=None,
                edf_patient_id_raw=raw_header.local_patient_id,
                edf_recording_id_raw=raw_header.local_recording_id,
                present_channel_names=(),
                candidate_eeg_channel_names=(),
                non_eeg_channel_names=(),
                unknown_channel_names=(),
                missing_expected_eeg_channels=tuple(sq_cfg.expected_eeg_channels),
                channel_statistics=(),
                generated_figures=(),
            ),
            None,
            [],
        )

    stats_list: list[ChannelSignalStatistics] = []
    cand_eeg: list[str] = []
    non_eeg: list[str] = []
    unknown_ch: list[str] = []
    cleaned_present_upper: set[str] = set()

    for idx, ch_name in enumerate(ch_names):
        hdr_ch = hdr_by_idx.get(idx)
        mne_type = mne_types[idx] if idx < len(mne_types) else "unknown"
        phys_unit = native_units[idx] if idx < len(native_units) else "unspecified"

        ch_stat = compute_channel_signal_statistics(
            native_signals[idx],
            sfreq_hz=sfreq_hz,
            relative_source_path=recording.relative_source_path,
            original_filename=recording.original_filename,
            task_category=recording.task_category,
            raw_label=recording.raw_label,
            normalized_label=recording.normalized_label,
            unverified_candidate_prefix=recording.unverified_candidate_prefix,
            channel_index=idx,
            raw_channel_name=ch_name,
            mne_assigned_type=mne_type,
            physical_unit=phys_unit,
            edf_physical_min=hdr_ch.physical_min if hdr_ch else None,
            edf_physical_max=hdr_ch.physical_max if hdr_ch else None,
            edf_transducer_type=hdr_ch.transducer_type if hdr_ch else "",
            edf_prefiltering=hdr_ch.prefiltering if hdr_ch else "",
            sq_config=sq_cfg,
            audit_config=aud_cfg,
        )
        stats_list.append(ch_stat)
        cleaned_present_upper.add(ch_stat.cleaned_channel_name.upper())

        if ch_stat.is_candidate_eeg:
            cand_eeg.append(ch_name)
        elif ch_stat.audited_channel_role.startswith("non_eeg"):
            non_eeg.append(ch_name)
        else:
            unknown_ch.append(ch_name)

    missing_expected = tuple(
        exp_ch
        for exp_ch in sq_cfg.expected_eeg_channels
        if exp_ch.strip().upper() not in cleaned_present_upper
    )

    fig_paths: list[str] = []
    if sq_cfg.generate_plots and figures_dir is not None:
        fig_paths = generate_recording_figures(
            recording=recording,
            sfreq_hz=sfreq_hz,
            native_signals=native_signals,
            channel_stats=stats_list,
            sq_config=sq_cfg,
            figures_dir=figures_dir,
        )

    inspection = RecordingQualityInspection(
        recording=recording,
        is_readable=True,
        error_message=None,
        sfreq_hz=sfreq_hz,
        n_samples=n_samples,
        duration_sec=duration_sec,
        edf_patient_id_raw=raw_header.local_patient_id,
        edf_recording_id_raw=raw_header.local_recording_id,
        present_channel_names=tuple(ch_names),
        candidate_eeg_channel_names=tuple(cand_eeg),
        non_eeg_channel_names=tuple(non_eeg),
        unknown_channel_names=tuple(unknown_ch),
        missing_expected_eeg_channels=missing_expected,
        channel_statistics=tuple(stats_list),
        generated_figures=tuple(fig_paths),
    )
    return inspection, native_signals, native_units


def _save_figure(fig: Figure, dest_path: Path) -> str:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    FigureCanvasAgg(fig)
    fig.savefig(dest_path, dpi=150, bbox_inches="tight")
    fig.clear()
    return dest_path.as_posix()


def generate_recording_figures(
    recording: SelectedRecordingInput,
    sfreq_hz: float,
    native_signals: np.ndarray,
    channel_stats: list[ChannelSignalStatistics],
    sq_config: SignalQualityConfig,
    figures_dir: Path,
) -> list[str]:
    """Generate multi-channel waveform, individual-channel detail, and amplitude plots."""
    saved_paths: list[str] = []
    task_tok = _sanitize_token(recording.task_category)
    stem_tok = _sanitize_token(Path(recording.original_filename).stem)
    lbl_display = recording.raw_label or "unknown"

    n_samples = native_signals.shape[1]
    if sq_config.plot_duration_sec is not None:
        stop_sample = min(n_samples, max(1, int(round(sq_config.plot_duration_sec * sfreq_hz))))
    else:
        stop_sample = n_samples

    time_sec = np.arange(stop_sample, dtype=float) / sfreq_hz

    # Identify candidate EEG channels (or fall back to all channels if none matched known EEG)
    eeg_indices = [st.channel_index for st in channel_stats if st.is_candidate_eeg]
    plot_indices = eeg_indices if eeg_indices else [st.channel_index for st in channel_stats]

    # 1. Stacked Multi-Channel Raw EEG Waveform Plot
    if plot_indices and stop_sample > 0:
        n_plot = len(plot_indices)
        fig_height = max(4.5, min(14.0, 0.55 * n_plot + 2.0))
        fig = Figure(figsize=(12, fig_height))
        ax = fig.add_subplot(111)

        stds = [
            channel_stats[i].measured_std or 0.0
            for i in plot_indices
            if (channel_stats[i].measured_std or 0.0) > 0
        ]
        spacing = float(np.median(stds) * 6.0) if stds else 50.0
        if spacing <= 0:
            spacing = 50.0

        yticks: list[float] = []
        yticklabels: list[str] = []

        for row_pos, ch_idx in enumerate(plot_indices):
            st = channel_stats[ch_idx]
            trace = native_signals[ch_idx, :stop_sample]
            offset = (n_plot - 1 - row_pos) * spacing
            finite_trace = trace[np.isfinite(trace)]
            mean_val = float(np.mean(finite_trace)) if finite_trace.size > 0 else 0.0
            centered = trace - mean_val

            has_flag = bool(st.heuristic_quality_flags)
            color = "#d95f02" if has_flag else "#1f77b4"
            label_tag = " [FLAG]" if has_flag else ""
            ax.plot(
                time_sec,
                centered + offset,
                color=color,
                linewidth=0.75,
                alpha=0.9,
            )
            yticks.append(offset)
            yticklabels.append(
                f"{st.cleaned_channel_name} ({st.physical_unit}){label_tag}"
            )

        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, fontsize=8)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Channel (each trace centered at its raw mean)")
        ax.set_title(
            f"Raw Multi-Channel EEG Waveforms | Task: {recording.task_category} | "
            f"Label: {lbl_display} | File: {recording.original_filename}\n"
            f"(Vertical spacing = {spacing:.2f} native units; blue=normal, orange=heuristic flag)"
        )
        ax.grid(True, linestyle="--", alpha=0.3)

        multi_path = figures_dir / f"raw_waveform_{task_tok}_{stem_tok}.png"
        saved_paths.append(_save_figure(fig, multi_path))

    # 2. Individual-Channel Detail Plots for Focus/Flagged EEG Channels
    focus_upper = {ch.upper() for ch in sq_config.focus_channels_for_individual_plots}
    selected_indiv_indices: list[int] = []
    for ch_idx in plot_indices:
        st = channel_stats[ch_idx]
        if st.cleaned_channel_name.upper() in focus_upper or st.heuristic_quality_flags:
            selected_indiv_indices.append(ch_idx)

    if not selected_indiv_indices and plot_indices:
        selected_indiv_indices = plot_indices[:1]

    selected_indiv_indices = selected_indiv_indices[
        : sq_config.max_individual_channel_plots_per_recording
    ]

    for ch_idx in selected_indiv_indices:
        st = channel_stats[ch_idx]
        trace = native_signals[ch_idx, :stop_sample]
        ch_tok = _sanitize_token(st.cleaned_channel_name)

        fig = Figure(figsize=(10, 3.8))
        ax = fig.add_subplot(111)
        ax.plot(
            time_sec,
            trace,
            color="#2b5c8f",
            linewidth=0.85,
            label=f"{st.raw_channel_name} ({st.physical_unit})",
        )
        if st.measured_mean is not None:
            ax.axhline(
                st.measured_mean,
                color="#e41a1c",
                linestyle="--",
                linewidth=0.9,
                label=f"Mean = {st.measured_mean:.2f} {st.physical_unit}",
            )
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel(f"Amplitude ({st.physical_unit})")
        flag_note = st.heuristic_quality_flags if st.heuristic_quality_flags else "none"
        std_val = st.measured_std if st.measured_std is not None else float("nan")
        ptp_val = (
            st.measured_amplitude_range_ptp
            if st.measured_amplitude_range_ptp is not None
            else float("nan")
        )
        ax.set_title(
            f"Channel {st.raw_channel_name} | Task: {recording.task_category} "
            f"(Label: {lbl_display}) | {recording.original_filename}\n"
            f"Std={std_val:.3f} {st.physical_unit}, Range={ptp_val:.3f} {st.physical_unit}, "
            f"Heuristic Flags: {flag_note}"
        )
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.4)

        indiv_path = figures_dir / f"channel_detail_{task_tok}_{stem_tok}_{ch_tok}.png"
        saved_paths.append(_save_figure(fig, indiv_path))

    # 3. Channel Amplitude Distribution Plot for this Recording
    if plot_indices:
        fig = Figure(figsize=(11, max(4.0, 0.4 * len(plot_indices) + 1.8)))
        ax = fig.add_subplot(111)

        centered_distributions: list[np.ndarray] = []
        ch_labels: list[str] = []
        for ch_idx in plot_indices:
            st = channel_stats[ch_idx]
            vals = native_signals[ch_idx]
            finite_v = vals[np.isfinite(vals)]
            if finite_v.size > 0:
                centered_distributions.append(finite_v - float(np.mean(finite_v)))
            else:
                centered_distributions.append(np.array([0.0]))
            ch_labels.append(f"{st.cleaned_channel_name} ({st.physical_unit})")

        ax.boxplot(
            centered_distributions,
            orientation="horizontal",
            tick_labels=ch_labels,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "#a6bddb", "edgecolor": "#045a8d"},
            medianprops={"color": "#d95f02", "linewidth": 1.4},
        )
        ax.set_xlabel("Mean-Centered Amplitude (original physical units, outliers hidden for scale)")
        ax.set_ylabel("Channel")
        ax.set_title(
            f"Channel Amplitude Distributions (IQR & Whiskers) | "
            f"Task: {recording.task_category} | {recording.original_filename}"
        )
        ax.grid(True, axis="x", linestyle="--", alpha=0.4)

        dist_path = figures_dir / f"amplitude_distribution_{task_tok}_{stem_tok}.png"
        saved_paths.append(_save_figure(fig, dist_path))

    return saved_paths


def generate_cross_task_comparison_figure(
    representative_by_task: dict[str, tuple[RecordingQualityInspection, np.ndarray]],
    sq_config: SignalQualityConfig,
    figures_dir: Path,
) -> str | None:
    """Create a comparison plot of representative recordings across Char, Digit, and Image."""
    if not representative_by_task:
        return None

    tasks_present = list(representative_by_task.keys())
    n_tasks = len(tasks_present)
    fig = Figure(figsize=(13, 3.6 * n_tasks))

    for idx, task_name in enumerate(tasks_present):
        insp, sig_matrix = representative_by_task[task_name]
        sfreq = insp.sfreq_hz or 1.0
        n_samples = sig_matrix.shape[1]
        stop_sample = (
            min(n_samples, max(1, int(round(sq_config.plot_duration_sec * sfreq))))
            if sq_config.plot_duration_sec is not None
            else n_samples
        )
        t_sec = np.arange(stop_sample, dtype=float) / sfreq

        # Pick first candidate EEG channel (or channel 0) for waveform panel
        eeg_stats = [st for st in insp.channel_statistics if st.is_candidate_eeg]
        rep_stat = eeg_stats[0] if eeg_stats else insp.channel_statistics[0]
        trace = sig_matrix[rep_stat.channel_index, :stop_sample]

        ax_wave = fig.add_subplot(n_tasks, 2, 2 * idx + 1)
        ax_wave.plot(t_sec, trace, color="#1b9e77", linewidth=0.8)
        ax_wave.set_title(
            f"[{task_name}] {insp.recording.original_filename} "
            f"(Label: {insp.recording.raw_label}) - Ch: {rep_stat.cleaned_channel_name}",
            fontsize=9,
        )
        ax_wave.set_xlabel("Time (s)", fontsize=8)
        ax_wave.set_ylabel(f"Amplitude ({rep_stat.physical_unit})", fontsize=8)
        ax_wave.grid(True, linestyle=":", alpha=0.4)

        # Right panel: per-channel standard deviation bar chart across candidate EEG channels
        ax_std = fig.add_subplot(n_tasks, 2, 2 * idx + 2)
        target_stats = eeg_stats if eeg_stats else list(insp.channel_statistics)
        ch_names = [s.cleaned_channel_name for s in target_stats]
        ch_stds = [s.measured_std or 0.0 for s in target_stats]
        bar_colors = [
            "#d95f02" if s.heuristic_quality_flags else "#7570b3" for s in target_stats
        ]
        ax_std.bar(range(len(ch_names)), ch_stds, color=bar_colors)
        ax_std.set_xticks(range(len(ch_names)))
        ax_std.set_xticklabels(ch_names, rotation=45, ha="right", fontsize=7)
        ax_std.set_title(
            f"[{task_name}] Channel Std Dev ({rep_stat.physical_unit}) | "
            f"Duration={insp.duration_sec:.2f}s, Fs={sfreq:.1f}Hz",
            fontsize=9,
        )
        ax_std.set_ylabel(f"Std ({rep_stat.physical_unit})", fontsize=8)
        ax_std.grid(True, axis="y", linestyle="--", alpha=0.4)

    fig.suptitle(
        "Cross-Task Comparison of Representative Recordings (Char / Digit / Image)",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    dest = figures_dir / "task_comparison_Char_Digit_Image.png"
    return _save_figure(fig, dest)


def generate_dataset_amplitude_figure(
    all_channel_stats: list[ChannelSignalStatistics],
    figures_dir: Path,
) -> str | None:
    """Plot aggregated per-channel standard deviation across all inspected EEG recordings."""
    eeg_rows = [
        st
        for st in all_channel_stats
        if st.is_candidate_eeg and st.measured_std is not None
    ]
    if not eeg_rows:
        return None

    df = pd.DataFrame([asdict(r) for r in eeg_rows])
    grouped = df.groupby("cleaned_channel_name", sort=False)["measured_std"].apply(list)

    fig = Figure(figsize=(11, 4.5))
    ax = fig.add_subplot(111)
    labels = list(grouped.index)
    data = list(grouped.values)

    ax.boxplot(
        data,
        orientation="vertical",
        tick_labels=labels,
        patch_artist=True,
        boxprops={"facecolor": "#ccebc5", "edgecolor": "#238b45"},
        medianprops={"color": "#d95f02", "linewidth": 1.5},
    )
    unit_set = sorted({r.physical_unit for r in eeg_rows})
    unit_str = ", ".join(unit_set)
    ax.set_xlabel("Candidate EEG Channel")
    ax.set_ylabel(f"Measured Channel Std Dev ({unit_str})")
    ax.set_title("Per-Channel Signal Standard Deviation Across Inspected Recordings")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    dest = figures_dir / "dataset_channel_amplitude_distribution.png"
    return _save_figure(fig, dest)


def _build_quality_summary_and_markdown(
    config: ProjectConfig,
    manifest_df: pd.DataFrame,
    inspections: list[RecordingQualityInspection],
    all_channel_stats: list[ChannelSignalStatistics],
    errors: list[InspectionErrorRecord],
    generated_figures: list[str],
) -> tuple[dict[str, Any], str]:
    """Build structured JSON summary and Markdown report for Phase 2 signal quality."""
    readable_inspections = [i for i in inspections if i.is_readable]

    # 1. Recording-duration and sampling-rate variability across manifest and inspected set
    manifest_valid = manifest_df[manifest_df["is_valid_edf"].astype(bool)]
    duration_by_task: dict[str, Any] = {}
    for task in config.discovery.expected_tasks:
        sub = manifest_valid[
            manifest_valid["task_category"].astype(str).str.lower() == task.lower()
        ]
        if not sub.empty and "duration_sec" in sub.columns:
            durs = pd.to_numeric(sub["duration_sec"], errors="coerce").dropna().to_numpy()
            if durs.size > 0:
                duration_by_task[task] = {
                    "n_valid_recordings_in_manifest": int(durs.size),
                    "min_sec": float(np.min(durs)),
                    "max_sec": float(np.max(durs)),
                    "mean_sec": float(np.mean(durs)),
                    "median_sec": float(np.median(durs)),
                    "std_sec": float(np.std(durs)),
                }

    sfreqs_inspected = sorted(
        {i.sfreq_hz for i in readable_inspections if i.sfreq_hz is not None}
    )

    # 2. Channel consistency across inspected recordings
    missing_by_recording = {
        i.recording.relative_source_path: list(i.missing_expected_eeg_channels)
        for i in readable_inspections
        if i.missing_expected_eeg_channels
    }
    channel_sets = {
        tuple(i.candidate_eeg_channel_names) for i in readable_inspections
    }

    # 3. Amplitude distribution summary across candidate EEG channels
    eeg_stats = [s for s in all_channel_stats if s.is_candidate_eeg]
    units_observed = sorted({s.physical_unit for s in all_channel_stats})
    calibration_questions = sorted(
        {
            f"{s.relative_source_path} [{s.raw_channel_name}]: {s.calibration_or_unit_note}"
            for s in all_channel_stats
            if s.calibration_or_unit_note
        }
    )

    eeg_stds = [s.measured_std for s in eeg_stats if s.measured_std is not None]
    eeg_ptps = [
        s.measured_amplitude_range_ptp
        for s in eeg_stats
        if s.measured_amplitude_range_ptp is not None
    ]

    # 4. Heuristic quality flags summary
    flagged_channels = [
        {
            "relative_source_path": s.relative_source_path,
            "task_category": s.task_category,
            "raw_channel_name": s.raw_channel_name,
            "audited_channel_role": s.audited_channel_role,
            "heuristic_quality_flags": s.heuristic_quality_flags,
            "measured_std": s.measured_std,
            "measured_amplitude_range_ptp": s.measured_amplitude_range_ptp,
            "n_non_finite": s.n_non_finite,
        }
        for s in all_channel_stats
        if s.heuristic_quality_flags
    ]

    summary: dict[str, Any] = {
        "seed": config.seed,
        "data_root": str(config.data_root),
        "n_manifest_recordings_total": int(len(manifest_df)),
        "n_manifest_valid_edf": int(len(manifest_valid)),
        "n_selected_recordings": len(inspections),
        "n_readable_inspected_recordings": len(readable_inspections),
        "n_inspection_errors": len(errors),
        "selected_recordings": [
            {
                "relative_source_path": i.recording.relative_source_path,
                "task_category": i.recording.task_category,
                "raw_label": i.recording.raw_label,
                "normalized_label": i.recording.normalized_label,
                "unverified_candidate_prefix": i.recording.unverified_candidate_prefix,
                "selection_reason": i.recording.selection_reason,
                "is_readable": i.is_readable,
                "duration_sec": i.duration_sec,
                "sfreq_hz": i.sfreq_hz,
                "n_candidate_eeg_channels": len(i.candidate_eeg_channel_names),
                "missing_expected_eeg_channels": list(i.missing_expected_eeg_channels),
            }
            for i in inspections
        ],
        "sampling_frequencies_hz_observed": sfreqs_inspected,
        "recording_duration_variability_by_task": duration_by_task,
        "channel_consistency": {
            "expected_eeg_channels": list(config.signal_quality.expected_eeg_channels),
            "is_candidate_eeg_set_identical_across_inspected": len(channel_sets) <= 1,
            "distinct_candidate_eeg_channel_tuples_count": len(channel_sets),
            "recordings_with_missing_expected_eeg_channels": missing_by_recording,
        },
        "physical_units_and_calibration": {
            "physical_units_observed": units_observed,
            "candidate_eeg_std_summary": (
                {
                    "min": float(np.min(eeg_stds)),
                    "median": float(np.median(eeg_stds)),
                    "max": float(np.max(eeg_stds)),
                }
                if eeg_stds
                else None
            ),
            "candidate_eeg_ptp_range_summary": (
                {
                    "min": float(np.min(eeg_ptps)),
                    "median": float(np.median(eeg_ptps)),
                    "max": float(np.max(eeg_ptps)),
                }
                if eeg_ptps
                else None
            ),
            "unresolved_unit_or_calibration_notes": calibration_questions,
        },
        "heuristic_quality_summary": {
            "n_total_channel_inspections": len(all_channel_stats),
            "n_flagged_channel_instances": len(flagged_channels),
            "n_flatline_instances": sum(1 for s in all_channel_stats if s.heuristic_is_flatline),
            "n_near_flatline_instances": sum(
                1 for s in all_channel_stats if s.heuristic_is_near_flatline
            ),
            "n_potential_clipping_instances": sum(
                1 for s in all_channel_stats if s.heuristic_potential_clipping
            ),
            "n_suspicious_repeated_instances": sum(
                1 for s in all_channel_stats if s.heuristic_suspicious_repeated_values
            ),
            "n_counter_like_instances": sum(
                1 for s in all_channel_stats if s.heuristic_is_counter_like
            ),
            "n_non_finite_instances": sum(
                1 for s in all_channel_stats if s.heuristic_has_non_finite
            ),
            "flagged_channels": flagged_channels,
        },
        "errors": [asdict(e) for e in errors],
        "generated_figures": generated_figures,
        "scientific_safeguards": {
            "preprocessing_applied": False,
            "imagined_speech_window_inferred": False,
            "participant_prefix_verified": False,
            "heuristic_flag_disclaimer": (
                "Heuristic flags (flatline, near-flatline, potential clipping, repeated values) "
                "are numerical indicators for quality review and do NOT prove that a signal "
                "is a physiological or hardware artifact."
            ),
        },
    }

    # Build Markdown report
    lines: list[str] = [
        "# Phase 2: EEG Signal Quality Analysis Report",
        "",
        "## 1. Overview & Scientific Safeguards",
        f"- **Dataset Root**: `{config.data_root}`",
        f"- **Random Seed**: `{config.seed}`",
        f"- **Total Recordings in Manifest**: `{len(manifest_df)}` (`{len(manifest_valid)}` valid EDFs)",
        f"- **Selected Representative Recordings Inspected**: `{len(readable_inspections)} / {len(inspections)}`",
        "- **Preprocessing Applied**: None (raw unmodified EDF signals in original physical units).",
        "- **Participant Identification Note**: Filename prefixes are reported strictly as `unverified_candidate_prefix` and are not assumed to identify participants.",
        "- **Heuristic Flag Disclaimer**: Heuristic flags (`heuristic_*`) are diagnostic indicators for human review and do not constitute definitive proof of artifacts.",
        "",
        "## 2. Selected Representative Recordings",
        "",
        "| Task | Relative Path | Raw Label | Unverified Prefix | Fs (Hz) | Duration (s) | Candidate EEG Ch | Missing Expected EEG Ch |",
        "| :--- | :--- | :--- | :--- | ---: | ---: | ---: | :--- |",
    ]

    for insp in inspections:
        rec = insp.recording
        if insp.is_readable:
            missing_str = (
                ", ".join(insp.missing_expected_eeg_channels)
                if insp.missing_expected_eeg_channels
                else "None"
            )
            lines.append(
                f"| `{rec.task_category}` | `{rec.relative_source_path}` | `{rec.raw_label}` | "
                f"`{rec.unverified_candidate_prefix}` | {insp.sfreq_hz:.1f} | "
                f"{insp.duration_sec:.2f} | {len(insp.candidate_eeg_channel_names)} | {missing_str} |"
            )
        else:
            lines.append(
                f"| `{rec.task_category}` | `{rec.relative_source_path}` | `{rec.raw_label}` | "
                f"`{rec.unverified_candidate_prefix}` | ERROR | ERROR | 0 | `{insp.error_message}` |"
            )

    lines.extend(
        [
            "",
            "## 3. Recording-Duration & Sampling-Rate Variability",
            f"- **Observed Sampling Frequencies (Hz)**: `{sfreqs_inspected}`",
            "",
            "| Task | Valid Manifest Recordings | Min Duration (s) | Median Duration (s) | Mean Duration (s) | Max Duration (s) | Std (s) |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for task, d_stats in duration_by_task.items():
        lines.append(
            f"| `{task}` | {d_stats['n_valid_recordings_in_manifest']} | "
            f"{d_stats['min_sec']:.2f} | {d_stats['median_sec']:.2f} | "
            f"{d_stats['mean_sec']:.2f} | {d_stats['max_sec']:.2f} | {d_stats['std_sec']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## 4. Channel Consistency & Physical Units",
            f"- **Configured Expected EEG Montage ({len(config.signal_quality.expected_eeg_channels)} channels)**: "
            f"`{', '.join(config.signal_quality.expected_eeg_channels)}`",
            f"- **Candidate EEG Channel Set Identical Across Inspected Files**: "
            f"`{summary['channel_consistency']['is_candidate_eeg_set_identical_across_inspected']}`",
            f"- **Observed Physical Units**: `{units_observed}`",
        ]
    )
    if calibration_questions:
        lines.append("- **Unresolved Unit / Calibration Observations**:")
        for note in calibration_questions[:20]:
            lines.append(f"  - `{note}`")
    else:
        lines.append("- **Unresolved Unit / Calibration Observations**: None detected.")

    lines.extend(
        [
            "",
            "## 5. Heuristic Signal-Quality Indicators",
            f"- **Total Channel Instances Inspected**: `{len(all_channel_stats)}`",
            f"- **Flagged Channel Instances**: `{len(flagged_channels)}`",
            f"  - Flatline (`heuristic_is_flatline`): `{summary['heuristic_quality_summary']['n_flatline_instances']}`",
            f"  - Near-Flatline (`heuristic_is_near_flatline`): `{summary['heuristic_quality_summary']['n_near_flatline_instances']}`",
            f"  - Potential Clipping (`heuristic_potential_clipping`): `{summary['heuristic_quality_summary']['n_potential_clipping_instances']}`",
            f"  - Suspicious Repeated Values (`heuristic_suspicious_repeated_values`): `{summary['heuristic_quality_summary']['n_suspicious_repeated_instances']}`",
            f"  - Counter/Ramp Pattern (`heuristic_is_counter_like`): `{summary['heuristic_quality_summary']['n_counter_like_instances']}`",
            f"  - Non-Finite Samples (`heuristic_has_non_finite`): `{summary['heuristic_quality_summary']['n_non_finite_instances']}`",
        ]
    )

    if flagged_channels:
        lines.extend(
            [
                "",
                "### Channels with Heuristic Quality Flags",
                "| Recording | Task | Channel | Audited Role | Measured Std | Measured PtP Range | Heuristic Flags |",
                "| :--- | :--- | :--- | :--- | ---: | ---: | :--- |",
            ]
        )
        for fc in flagged_channels:
            std_s = f"{fc['measured_std']:.4f}" if fc["measured_std"] is not None else "N/A"
            ptp_s = (
                f"{fc['measured_amplitude_range_ptp']:.4f}"
                if fc["measured_amplitude_range_ptp"] is not None
                else "N/A"
            )
            lines.append(
                f"| `{fc['relative_source_path']}` | `{fc['task_category']}` | "
                f"`{fc['raw_channel_name']}` | `{fc['audited_channel_role']}` | "
                f"{std_s} | {ptp_s} | `{fc['heuristic_quality_flags']}` |"
            )

    if errors:
        lines.extend(
            [
                "",
                "## 6. Unreadable or Invalid Recording Errors",
            ]
        )
        for err in errors:
            lines.append(
                f"- `{err.relative_source_path}` (`{err.task_category}`, stage=`{err.error_stage}`): {err.error_message}"
            )

    lines.extend(
        [
            "",
            "## 7. Generated Visualizations",
        ]
    )
    for fig_p in generated_figures:
        lines.append(f"- `{fig_p}`")

    return summary, "\n".join(lines) + "\n"


def run_signal_quality_analysis(
    config: ProjectConfig,
    *,
    manifest_source: Path | str | pd.DataFrame | None = None,
    output_dir: Path | str | None = None,
    auto_run_audit_if_missing: bool = True,
    save_outputs: bool = True,
) -> SignalQualityAnalysisResult:
    """Execute the Phase 2 EEG signal-quality analysis and visualization pipeline.

    Parameters
    ----------
    config : ProjectConfig
        Validated project configuration.
    manifest_source : Path | str | pd.DataFrame | None
        Optional explicit path to ``manifest.csv`` or in-memory DataFrame. If ``None``,
        looks for ``config.output_dir / "manifest.csv"`` and runs Phase 1 audit if
        ``auto_run_audit_if_missing=True``.
    output_dir : Path | str | None
        Destination directory for ``signal_statistics.csv``, ``signal_quality_summary.json``,
        ``signal_quality_report.md``, and ``figures/``. Defaults to ``config.output_dir``.
    auto_run_audit_if_missing : bool
        If ``True`` and no manifest file exists yet, runs ``run_dataset_audit`` first.
    save_outputs : bool
        If ``True``, writes CSV, JSON, Markdown, and PNG figures to ``output_dir``.

    Returns
    -------
    SignalQualityAnalysisResult
        Structured results of the signal-quality analysis.
    """
    resolved_out_dir = Path(output_dir) if output_dir is not None else Path(config.output_dir)
    figures_dir = resolved_out_dir / "figures" if save_outputs else None

    if manifest_source is None:
        default_manifest_path = Path(config.output_dir) / "manifest.csv"
        if default_manifest_path.is_file():
            manifest_df = load_and_validate_manifest(default_manifest_path)
        elif auto_run_audit_if_missing:
            logger.info(
                "No existing manifest.csv at %s; running Phase 1 dataset audit first.",
                default_manifest_path,
            )
            audit_res = run_dataset_audit(config, save_outputs=save_outputs)
            manifest_df = pd.DataFrame([asdict(r) for r in audit_res.manifest_records])
            manifest_df = load_and_validate_manifest(manifest_df)
        else:
            raise SignalQualityError(
                f"Manifest file not found at {default_manifest_path} and "
                "auto_run_audit_if_missing=False."
            )
    else:
        manifest_df = load_and_validate_manifest(manifest_source)

    errors: list[InspectionErrorRecord] = []

    # Record any invalid EDF rows already flagged in the Phase 1 manifest
    invalid_manifest_rows = manifest_df[~manifest_df["is_valid_edf"].astype(bool)]
    for _, row in invalid_manifest_rows.iterrows():
        errors.append(
            InspectionErrorRecord(
                relative_source_path=str(row["relative_source_path"]),
                task_category=str(row["task_category"]),
                error_stage="phase1_manifest_validation",
                error_message=str(row.get("validation_error") or "Invalid EDF in manifest"),
            )
        )

    selected_recordings = select_representative_recordings(
        manifest_df=manifest_df,
        data_root=config.data_root,
        expected_tasks=config.discovery.expected_tasks,
        recordings_per_task=config.signal_quality.recordings_per_task,
        specific_recordings=config.signal_quality.specific_recordings,
        seed=config.seed,
    )

    inspections: list[RecordingQualityInspection] = []
    all_channel_stats: list[ChannelSignalStatistics] = []
    all_figures: list[str] = []
    first_readable_by_task: dict[str, tuple[RecordingQualityInspection, np.ndarray]] = {}

    for rec_input in selected_recordings:
        insp, sig_matrix, _units = inspect_recording_signal_quality(
            recording=rec_input,
            sq_config=config.signal_quality,
            audit_config=config.edf_audit,
            figures_dir=figures_dir,
        )
        inspections.append(insp)
        if not insp.is_readable:
            errors.append(
                InspectionErrorRecord(
                    relative_source_path=rec_input.relative_source_path,
                    task_category=rec_input.task_category,
                    error_stage="phase2_signal_inspection",
                    error_message=insp.error_message or "Unreadable EDF file",
                )
            )
        else:
            all_channel_stats.extend(insp.channel_statistics)
            all_figures.extend(insp.generated_figures)
            if (
                rec_input.task_category not in first_readable_by_task
                and sig_matrix is not None
            ):
                first_readable_by_task[rec_input.task_category] = (insp, sig_matrix)

    if config.signal_quality.generate_plots and figures_dir is not None:
        cross_fig = generate_cross_task_comparison_figure(
            representative_by_task=first_readable_by_task,
            sq_config=config.signal_quality,
            figures_dir=figures_dir,
        )
        if cross_fig is not None:
            all_figures.append(cross_fig)

        amp_fig = generate_dataset_amplitude_figure(
            all_channel_stats=all_channel_stats,
            figures_dir=figures_dir,
        )
        if amp_fig is not None:
            all_figures.append(amp_fig)

    summary, report_md = _build_quality_summary_and_markdown(
        config=config,
        manifest_df=manifest_df,
        inspections=inspections,
        all_channel_stats=all_channel_stats,
        errors=errors,
        generated_figures=all_figures,
    )

    if save_outputs:
        resolved_out_dir.mkdir(parents=True, exist_ok=True)
        stats_df = pd.DataFrame([asdict(s) for s in all_channel_stats])
        stats_df.to_csv(resolved_out_dir / "signal_statistics.csv", index=False)

        with (resolved_out_dir / "signal_quality_summary.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(summary, handle, indent=2, sort_keys=False)

        (resolved_out_dir / "signal_quality_report.md").write_text(
            report_md, encoding="utf-8"
        )

        logger.info(
            "Saved Phase 2 signal-quality outputs to %s (recordings=%d, channels=%d, figures=%d).",
            resolved_out_dir,
            len(inspections),
            len(all_channel_stats),
            len(all_figures),
        )

    return SignalQualityAnalysisResult(
        selected_recordings=selected_recordings,
        recording_inspections=inspections,
        channel_statistics=all_channel_stats,
        errors=errors,
        generated_figures=all_figures,
        summary=summary,
        report_markdown=report_md,
    )
