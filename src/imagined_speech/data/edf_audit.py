"""EDF header, channel name, and in-memory signal property auditing (Rules 7, 9, 10, 11, 12).

This module validates EDF recordings without modifying source files. In compliance with
Rule 10, it never assumes a channel is EEG merely because an EDF reader defaults its type
to ``'eeg'``. Instead, it cross-checks raw EDF header metadata, channel names against
configurable montages/non-EEG patterns, and actual in-memory signal properties (flatlines,
sample counters, discrete status channels, NaN/Inf).
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

from imagined_speech.config import EDFAuditConfig, ProjectConfig
from imagined_speech.data.discovery import DiscoveredRecording, DiscoveryReport, discover_dataset
from imagined_speech.exceptions import EDFValidationError

logger = logging.getLogger("imagined_speech.data.edf_audit")

_EEG_PREFIX_STRIP_RE = re.compile(
    r"^(?:EEG[\s_\-]+)|(?:[\s_\-]+(?:REF|LE|AR|AVG|A1|A2|M1|M2))$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RawEDFChannelHeader:
    """Per-channel metadata parsed directly from the binary EDF header block."""

    channel_index: int
    label: str
    transducer_type: str
    physical_dimension: str
    physical_min: float | None
    physical_max: float | None
    digital_min: int | None
    digital_max: int | None
    prefiltering: str
    samples_per_record: int | None


@dataclass(frozen=True)
class RawEDFFixedHeader:
    """Fixed 256-byte EDF header metadata and per-channel header records."""

    version: str
    local_patient_id: str
    local_recording_id: str
    start_date: str
    start_time: str
    header_bytes: int
    reserved: str
    n_data_records: int
    record_duration_sec: float
    n_channels: int
    channels: tuple[RawEDFChannelHeader, ...]


@dataclass(frozen=True)
class ChannelAuditRecord:
    """Detailed audit record for a single channel within an EDF recording (Rule 10)."""

    relative_source_path: str
    task_category: str
    channel_index: int
    raw_channel_name: str
    cleaned_channel_name: str
    mne_assigned_type: str
    edf_physical_dimension: str
    edf_transducer_type: str
    edf_prefiltering: str
    audited_channel_role: str
    is_candidate_eeg: bool
    has_usable_eeg_signal: bool
    is_flat: bool
    is_counter_like: bool
    has_nan_or_inf: bool
    n_unique_values_sampled: int | None
    signal_min: float | None
    signal_max: float | None
    signal_mean: float | None
    signal_std: float | None
    signal_ptp: float | None
    audit_flags: str


@dataclass(frozen=True)
class RecordingManifestRecord:
    """Manifest row for a single discovered EDF recording (Rules 10, 11, 12)."""

    original_filename: str
    relative_source_path: str
    task_category: str
    raw_label: str | None
    normalized_label: str | None
    unverified_candidate_prefix: str | None
    label_parse_pattern: str | None
    label_parse_warning: str | None
    file_size_bytes: int
    is_valid_edf: bool
    validation_error: str | None
    edf_patient_id_raw: str | None
    edf_recording_id_raw: str | None
    edf_start_date: str | None
    edf_start_time: str | None
    sfreq_hz: float | None
    n_samples: int | None
    duration_sec: float | None
    n_channels_total: int | None
    n_candidate_eeg_channels: int | None
    n_usable_eeg_channels: int | None
    n_non_eeg_channels: int | None
    n_unknown_channels: int | None
    n_flat_eeg_channels: int | None
    channel_names_json: str | None
    candidate_eeg_channels_json: str | None
    usable_eeg_channels_json: str | None
    non_eeg_channels_json: str | None
    unknown_channels_json: str | None
    flat_eeg_channels_json: str | None


@dataclass
class DatasetAuditResult:
    """Complete Phase 1 dataset discovery and EDF audit outputs."""

    manifest_records: list[RecordingManifestRecord] = field(default_factory=list)
    channel_records: list[ChannelAuditRecord] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def _safe_float(text: str) -> float | None:
    try:
        return float(text.strip())
    except ValueError:
        return None


def _safe_int(text: str) -> int | None:
    try:
        return int(text.strip())
    except ValueError:
        return None


def read_raw_edf_header(file_path: Path | str) -> RawEDFFixedHeader:
    """Parse the binary EDF fixed header (256 bytes) and channel headers in read-only mode.

    Parameters
    ----------
    file_path : Path | str
        Path to the EDF file.

    Returns
    -------
    RawEDFFixedHeader
        Parsed fixed and per-channel header fields.

    Raises
    ------
    EDFValidationError
        If the file is truncated, unreadable, or violates the EDF header structure.
    """
    path = Path(file_path)
    try:
        file_size = path.stat().st_size
    except OSError as exc:
        raise EDFValidationError(f"Cannot stat file {path}: {exc}") from exc

    if file_size < 256:
        raise EDFValidationError(
            f"File is smaller than 256-byte EDF fixed header ({file_size} bytes): {path.name}"
        )

    try:
        with path.open("rb") as handle:
            fixed = handle.read(256)
            if len(fixed) < 256:
                raise EDFValidationError(f"Truncated 256-byte fixed EDF header in {path.name}")

            version = fixed[0:8].decode("ascii", errors="replace").strip()
            local_patient_id = fixed[8:88].decode("ascii", errors="replace").strip()
            local_recording_id = fixed[88:168].decode("ascii", errors="replace").strip()
            start_date = fixed[168:176].decode("ascii", errors="replace").strip()
            start_time = fixed[176:184].decode("ascii", errors="replace").strip()
            header_bytes_raw = fixed[184:192].decode("ascii", errors="replace").strip()
            reserved = fixed[192:236].decode("ascii", errors="replace").strip()
            n_records_raw = fixed[236:244].decode("ascii", errors="replace").strip()
            duration_raw = fixed[244:252].decode("ascii", errors="replace").strip()
            n_channels_raw = fixed[252:256].decode("ascii", errors="replace").strip()

            try:
                header_bytes = int(header_bytes_raw)
                n_data_records = int(n_records_raw)
                record_duration_sec = float(duration_raw)
                n_channels = int(n_channels_raw)
            except ValueError as exc:
                raise EDFValidationError(
                    f"Malformed numeric fields in fixed EDF header of {path.name}: {exc}"
                ) from exc

            if n_channels <= 0:
                raise EDFValidationError(
                    f"Invalid number of channels ({n_channels}) in EDF header of {path.name}"
                )

            expected_header_bytes = 256 + (n_channels * 256)
            if file_size < expected_header_bytes:
                raise EDFValidationError(
                    f"File size ({file_size}) is smaller than total EDF header size "
                    f"({expected_header_bytes}) in {path.name}"
                )

            ch_block = handle.read(n_channels * 256)
            if len(ch_block) < n_channels * 256:
                raise EDFValidationError(f"Truncated channel header block in {path.name}")

    except OSError as exc:
        raise EDFValidationError(f"I/O error reading EDF header from {path.name}: {exc}") from exc

    def _slice_field(offset_per_ch: int, width: int) -> list[str]:
        start = offset_per_ch * n_channels
        return [
            ch_block[start + i * width : start + (i + 1) * width]
            .decode("ascii", errors="replace")
            .strip()
            for i in range(n_channels)
        ]

    labels = _slice_field(0, 16)
    transducers = _slice_field(16, 80)
    phys_dims = _slice_field(96, 8)
    phys_mins = _slice_field(104, 8)
    phys_maxs = _slice_field(112, 8)
    dig_mins = _slice_field(120, 8)
    dig_maxs = _slice_field(128, 8)
    prefilters = _slice_field(136, 80)
    spr_list = _slice_field(216, 8)

    channels: list[RawEDFChannelHeader] = []
    for idx in range(n_channels):
        channels.append(
            RawEDFChannelHeader(
                channel_index=idx,
                label=labels[idx],
                transducer_type=transducers[idx],
                physical_dimension=phys_dims[idx],
                physical_min=_safe_float(phys_mins[idx]),
                physical_max=_safe_float(phys_maxs[idx]),
                digital_min=_safe_int(dig_mins[idx]),
                digital_max=_safe_int(dig_maxs[idx]),
                prefiltering=prefilters[idx],
                samples_per_record=_safe_int(spr_list[idx]),
            )
        )

    return RawEDFFixedHeader(
        version=version,
        local_patient_id=local_patient_id,
        local_recording_id=local_recording_id,
        start_date=start_date,
        start_time=start_time,
        header_bytes=header_bytes,
        reserved=reserved,
        n_data_records=n_data_records,
        record_duration_sec=record_duration_sec,
        n_channels=n_channels,
        channels=tuple(channels),
    )


def clean_channel_name(raw_name: str) -> str:
    """Normalize a channel name by stripping common EEG prefix/reference suffixes."""
    cleaned = raw_name.strip()
    cleaned = _EEG_PREFIX_STRIP_RE.sub("", cleaned).strip()
    # Also handle dotted prefixes like "EEG.AF3" or "AF3-LE"
    if "." in cleaned and cleaned.upper().startswith("EEG."):
        cleaned = cleaned.split(".", 1)[1].strip()
    return cleaned


def detect_counter_like_signal(signal_1d: np.ndarray) -> bool:
    """Detect whether a 1D signal behaves like a monotonic ramp or wrapping sample counter.

    Many consumer/research EEG headsets (e.g., Emotiv EPOC+) include a ``COUNTER`` channel
    that increments by 1 each sample and wraps modulo 128 or 256. EDF readers often label
    such channels as EEG unless signal properties are checked (Rule 10).
    """
    if signal_1d.size < 16 or not np.all(np.isfinite(signal_1d)):
        return False

    diffs = np.diff(signal_1d)
    if diffs.size == 0:
        return False

    # Scale-invariant check OR direct integer-step check:
    # Note: MNE converts physical units (e.g. uV -> V by multiplying by 1e-6), so a step of 1
    # in raw units may appear as 1e-6 in Volts. Normalize by the positive median step.
    pos_diffs = diffs[diffs > 0]
    if pos_diffs.size < int(0.8 * diffs.size):
        return False

    median_step = float(np.median(pos_diffs))
    if median_step <= 0:
        return False

    unit_step_Mask = np.isclose(diffs, median_step, rtol=1e-2, atol=1e-9)
    wrap_mask = diffs < 0  # Counter wrap-around (e.g., 127 -> 0)
    fraction_unit_or_wrap = float(np.mean(unit_step_Mask | wrap_mask))
    fraction_unit = float(np.mean(unit_step_Mask))

    return fraction_unit >= 0.85 and fraction_unit_or_wrap >= 0.98


def classify_channel(
    raw_channel_name: str,
    mne_assigned_type: str,
    edf_phys_dim: str,
    signal_1d: np.ndarray | None,
    config: EDFAuditConfig,
) -> tuple[str, bool, bool, bool, bool, bool, int | None, list[str]]:
    """Classify a single channel based on name, header units, and actual signal properties.

    Returns
    -------
    tuple
        ``(audited_role, is_candidate_eeg, has_usable_eeg_signal, is_flat, is_counter_like,
        has_nan_or_inf, n_unique_sampled, flags)``
    """
    flags: list[str] = []
    cleaned = clean_channel_name(raw_channel_name)
    upper_raw = raw_channel_name.strip().upper()
    upper_clean = cleaned.upper()

    known_eeg_set = {name.upper() for name in config.known_eeg_channel_names}
    non_eeg_substrings = tuple(sub.upper() for sub in config.known_non_eeg_substrings)

    is_flat = False
    is_counter_like = False
    has_nan_or_inf = False
    n_unique: int | None = None

    if signal_1d is not None and signal_1d.size > 0:
        has_nan_or_inf = bool(not np.all(np.isfinite(signal_1d)))
        if has_nan_or_inf:
            flags.append("nan_or_inf_values")
        else:
            std_val = float(np.std(signal_1d))
            ptp_val = float(np.ptp(signal_1d))
            is_flat = bool(
                std_val <= config.flat_std_threshold or ptp_val <= config.flat_std_threshold
            )
            if is_flat:
                flags.append("flat_constant_signal")
            else:
                is_counter_like = detect_counter_like_signal(signal_1d)
                if is_counter_like:
                    flags.append("counter_or_ramp_signal")

            sample_slice = signal_1d[: min(signal_1d.size, 5000)]
            n_unique = int(np.unique(np.round(sample_slice, decimals=12)).size)
            if not is_flat and n_unique <= 3 and signal_1d.size >= 32:
                flags.append(f"discrete_low_cardinality(n={n_unique})")

    # Step 1: Check explicit non-EEG channel names/substrings or MNE non-EEG annotations
    matched_non_eeg_sub = any(
        sub in upper_raw or sub in upper_clean for sub in non_eeg_substrings
    )
    if mne_assigned_type.lower() in {"stim", "misc", "eog", "ecg", "emg", "syst"}:
        matched_non_eeg_sub = True

    if matched_non_eeg_sub:
        if mne_assigned_type.lower() == "eeg":
            flags.append("mne_misclassified_as_eeg")
        return (
            "non_eeg_metadata_or_aux",
            False,
            False,
            is_flat,
            is_counter_like,
            has_nan_or_inf,
            n_unique,
            flags,
        )

    # Step 2: Check if channel is a known standard EEG electrode name
    if upper_clean in known_eeg_set:
        has_usable = not (is_flat or is_counter_like or has_nan_or_inf)
        if not has_usable:
            flags.append("candidate_eeg_degraded_signal")
        return (
            "candidate_eeg",
            True,
            has_usable,
            is_flat,
            is_counter_like,
            has_nan_or_inf,
            n_unique,
            flags,
        )

    # Step 3: Unrecognized channel name — check if its signal proves it's a sample counter
    if is_counter_like:
        if mne_assigned_type.lower() == "eeg":
            flags.append("mne_misclassified_as_eeg")
        return (
            "non_eeg_counter_signal",
            False,
            False,
            is_flat,
            is_counter_like,
            has_nan_or_inf,
            n_unique,
            flags,
        )

    # Step 4: Unknown channel name; do NOT blindly assume it is EEG (Rule 10)
    flags.append("unrecognized_channel_name_needs_verification")
    if edf_phys_dim and edf_phys_dim.lower() not in {"uv", "mv", "v", "microvolt", "µv"}:
        flags.append(f"non_voltage_unit({edf_phys_dim})")

    return (
        "unknown_needs_review",
        False,
        False,
        is_flat,
        is_counter_like,
        has_nan_or_inf,
        n_unique,
        flags,
    )


def audit_single_edf(
    discovered: DiscoveredRecording,
    config: EDFAuditConfig | None = None,
) -> tuple[RecordingManifestRecord, list[ChannelAuditRecord]]:
    """Validate and audit a single discovered EDF recording (Rules 9, 10, 11, 12).

    Parameters
    ----------
    discovered : DiscoveredRecording
        Discovered file entry from ``discover_dataset``.
    config : EDFAuditConfig | None
        Audit configuration. Defaults to ``EDFAuditConfig()``.

    Returns
    -------
    tuple[RecordingManifestRecord, list[ChannelAuditRecord]]
        Manifest record for the file and per-channel audit records.
    """
    cfg = config or EDFAuditConfig()
    parsed = discovered.parsed_label

    raw_label = parsed.raw_label if parsed else None
    norm_label = parsed.normalized_label if parsed else None
    unverified_prefix = parsed.unverified_candidate_prefix if parsed else None
    parse_pattern = parsed.parse_pattern if parsed else None
    parse_warning = (
        parsed.parse_warning if parsed else discovered.label_parse_error
    )

    try:
        raw_header = read_raw_edf_header(discovered.absolute_path)
    except EDFValidationError as exc:
        return (
            RecordingManifestRecord(
                original_filename=discovered.original_filename,
                relative_source_path=discovered.relative_source_path,
                task_category=discovered.task_category,
                raw_label=raw_label,
                normalized_label=norm_label,
                unverified_candidate_prefix=unverified_prefix,
                label_parse_pattern=parse_pattern,
                label_parse_warning=parse_warning,
                file_size_bytes=discovered.file_size_bytes,
                is_valid_edf=False,
                validation_error=str(exc),
                edf_patient_id_raw=None,
                edf_recording_id_raw=None,
                edf_start_date=None,
                edf_start_time=None,
                sfreq_hz=None,
                n_samples=None,
                duration_sec=None,
                n_channels_total=None,
                n_candidate_eeg_channels=None,
                n_usable_eeg_channels=None,
                n_non_eeg_channels=None,
                n_unknown_channels=None,
                n_flat_eeg_channels=None,
                channel_names_json=None,
                candidate_eeg_channels_json=None,
                usable_eeg_channels_json=None,
                non_eeg_channels_json=None,
                unknown_channels_json=None,
                flat_eeg_channels_json=None,
            ),
            [],
        )

    try:
        raw_mne = mne.io.read_raw_edf(
            discovered.absolute_path,
            preload=False,
            verbose="ERROR",
        )
        sfreq = float(raw_mne.info["sfreq"])
        n_samples = int(raw_mne.n_times)
        duration_sec = float(n_samples / sfreq) if sfreq > 0 else 0.0
        mne_ch_names: list[str] = list(raw_mne.ch_names)
        mne_ch_types: list[str] = list(raw_mne.get_channel_types())

        signals_matrix: np.ndarray | None = None
        if cfg.inspect_signal_properties and n_samples > 0:
            stop_sample: int | None = None
            if cfg.max_inspection_duration_sec is not None:
                stop_sample = min(
                    n_samples, max(1, int(round(cfg.max_inspection_duration_sec * sfreq)))
                )
            signals_matrix = raw_mne.get_data(stop=stop_sample)
        raw_mne.close()
    except Exception as exc:  # noqa: BLE001
        return (
            RecordingManifestRecord(
                original_filename=discovered.original_filename,
                relative_source_path=discovered.relative_source_path,
                task_category=discovered.task_category,
                raw_label=raw_label,
                normalized_label=norm_label,
                unverified_candidate_prefix=unverified_prefix,
                label_parse_pattern=parse_pattern,
                label_parse_warning=parse_warning,
                file_size_bytes=discovered.file_size_bytes,
                is_valid_edf=False,
                validation_error=f"MNE read_raw_edf failed: {exc}",
                edf_patient_id_raw=raw_header.local_patient_id,
                edf_recording_id_raw=raw_header.local_recording_id,
                edf_start_date=raw_header.start_date,
                edf_start_time=raw_header.start_time,
                sfreq_hz=None,
                n_samples=None,
                duration_sec=None,
                n_channels_total=raw_header.n_channels,
                n_candidate_eeg_channels=None,
                n_usable_eeg_channels=None,
                n_non_eeg_channels=None,
                n_unknown_channels=None,
                n_flat_eeg_channels=None,
                channel_names_json=None,
                candidate_eeg_channels_json=None,
                usable_eeg_channels_json=None,
                non_eeg_channels_json=None,
                unknown_channels_json=None,
                flat_eeg_channels_json=None,
            ),
            [],
        )

    channel_records: list[ChannelAuditRecord] = []
    candidate_eeg_names: list[str] = []
    usable_eeg_names: list[str] = []
    non_eeg_names: list[str] = []
    unknown_names: list[str] = []
    flat_eeg_names: list[str] = []

    raw_hdr_by_idx = {ch.channel_index: ch for ch in raw_header.channels}

    for idx, ch_name in enumerate(mne_ch_names):
        mne_type = mne_ch_types[idx] if idx < len(mne_ch_types) else "unknown"
        hdr_ch = raw_hdr_by_idx.get(idx)
        phys_dim = hdr_ch.physical_dimension if hdr_ch else ""
        transducer = hdr_ch.transducer_type if hdr_ch else ""
        prefilt = hdr_ch.prefiltering if hdr_ch else ""

        sig_1d: np.ndarray | None = None
        sig_min: float | None = None
        sig_max: float | None = None
        sig_mean: float | None = None
        sig_std: float | None = None
        sig_ptp: float | None = None

        if signals_matrix is not None and idx < signals_matrix.shape[0]:
            sig_1d = signals_matrix[idx]
            if sig_1d.size > 0 and np.all(np.isfinite(sig_1d)):
                sig_min = float(np.min(sig_1d))
                sig_max = float(np.max(sig_1d))
                sig_mean = float(np.mean(sig_1d))
                sig_std = float(np.std(sig_1d))
                sig_ptp = float(np.ptp(sig_1d))

        (
            audited_role,
            is_cand_eeg,
            has_usable_eeg,
            is_flat,
            is_counter,
            has_nan,
            n_unique,
            flags,
        ) = classify_channel(
            raw_channel_name=ch_name,
            mne_assigned_type=mne_type,
            edf_phys_dim=phys_dim,
            signal_1d=sig_1d,
            config=cfg,
        )

        if is_cand_eeg:
            candidate_eeg_names.append(ch_name)
            if has_usable_eeg:
                usable_eeg_names.append(ch_name)
            if is_flat:
                flat_eeg_names.append(ch_name)
        elif audited_role.startswith("non_eeg"):
            non_eeg_names.append(ch_name)
        else:
            unknown_names.append(ch_name)

        channel_records.append(
            ChannelAuditRecord(
                relative_source_path=discovered.relative_source_path,
                task_category=discovered.task_category,
                channel_index=idx,
                raw_channel_name=ch_name,
                cleaned_channel_name=clean_channel_name(ch_name),
                mne_assigned_type=mne_type,
                edf_physical_dimension=phys_dim,
                edf_transducer_type=transducer,
                edf_prefiltering=prefilt,
                audited_channel_role=audited_role,
                is_candidate_eeg=is_cand_eeg,
                has_usable_eeg_signal=has_usable_eeg,
                is_flat=is_flat,
                is_counter_like=is_counter,
                has_nan_or_inf=has_nan,
                n_unique_values_sampled=n_unique,
                signal_min=sig_min,
                signal_max=sig_max,
                signal_mean=sig_mean,
                signal_std=sig_std,
                signal_ptp=sig_ptp,
                audit_flags=";".join(flags),
            )
        )

    manifest_record = RecordingManifestRecord(
        original_filename=discovered.original_filename,
        relative_source_path=discovered.relative_source_path,
        task_category=discovered.task_category,
        raw_label=raw_label,
        normalized_label=norm_label,
        unverified_candidate_prefix=unverified_prefix,
        label_parse_pattern=parse_pattern,
        label_parse_warning=parse_warning,
        file_size_bytes=discovered.file_size_bytes,
        is_valid_edf=True,
        validation_error=None,
        edf_patient_id_raw=raw_header.local_patient_id,
        edf_recording_id_raw=raw_header.local_recording_id,
        edf_start_date=raw_header.start_date,
        edf_start_time=raw_header.start_time,
        sfreq_hz=sfreq,
        n_samples=n_samples,
        duration_sec=duration_sec,
        n_channels_total=len(mne_ch_names),
        n_candidate_eeg_channels=len(candidate_eeg_names),
        n_usable_eeg_channels=len(usable_eeg_names),
        n_non_eeg_channels=len(non_eeg_names),
        n_unknown_channels=len(unknown_names),
        n_flat_eeg_channels=len(flat_eeg_names),
        channel_names_json=json.dumps(mne_ch_names),
        candidate_eeg_channels_json=json.dumps(candidate_eeg_names),
        usable_eeg_channels_json=json.dumps(usable_eeg_names),
        non_eeg_channels_json=json.dumps(non_eeg_names),
        unknown_channels_json=json.dumps(unknown_names),
        flat_eeg_channels_json=json.dumps(flat_eeg_names),
    )

    return manifest_record, channel_records


def _build_audit_summary(
    discovery_report: DiscoveryReport,
    manifest_records: list[RecordingManifestRecord],
    channel_records: list[ChannelAuditRecord],
) -> dict[str, Any]:
    """Aggregate dataset-level statistics and quality warnings from audit records."""
    total_files = len(manifest_records)
    valid_records = [r for r in manifest_records if r.is_valid_edf]
    invalid_records = [r for r in manifest_records if not r.is_valid_edf]

    by_task: dict[str, dict[str, Any]] = {}
    for rec in manifest_records:
        task_bucket = by_task.setdefault(
            rec.task_category,
            {
                "n_total": 0,
                "n_valid_edf": 0,
                "n_invalid_edf": 0,
                "normalized_label_counts": {},
                "unverified_prefix_counts": {},
            },
        )
        task_bucket["n_total"] += 1
        if rec.is_valid_edf:
            task_bucket["n_valid_edf"] += 1
        else:
            task_bucket["n_invalid_edf"] += 1

        if rec.normalized_label is not None:
            lbl_counts = task_bucket["normalized_label_counts"]
            lbl_counts[rec.normalized_label] = lbl_counts.get(rec.normalized_label, 0) + 1

        if rec.unverified_candidate_prefix is not None:
            pfx_counts = task_bucket["unverified_prefix_counts"]
            pfx_counts[rec.unverified_candidate_prefix] = (
                pfx_counts.get(rec.unverified_candidate_prefix, 0) + 1
            )

    sfreqs = sorted({r.sfreq_hz for r in valid_records if r.sfreq_hz is not None})
    durations = [r.duration_sec for r in valid_records if r.duration_sec is not None]
    distinct_patient_headers = sorted(
        {r.edf_patient_id_raw for r in valid_records if r.edf_patient_id_raw is not None}
    )

    # Summarize unique channel roles across the dataset
    channel_role_summary: dict[str, list[str]] = {}
    for ch in channel_records:
        role_list = channel_role_summary.setdefault(ch.audited_channel_role, [])
        if ch.raw_channel_name not in role_list:
            role_list.append(ch.raw_channel_name)

    return {
        "data_root": str(discovery_report.data_root),
        "n_discovered_files": total_files,
        "n_valid_edf_files": len(valid_records),
        "n_invalid_edf_files": len(invalid_records),
        "invalid_files": [
            {
                "relative_source_path": r.relative_source_path,
                "validation_error": r.validation_error,
            }
            for r in invalid_records
        ],
        "missing_task_dirs": discovery_report.missing_task_dirs,
        "unexpected_root_entries": discovery_report.unexpected_root_entries,
        "ignored_non_edf_files": discovery_report.ignored_non_edf_files,
        "distinct_sampling_rates_hz": sfreqs,
        "duration_sec_stats": (
            {
                "min": float(np.min(durations)),
                "max": float(np.max(durations)),
                "mean": float(np.mean(durations)),
                "median": float(np.median(durations)),
            }
            if durations
            else None
        ),
        "distinct_edf_patient_id_headers": distinct_patient_headers,
        "participant_id_verification_note": (
            "Rule 12: Filename prefixes are stored in `unverified_candidate_prefix` and must "
            "NOT be treated as verified participant IDs until cross-checked against EDF "
            "`local_patient_id` headers and dataset documentation."
        ),
        "channel_roles_discovered": channel_role_summary,
        "tasks": by_task,
        "discovery_warnings": discovery_report.warnings,
    }


def run_dataset_audit(
    config: ProjectConfig,
    save_outputs: bool = True,
) -> DatasetAuditResult:
    """Execute Phase 1 dataset discovery and EDF metadata/channel audit.

    Parameters
    ----------
    config : ProjectConfig
        Validated project configuration.
    save_outputs : bool
        If ``True``, writes ``manifest.csv``, ``channel_audit.csv``, and
        ``audit_summary.json`` into ``config.output_dir``.

    Returns
    -------
    DatasetAuditResult
        In-memory manifest records, channel audit records, and summary dictionary.
    """
    discovery_report = discover_dataset(
        data_root=config.data_root,
        discovery_config=config.discovery,
        label_config=config.label_parsing,
    )

    manifest_records: list[RecordingManifestRecord] = []
    all_channel_records: list[ChannelAuditRecord] = []

    for discovered in discovery_report.recordings:
        m_rec, c_recs = audit_single_edf(discovered, config=config.edf_audit)
        manifest_records.append(m_rec)
        all_channel_records.extend(c_recs)

    summary = _build_audit_summary(
        discovery_report=discovery_report,
        manifest_records=manifest_records,
        channel_records=all_channel_records,
    )

    result = DatasetAuditResult(
        manifest_records=manifest_records,
        channel_records=all_channel_records,
        summary=summary,
    )

    if save_outputs:
        out_dir = Path(config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        manifest_df = pd.DataFrame([asdict(r) for r in manifest_records])
        manifest_path = out_dir / "manifest.csv"
        manifest_df.to_csv(manifest_path, index=False)

        channel_df = pd.DataFrame([asdict(c) for c in all_channel_records])
        channel_path = out_dir / "channel_audit.csv"
        channel_df.to_csv(channel_path, index=False)

        summary_path = out_dir / "audit_summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=False)

        logger.info(
            "Saved Phase 1 audit artifacts to %s (manifest=%d rows, channels=%d rows).",
            out_dir,
            len(manifest_records),
            len(all_channel_records),
        )

    return result
