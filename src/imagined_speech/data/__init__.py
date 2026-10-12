"""Data discovery, label parsing, EDF audit, and signal-quality subpackage."""

from imagined_speech.data.discovery import DiscoveredRecording, DiscoveryReport, discover_dataset
from imagined_speech.data.edf_audit import (
    ChannelAuditRecord,
    DatasetAuditResult,
    RawEDFChannelHeader,
    RawEDFFixedHeader,
    RecordingManifestRecord,
    audit_single_edf,
    classify_channel,
    clean_channel_name,
    detect_counter_like_signal,
    read_raw_edf_header,
    run_dataset_audit,
)
from imagined_speech.data.label_parser import (
    ParsedRecordingLabel,
    normalize_label,
    parse_recording_label,
)
from imagined_speech.data.signal_quality import (
    ChannelSignalStatistics,
    InspectionErrorRecord,
    RecordingQualityInspection,
    SelectedRecordingInput,
    SignalQualityAnalysisResult,
    compute_channel_signal_statistics,
    inspect_recording_signal_quality,
    load_and_validate_manifest,
    run_signal_quality_analysis,
    select_representative_recordings,
)

__all__ = [
    "ChannelAuditRecord",
    "ChannelSignalStatistics",
    "DatasetAuditResult",
    "DiscoveredRecording",
    "DiscoveryReport",
    "InspectionErrorRecord",
    "ParsedRecordingLabel",
    "RawEDFChannelHeader",
    "RawEDFFixedHeader",
    "RecordingManifestRecord",
    "RecordingQualityInspection",
    "SelectedRecordingInput",
    "SignalQualityAnalysisResult",
    "audit_single_edf",
    "classify_channel",
    "clean_channel_name",
    "compute_channel_signal_statistics",
    "detect_counter_like_signal",
    "discover_dataset",
    "inspect_recording_signal_quality",
    "load_and_validate_manifest",
    "normalize_label",
    "parse_recording_label",
    "read_raw_edf_header",
    "run_dataset_audit",
    "run_signal_quality_analysis",
    "select_representative_recordings",
]
