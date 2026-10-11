"""Data discovery, label parsing, and EDF audit subpackage."""

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

__all__ = [
    "ChannelAuditRecord",
    "DatasetAuditResult",
    "DiscoveredRecording",
    "DiscoveryReport",
    "ParsedRecordingLabel",
    "RawEDFChannelHeader",
    "RawEDFFixedHeader",
    "RecordingManifestRecord",
    "audit_single_edf",
    "classify_channel",
    "clean_channel_name",
    "detect_counter_like_signal",
    "discover_dataset",
    "normalize_label",
    "parse_recording_label",
    "read_raw_edf_header",
    "run_dataset_audit",
]
