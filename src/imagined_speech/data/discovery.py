"""Dataset directory discovery for imagined speech EDF recordings (Rules 7, 11)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from imagined_speech.config import DiscoveryConfig, LabelParsingConfig
from imagined_speech.data.label_parser import ParsedRecordingLabel, parse_recording_label
from imagined_speech.exceptions import DatasetDiscoveryError, LabelParseError

logger = logging.getLogger("imagined_speech.data.discovery")


@dataclass(frozen=True)
class DiscoveredRecording:
    """Metadata for a single discovered recording file before EDF header inspection."""

    absolute_path: Path
    original_filename: str
    relative_source_path: str
    task_category: str
    file_size_bytes: int
    parsed_label: ParsedRecordingLabel | None
    label_parse_error: str | None = None


@dataclass
class DiscoveryReport:
    """Summary of dataset directory discovery across all task folders."""

    data_root: Path
    recordings: list[DiscoveredRecording] = field(default_factory=list)
    missing_task_dirs: list[str] = field(default_factory=list)
    unexpected_root_entries: list[str] = field(default_factory=list)
    ignored_non_edf_files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def discover_dataset(
    data_root: Path | str,
    discovery_config: DiscoveryConfig | None = None,
    label_config: LabelParsingConfig | None = None,
) -> DiscoveryReport:
    """Scan the dataset root directory for expected task folders and EDF recordings.

    Parameters
    ----------
    data_root : Path | str
        Root directory containing task folders (e.g., ``Char``, ``Digit``, ``Image``).
    discovery_config : DiscoveryConfig | None
        Optional discovery configuration. Defaults to ``DiscoveryConfig()``.
    label_config : LabelParsingConfig | None
        Optional label parsing configuration. Defaults to ``LabelParsingConfig()``.

    Returns
    -------
    DiscoveryReport
        Structured discovery results including all candidate recordings and warnings.

    Raises
    ------
    DatasetDiscoveryError
        If ``data_root`` does not exist, is not a directory, or is missing required
        task subdirectories when ``allow_missing_task_dirs=False``.
    """
    d_cfg = discovery_config or DiscoveryConfig()
    l_cfg = label_config or LabelParsingConfig()
    root = Path(data_root)

    if not root.exists():
        raise DatasetDiscoveryError(f"Dataset root path does not exist: {root}")
    if not root.is_dir():
        raise DatasetDiscoveryError(f"Dataset root path is not a directory: {root}")

    report = DiscoveryReport(data_root=root)

    # Map existing root entries case-insensitively to handle filesystem variations
    existing_dirs_by_lower: dict[str, Path] = {}
    expected_lower_map = {t.lower(): t for t in d_cfg.expected_tasks}

    for entry in sorted(root.iterdir(), key=lambda p: p.name):
        if entry.is_dir():
            lower_name = entry.name.lower()
            if lower_name in expected_lower_map:
                existing_dirs_by_lower[lower_name] = entry
                canonical_name = expected_lower_map[lower_name]
                if entry.name != canonical_name:
                    msg = (
                        f"Task directory {entry.name!r} differs in casing from canonical "
                        f"name {canonical_name!r}."
                    )
                    logger.warning(msg)
                    report.warnings.append(msg)
            else:
                report.unexpected_root_entries.append(entry.name)
        else:
            report.unexpected_root_entries.append(entry.name)

    for canonical_task in d_cfg.expected_tasks:
        matched_dir = existing_dirs_by_lower.get(canonical_task.lower())
        if matched_dir is None:
            report.missing_task_dirs.append(canonical_task)

    if report.missing_task_dirs and not d_cfg.allow_missing_task_dirs:
        raise DatasetDiscoveryError(
            f"Missing required task directories in {root}: {report.missing_task_dirs}. "
            f"Expected tasks: {list(d_cfg.expected_tasks)}."
        )

    allowed_exts = {ext.lower() for ext in d_cfg.allowed_extensions}

    for canonical_task in d_cfg.expected_tasks:
        task_dir = existing_dirs_by_lower.get(canonical_task.lower())
        if task_dir is None:
            continue

        iterator = (
            task_dir.rglob("*") if d_cfg.recursive_within_task else task_dir.iterdir()
        )
        all_files = sorted(
            (p for p in iterator if p.is_file()),
            key=lambda p: p.as_posix().lower(),
        )

        for file_path in all_files:
            rel_posix = file_path.relative_to(root).as_posix()
            if file_path.suffix.lower() not in allowed_exts:
                report.ignored_non_edf_files.append(rel_posix)
                continue

            file_size = file_path.stat().st_size
            parsed: ParsedRecordingLabel | None = None
            parse_err: str | None = None

            try:
                parsed = parse_recording_label(
                    file_path=file_path,
                    data_root=root,
                    task_category=canonical_task,
                    config=l_cfg,
                )
                if parsed.parse_warning:
                    report.warnings.append(f"{rel_posix}: {parsed.parse_warning}")
            except LabelParseError as exc:
                parse_err = str(exc)
                logger.warning("Label parse failure for %s: %s", rel_posix, parse_err)
                report.warnings.append(f"{rel_posix}: {parse_err}")

            report.recordings.append(
                DiscoveredRecording(
                    absolute_path=file_path,
                    original_filename=file_path.name,
                    relative_source_path=rel_posix,
                    task_category=canonical_task,
                    file_size_bytes=file_size,
                    parsed_label=parsed,
                    label_parse_error=parse_err,
                )
            )

    logger.info(
        "Discovered %d EDF recording(s) across %d task folder(s) in %s.",
        len(report.recordings),
        len(d_cfg.expected_tasks) - len(report.missing_task_dirs),
        root,
    )
    return report
