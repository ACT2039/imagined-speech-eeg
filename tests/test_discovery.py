"""Unit tests for dataset directory discovery across Char, Digit, and Image folders."""

from __future__ import annotations

from pathlib import Path

import pytest

from imagined_speech.config import DiscoveryConfig
from imagined_speech.data.discovery import discover_dataset
from imagined_speech.exceptions import DatasetDiscoveryError


def test_discover_dataset_across_three_tasks(tmp_path: Path) -> None:
    """discover_dataset finds EDF files across Char, Digit, and Image and records non-EDF files."""
    for task in ("Char", "Digit", "Image"):
        (tmp_path / task).mkdir()

    (tmp_path / "Char" / "Sub1_A.edf").write_bytes(b"dummy")
    (tmp_path / "Digit" / "Sub1_3.EDF").write_bytes(b"dummy")
    (tmp_path / "Image" / "Sub1_Apple.edf").write_bytes(b"dummy")
    (tmp_path / "Image" / "notes.txt").write_text("ignore me", encoding="utf-8")
    (tmp_path / "README_EXTRA.md").write_text("root file", encoding="utf-8")

    report = discover_dataset(tmp_path)

    assert len(report.recordings) == 3
    assert report.missing_task_dirs == []
    assert "Image/notes.txt" in report.ignored_non_edf_files
    assert "README_EXTRA.md" in report.unexpected_root_entries

    by_task = {rec.task_category: rec for rec in report.recordings}
    assert set(by_task.keys()) == {"Char", "Digit", "Image"}
    assert by_task["Char"].parsed_label is not None
    assert by_task["Char"].parsed_label.normalized_label == "a"
    assert by_task["Digit"].parsed_label is not None
    assert by_task["Digit"].parsed_label.normalized_label == "3"
    assert by_task["Image"].parsed_label is not None
    assert by_task["Image"].parsed_label.normalized_label == "apple"


def test_discover_dataset_missing_task_directory_raises(tmp_path: Path) -> None:
    """Missing expected task directory raises DatasetDiscoveryError by default."""
    (tmp_path / "Char").mkdir()
    (tmp_path / "Digit").mkdir()
    # "Image" is missing

    with pytest.raises(DatasetDiscoveryError, match="Image"):
        discover_dataset(tmp_path)

    # Succeeds when allow_missing_task_dirs=True
    report = discover_dataset(
        tmp_path,
        discovery_config=DiscoveryConfig(allow_missing_task_dirs=True),
    )
    assert report.missing_task_dirs == ["Image"]
