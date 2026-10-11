"""Unit tests for filename and path label parsing and normalization (Rules 11, 12)."""

from __future__ import annotations

from pathlib import Path

import pytest

from imagined_speech.config import LabelParsingConfig
from imagined_speech.data.label_parser import normalize_label, parse_recording_label
from imagined_speech.exceptions import LabelParseError


def test_parse_prefix_and_label_preserves_raw_values(tmp_path: Path) -> None:
    """Standard <prefix>_<label>.edf preserves original filename, path, raw label, and unverified prefix."""
    char_dir = tmp_path / "Char"
    char_dir.mkdir()
    file_path = char_dir / "Alan_A.edf"
    file_path.touch()

    parsed = parse_recording_label(
        file_path=file_path,
        data_root=tmp_path,
        task_category="Char",
    )

    assert parsed.original_filename == "Alan_A.edf"
    assert parsed.relative_source_path == "Char/Alan_A.edf"
    assert parsed.task_category == "Char"
    assert parsed.raw_label == "A"
    assert parsed.normalized_label == "a"
    assert parsed.unverified_candidate_prefix == "Alan"
    assert parsed.parse_pattern == "prefix_and_label"
    assert parsed.parse_warning is None


def test_parse_single_token_filename(tmp_path: Path) -> None:
    """Single-token filename <label>.edf sets unverified_candidate_prefix to None."""
    digit_dir = tmp_path / "Digit"
    digit_dir.mkdir()
    file_path = digit_dir / "7.edf"
    file_path.touch()

    parsed = parse_recording_label(
        file_path=file_path,
        data_root=tmp_path,
        task_category="Digit",
    )

    assert parsed.raw_label == "7"
    assert parsed.normalized_label == "7"
    assert parsed.unverified_candidate_prefix is None
    assert parsed.parse_pattern == "single_token"


def test_parse_prefix_task_label_and_multitoken(tmp_path: Path) -> None:
    """Multi-token stems handle repeated task names and flag genuine multi-token labels."""
    image_dir = tmp_path / "Image"
    image_dir.mkdir()

    # Case 1: <prefix>_<task>_<label>
    f1 = image_dir / "Sub01_Image_Apple.edf"
    f1.touch()
    p1 = parse_recording_label(f1, tmp_path, "Image")
    assert p1.unverified_candidate_prefix == "Sub01"
    assert p1.raw_label == "Apple"
    assert p1.normalized_label == "apple"
    assert p1.parse_pattern == "prefix_task_label"

    # Case 2: <prefix>_<multi_word_label> emits a verification warning
    f2 = image_dir / "Sub01_Mobile_Phone.edf"
    f2.touch()
    p2 = parse_recording_label(f2, tmp_path, "Image")
    assert p2.unverified_candidate_prefix == "Sub01"
    assert p2.raw_label == "Mobile_Phone"
    assert p2.normalized_label == "mobile_phone"
    assert p2.parse_pattern == "prefix_multitoken_label"
    assert p2.parse_warning is not None


def test_normalize_label_modes_and_errors() -> None:
    """normalize_label supports lower, upper, preserve_case and rejects empty strings."""
    assert normalize_label("  Apple - Pie ", case_mode="lower") == "apple_pie"
    assert normalize_label("  Apple - Pie ", case_mode="upper") == "APPLE_PIE"
    assert normalize_label("  Apple - Pie ", case_mode="preserve_case") == "Apple_Pie"

    with pytest.raises(LabelParseError):
        normalize_label("   ___   ")


def test_parse_recording_label_outside_root_raises(tmp_path: Path) -> None:
    """File outside data_root raises LabelParseError."""
    other_file = tmp_path / "outside.edf"
    other_file.touch()
    sub_root = tmp_path / "dataset"
    sub_root.mkdir()

    with pytest.raises(LabelParseError):
        parse_recording_label(
            file_path=other_file,
            data_root=sub_root,
            task_category="Char",
            config=LabelParsingConfig(),
        )
