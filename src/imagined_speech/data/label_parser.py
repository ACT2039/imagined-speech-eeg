"""Filename and path label parsing with strict preservation of raw values (Rules 7, 11, 12).

Important (Rule 12):
Filename prefixes are extracted strictly as ``unverified_candidate_prefix``. They must NOT
be assumed to identify human participants until independently verified against EDF headers
or dataset documentation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from imagined_speech.config import LabelParsingConfig
from imagined_speech.exceptions import LabelParseError

_TRIAL_TOKEN_RE = re.compile(r"^(?:trial|run|rep|session|ses)\d+$|^\(\d+\)$", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedRecordingLabel:
    """Structured result of parsing a recording's path and filename (Rules 11, 12)."""

    original_filename: str
    relative_source_path: str
    task_category: str
    raw_label: str
    normalized_label: str
    unverified_candidate_prefix: str | None
    stem_tokens: tuple[str, ...]
    parse_pattern: str
    parse_warning: str | None = None


def normalize_label(raw_label: str, case_mode: str = "lower") -> str:
    """Normalize a raw label string consistently while preserving semantic content.

    Parameters
    ----------
    raw_label : str
        Original unnormalized label string.
    case_mode : str
        One of ``"lower"``, ``"upper"``, or ``"preserve_case"``.

    Returns
    -------
    str
        Normalized label string.

    Raises
    ------
    LabelParseError
        If the normalized label is empty.
    """
    cleaned = raw_label.strip()
    cleaned = re.sub(r"[\s\-]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        raise LabelParseError(f"Label became empty after normalization: {raw_label!r}")

    if case_mode == "lower":
        return cleaned.lower()
    if case_mode == "upper":
        return cleaned.upper()
    if case_mode == "preserve_case":
        return cleaned
    raise LabelParseError(f"Unsupported case_mode: {case_mode!r}")


def _split_stem(stem: str, delimiters: tuple[str, ...]) -> list[str]:
    """Split a filename stem by configured delimiters, removing empty tokens."""
    escaped = [re.escape(d) for d in delimiters if d]
    pattern = "|".join(escaped + [r"\s+"])
    tokens = [tok.strip() for tok in re.split(pattern, stem) if tok.strip()]
    return tokens


def parse_recording_label(
    file_path: Path | str,
    data_root: Path | str,
    task_category: str,
    config: LabelParsingConfig | None = None,
) -> ParsedRecordingLabel:
    """Parse task label and unverified prefix metadata from a recording file path.

    Parameters
    ----------
    file_path : Path | str
        Path to the recording file.
    data_root : Path | str
        Root directory of the raw dataset used to compute ``relative_source_path``.
    task_category : str
        Task directory name (e.g., ``"Char"``, ``"Digit"``, ``"Image"``).
    config : LabelParsingConfig | None
        Optional label parsing configuration. Defaults to ``LabelParsingConfig()``.

    Returns
    -------
    ParsedRecordingLabel
        Parsed and normalized label metadata alongside original raw strings.

    Raises
    ------
    LabelParseError
        If the filename stem is empty, relative path computation fails, or no label
        can be extracted.
    """
    cfg = config or LabelParsingConfig()
    path = Path(file_path)
    root = Path(data_root)

    if not task_category or not task_category.strip():
        raise LabelParseError("task_category must be a non-empty string.")

    original_filename = path.name
    if not original_filename:
        raise LabelParseError(f"File path has no filename component: {file_path!r}")

    try:
        rel_path = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        try:
            rel_path = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise LabelParseError(
                f"File {path} is not located inside data_root {root}."
            ) from exc

    stem = path.stem.strip()
    if not stem:
        raise LabelParseError(
            f"Filename stem is empty for file {original_filename!r} in {rel_path!r}."
        )

    tokens = _split_stem(stem, cfg.delimiters)
    if not tokens:
        raise LabelParseError(
            f"Could not extract any tokens from filename stem {stem!r} ({rel_path})."
        )

    warning: str | None = None
    unverified_prefix: str | None = None

    # Strip trailing copy/trial indicators like "(1)" or "trial1" if more than one token exists
    effective_tokens = list(tokens)
    if len(effective_tokens) >= 2 and _TRIAL_TOKEN_RE.match(effective_tokens[-1]):
        removed_trial = effective_tokens.pop()
        warning = (
            f"Trailing trial/copy token {removed_trial!r} stripped from stem {stem!r}; "
            "verify label assignment."
        )

    if len(effective_tokens) == 1:
        raw_label = effective_tokens[0]
        parse_pattern = "single_token"
    elif len(effective_tokens) == 2:
        if cfg.extract_candidate_prefix:
            unverified_prefix = effective_tokens[0]
        raw_label = effective_tokens[1]
        parse_pattern = "prefix_and_label"
    else:
        # 3 or more tokens: check if the second token repeats the task category name
        # e.g., "<prefix>_<task>_<label>" such as "Sub01_Char_A"
        if cfg.extract_candidate_prefix:
            unverified_prefix = effective_tokens[0]

        if effective_tokens[1].lower() == task_category.strip().lower():
            raw_label = "_".join(effective_tokens[2:])
            parse_pattern = "prefix_task_label"
        else:
            raw_label = "_".join(effective_tokens[1:])
            parse_pattern = "prefix_multitoken_label"
            extra_warn = (
                f"Multi-token stem {stem!r} parsed as unverified_prefix={unverified_prefix!r}, "
                f"raw_label={raw_label!r}. Verify multi-token convention."
            )
            warning = f"{warning}; {extra_warn}" if warning else extra_warn

    normalized = normalize_label(raw_label, case_mode=cfg.case_mode)

    return ParsedRecordingLabel(
        original_filename=original_filename,
        relative_source_path=rel_path,
        task_category=task_category.strip(),
        raw_label=raw_label,
        normalized_label=normalized,
        unverified_candidate_prefix=unverified_prefix,
        stem_tokens=tuple(tokens),
        parse_pattern=parse_pattern,
        parse_warning=warning,
    )
