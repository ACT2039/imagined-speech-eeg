"""Command-line interface for Phase 1 dataset discovery and EDF metadata/channel audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from imagined_speech.config import load_config
from imagined_speech.data.edf_audit import run_dataset_audit
from imagined_speech.exceptions import ImaginedSpeechError
from imagined_speech.utils.logging_utils import setup_logging
from imagined_speech.utils.reproducibility import set_global_seed


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="imagined-speech-audit",
        description=(
            "Phase 1: Discover imagined-speech EDF recordings (Char, Digit, Image) "
            "and audit EDF headers, channel types, and signal properties."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to a YAML configuration file (e.g., configs/default_config.yaml).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root directory containing the Char, Digit, and Image subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where manifest.csv, channel_audit.csv, and audit_summary.json will be saved.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed override for reproducibility.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run the audit in-memory and print the summary without writing output files.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the Phase 1 CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = load_config(
            config_path=args.config,
            data_root=args.data_root,
            output_dir=args.output_dir,
            seed=args.seed,
            log_level=args.log_level,
        )
        logger = setup_logging(level=config.log_level)
        set_global_seed(config.seed)

        logger.info("Starting Phase 1 Dataset Audit on data_root=%s", config.data_root)
        result = run_dataset_audit(config=config, save_outputs=not args.no_save)
        print(json.dumps(result.summary, indent=2))
        return 0
    except ImaginedSpeechError as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
