"""Command-line interface for Phase 1 dataset audit and Phase 2 signal-quality analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from imagined_speech.config import load_config
from imagined_speech.data.edf_audit import run_dataset_audit
from imagined_speech.data.signal_quality import run_signal_quality_analysis
from imagined_speech.exceptions import ImaginedSpeechError
from imagined_speech.utils.logging_utils import setup_logging
from imagined_speech.utils.reproducibility import set_global_seed


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="imagined-speech-audit",
        description=(
            "Discover imagined-speech EDF recordings (Char, Digit, Image), "
            "audit EDF metadata/channels (Phase 1), and analyze EEG signal quality (Phase 2)."
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="audit",
        choices=["audit", "signal-quality", "both"],
        help="Execution mode: 'audit' (Phase 1), 'signal-quality' (Phase 2), or 'both'.",
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
        help="Directory where manifest, signal statistics, reports, and figures will be saved.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional path to an existing manifest.csv for 'signal-quality' mode.",
    )
    parser.add_argument(
        "--recordings-per-task",
        type=int,
        default=None,
        help="Number of representative recordings per task for signal-quality analysis.",
    )
    parser.add_argument(
        "--recording",
        action="append",
        default=None,
        help="Specific recording relative path or filename to inspect (may be passed multiple times).",
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
        help="Run in-memory and print the JSON summary without writing output files.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the Phase 1 & Phase 2 CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = load_config(
            config_path=args.config,
            data_root=args.data_root,
            output_dir=args.output_dir,
            seed=args.seed,
            log_level=args.log_level,
            recordings_per_task=args.recordings_per_task,
            specific_recordings=args.recording,
        )
        logger = setup_logging(level=config.log_level)
        set_global_seed(config.seed)

        save_outputs = not args.no_save
        if args.mode == "audit":
            logger.info("Starting Phase 1 Dataset Audit on data_root=%s", config.data_root)
            audit_res = run_dataset_audit(config=config, save_outputs=save_outputs)
            print(json.dumps(audit_res.summary, indent=2))
        elif args.mode == "signal-quality":
            logger.info("Starting Phase 2 Signal Quality Analysis on data_root=%s", config.data_root)
            sq_res = run_signal_quality_analysis(
                config=config,
                manifest_source=args.manifest,
                output_dir=config.output_dir,
                save_outputs=save_outputs,
            )
            print(json.dumps(sq_res.summary, indent=2))
        else:
            logger.info("Running Phase 1 Audit + Phase 2 Signal Quality Analysis")
            audit_res = run_dataset_audit(config=config, save_outputs=save_outputs)
            sq_res = run_signal_quality_analysis(
                config=config,
                manifest_source=args.manifest,
                output_dir=config.output_dir,
                save_outputs=save_outputs,
            )
            print(
                json.dumps(
                    {
                        "phase1_audit_summary": audit_res.summary,
                        "phase2_signal_quality_summary": sq_res.summary,
                    },
                    indent=2,
                )
            )
        return 0
    except ImaginedSpeechError as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
