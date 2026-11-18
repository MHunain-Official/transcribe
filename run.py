#!/usr/bin/env python3
"""Preprocess audio then transcribe with MAI-Transcribe-2.

Writes a per-file folder with before/ (convert only) and after/ (chosen mode),
plus comparison.md so you can spot missing content.

For many files in parallel with retries / skip-on-error, use batch.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from job import (
    DEFAULT_KEEP_SILENCE_S,
    DEFAULT_MIN_PAUSE_S,
    DEFAULT_MODE,
    DEFAULT_THRESHOLD_DB,
    process_file,
)
from transcribe import DEFAULT_RETRIES, DEFAULT_TIMEOUT_S, TranscribeError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Cost + accuracy: compress long dead-air, keep speech, "
            "then transcribe with MAI-Transcribe-2. Default mode=smart. "
            "Outputs out/<name>/{before,after}/ plus comparison.md."
        )
    )
    parser.add_argument("input", type=Path, help="Path to input audio file")
    parser.add_argument(
        "--skip-transcribe",
        action="store_true",
        help="Only preprocess; do not call the API",
    )
    parser.add_argument(
        "--skip-before",
        action="store_true",
        help="Skip before/ baseline (convert-only + transcribe) to save one API call",
    )
    parser.add_argument(
        "--mode",
        choices=["smart", "off", "edges", "long"],
        default=DEFAULT_MODE,
        help=(
            "smart=compress dead-air keep speech (default), "
            "off=no cut, edges=trim ends only, long=legacy"
        ),
    )
    parser.add_argument(
        "--threshold-db",
        type=float,
        default=DEFAULT_THRESHOLD_DB,
        help=f"Silence threshold in dB (default: {DEFAULT_THRESHOLD_DB})",
    )
    parser.add_argument(
        "--min-pause",
        type=float,
        default=DEFAULT_MIN_PAUSE_S,
        help=f"Only compress dead-air pauses >= seconds (default: {DEFAULT_MIN_PAUSE_S})",
    )
    parser.add_argument(
        "--keep-silence",
        type=float,
        default=DEFAULT_KEEP_SILENCE_S,
        help=f"Seconds of pause left after compressing a long gap (default: {DEFAULT_KEEP_SILENCE_S})",
    )
    parser.add_argument(
        "--locale",
        default=None,
        help=(
            "MAI language: ur (default, Urdu script), en, hi, or auto. "
            "Use ur for Eng+Urdu so speech is not written as Hindi."
        ),
    )
    parser.add_argument(
        "--api-retries",
        type=int,
        default=DEFAULT_RETRIES,
        help=f"API retries on timeout/5xx/429 (default: {DEFAULT_RETRIES})",
    )
    parser.add_argument(
        "--api-timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help=f"API read timeout seconds (default: {DEFAULT_TIMEOUT_S})",
    )
    parser.add_argument(
        "--no-parallel-api",
        action="store_true",
        help="Run before/after API calls sequentially",
    )
    args = parser.parse_args(argv)

    try:
        result = process_file(
            args.input,
            mode=args.mode,
            threshold_db=args.threshold_db,
            min_pause_s=args.min_pause,
            keep_silence_s=args.keep_silence,
            locale=args.locale,
            skip_transcribe=args.skip_transcribe,
            skip_before=args.skip_before,
            parallel_api=not args.no_parallel_api,
            api_retries=args.api_retries,
            api_timeout_s=args.api_timeout,
        )
    except (TranscribeError, FileNotFoundError, SystemExit) as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"\nJob folder: {result.job_dir}")
    if result.job_dir:
        print(f"  before/ → {Path(result.job_dir) / 'before'}")
        print(f"  after/  → {Path(result.job_dir) / 'after'}")
        if (Path(result.job_dir) / "comparison.md").exists():
            print("  comparison.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
