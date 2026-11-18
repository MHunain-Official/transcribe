#!/usr/bin/env python3
"""Parallel batch runner for many recordings.

- Processes multiple files concurrently
- Retries failed jobs
- Soft job timeout: if a file hangs, skip it, log what happened, keep going
- Writes out/batch_errors.jsonl + out/batch_summary.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from job import (
    DEFAULT_KEEP_SILENCE_S,
    DEFAULT_MIN_PAUSE_S,
    DEFAULT_MODE,
    DEFAULT_THRESHOLD_DB,
    JobResult,
    OUT_DIR,
    run_file_safe,
    stem_of,
)
from transcribe import DEFAULT_RETRIES, DEFAULT_TIMEOUT_S


AUDIO_EXTS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".ogg",
    ".webm",
    ".flac",
    ".aac",
    ".mp4",
    ".gsm",
    "",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def collect_inputs(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            for child in sorted(path.iterdir()):
                if child.is_file() and child.suffix.lower() in AUDIO_EXTS:
                    # extensionless voicenotes (146, 405, …) are included via ""
                    files.append(child)
                elif child.is_file() and not child.suffix:
                    files.append(child)
        elif path.is_file():
            files.append(path)
        else:
            print(f"WARN: not found, skipping: {path}", file=sys.stderr)
    # de-dupe preserve order
    seen: set[str] = set()
    out: list[Path] = []
    for f in files:
        key = str(f.resolve())
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(path: Path, results: list[JobResult], started: str, workers: int) -> None:
    ok = [r for r in results if r.status == "ok"]
    err = [r for r in results if r.status == "error"]
    hung = [r for r in results if r.status == "timeout"]
    walls = [r.total_wall_s for r in results if r.total_wall_s is not None]
    seq = sum(walls) if walls else None
    par = max(walls) if walls else None
    lines = [
        "# Batch summary",
        "",
        f"- Started: `{started}`",
        f"- Finished: `{_utc_now()}`",
        f"- Workers: `{workers}`",
        f"- Total: **{len(results)}**",
        f"- OK: **{len(ok)}**",
        f"- Errors (skipped): **{len(err)}**",
        f"- Timeouts (skipped): **{len(hung)}**",
        f"- Sum of job times (≈ sequential): **{seq:.1f}s**" if seq is not None else "",
        f"- Max job time (≈ parallel wall): **{par:.1f}s**" if par is not None else "",
        f"- Est. speedup: **{(seq / par):.2f}x**" if seq and par else "",
        "",
        "## Results",
        "",
        "| File | Status | Attempts | Job time | Saved % | Error |",
        "|---|---|---:|---:|---:|---|",
    ]
    for r in results:
        saved = f"{r.saved_pct:.1f}" if r.saved_pct is not None else ""
        job_t = f"{r.total_wall_s:.1f}s" if r.total_wall_s is not None else ""
        err_s = (r.error or "").replace("|", "\\|").replace("\n", " ")[:120]
        lines.append(
            f"| `{r.stem}` | {r.status} | {r.attempts} | {job_t} | {saved} | {err_s} |"
        )
    if err or hung:
        lines += ["", "## Failures detail", ""]
        for r in err + hung:
            lines.append(f"### `{r.stem}` — {r.status}")
            lines.append("")
            lines.append(f"- input: `{r.input}`")
            lines.append(f"- type: `{r.error_type}`")
            lines.append(f"- error: {r.error}")
            if r.traceback:
                lines.append("")
                lines.append("```")
                lines.append(r.traceback.rstrip())
                lines.append("```")
            lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_batch(
    inputs: list[Path],
    *,
    workers: int = 3,
    job_retries: int = 2,
    job_timeout_s: float = 900.0,
    mode: str = DEFAULT_MODE,
    locale: str | None = None,
    skip_before: bool = False,
    skip_transcribe: bool = False,
    parallel_api: bool = True,
    api_retries: int = DEFAULT_RETRIES,
    api_timeout_s: float = DEFAULT_TIMEOUT_S,
    threshold_db: float = DEFAULT_THRESHOLD_DB,
    min_pause_s: float = DEFAULT_MIN_PAUSE_S,
    keep_silence_s: float = DEFAULT_KEEP_SILENCE_S,
) -> list[JobResult]:
    started = _utc_now()
    errors_path = OUT_DIR / "batch_errors.jsonl"
    summary_path = OUT_DIR / "batch_summary.md"
    results: list[JobResult] = []
    # Fresh run marker
    append_jsonl(
        errors_path,
        {
            "event": "batch_start",
            "started_at": started,
            "files": [str(p) for p in inputs],
            "workers": workers,
            "job_retries": job_retries,
            "job_timeout_s": job_timeout_s,
        },
    )

    kwargs = dict(
        mode=mode,
        locale=locale,
        skip_before=skip_before,
        skip_transcribe=skip_transcribe,
        parallel_api=parallel_api,
        api_retries=api_retries,
        api_timeout_s=api_timeout_s,
        threshold_db=threshold_db,
        min_pause_s=min_pause_s,
        keep_silence_s=keep_silence_s,
        attempts=job_retries,
    )

    print(
        f"Batch: {len(inputs)} file(s), workers={workers}, "
        f"job_timeout={job_timeout_s:.0f}s, job_retries={job_retries}"
    )

    with ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="job") as pool:
        future_map: dict[Future[JobResult], Path] = {
            pool.submit(run_file_safe, path, **kwargs): path for path in inputs
        }
        pending = set(future_map.keys())
        # Poll with per-future deadlines so one hang cannot block the batch forever
        deadlines = {fut: time.monotonic() + job_timeout_s for fut in pending}

        while pending:
            # Wait briefly for any completion
            done_set = set()
            try:
                for fut in as_completed(pending, timeout=1.0):
                    done_set.add(fut)
                    break
            except TimeoutError:
                done_set = set()

            now = time.monotonic()
            timed_out = [fut for fut in pending if now > deadlines[fut] and fut not in done_set]

            for fut in list(done_set):
                path = future_map[fut]
                pending.discard(fut)
                try:
                    result = fut.result(timeout=0)
                except Exception as exc:  # noqa: BLE001
                    result = JobResult(
                        input=str(path),
                        stem=stem_of(path),
                        status="error",
                        error=str(exc),
                        error_type=type(exc).__name__,
                        started_at=started,
                        finished_at=_utc_now(),
                        extras={"note": "future raised; skipped"},
                    )
                results.append(result)
                status_icon = "OK" if result.status == "ok" else result.status.upper()
                print(f"[{status_icon}] {result.stem}")
                if result.status != "ok":
                    append_jsonl(errors_path, {"event": "job_failed", **result.to_dict()})

            for fut in timed_out:
                path = future_map[fut]
                pending.discard(fut)
                # Cannot forcibly kill a stuck thread; abandon waiting and keep batch alive.
                fut.cancel()
                result = JobResult(
                    input=str(path),
                    stem=stem_of(path),
                    status="timeout",
                    error=(
                        f"Job exceeded {job_timeout_s:.0f}s — skipped so batch can continue. "
                        "Underlying thread may still finish later; check out/ later."
                    ),
                    error_type="TimeoutError",
                    started_at=started,
                    finished_at=_utc_now(),
                    extras={
                        "note": "skipped due to hang/timeout; other files kept processing",
                        "job_timeout_s": job_timeout_s,
                    },
                )
                results.append(result)
                print(f"[TIMEOUT/SKIP] {result.stem}")
                append_jsonl(errors_path, {"event": "job_timeout", **result.to_dict()})

    # Stable order like inputs
    by_stem = {r.stem: r for r in results}
    ordered = [by_stem[stem_of(p)] for p in inputs if stem_of(p) in by_stem]
    # include any extras
    for r in results:
        if r not in ordered:
            ordered.append(r)

    write_summary(summary_path, ordered, started, workers)
    results_path = OUT_DIR / "batch_results.json"
    results_path.write_text(
        json.dumps(
            {
                "started_at": started,
                "finished_at": _utc_now(),
                "workers": workers,
                "results": [r.to_dict() for r in ordered],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    append_jsonl(
        errors_path,
        {
            "event": "batch_end",
            "finished_at": _utc_now(),
            "ok": sum(1 for r in ordered if r.status == "ok"),
            "error": sum(1 for r in ordered if r.status == "error"),
            "timeout": sum(1 for r in ordered if r.status == "timeout"),
            "summary": str(summary_path),
            "results": str(results_path),
        },
    )
    print(f"\nSummary: {summary_path}")
    print(f"Results: {results_path}")
    print(f"Errors:  {errors_path}")
    return ordered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Parallel batch: preprocess + MAI transcribe many recordings. "
            "Retries failures; on hang/error skips that file and keeps going."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Audio files and/or folders (e.g. ../voicenotes)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Parallel file workers (default: 3)",
    )
    parser.add_argument(
        "--job-retries",
        type=int,
        default=2,
        help="Full-job retries after error (default: 2)",
    )
    parser.add_argument(
        "--job-timeout",
        type=float,
        default=900.0,
        help="Per-file wall timeout seconds; then skip & continue (default: 900)",
    )
    parser.add_argument(
        "--api-retries",
        type=int,
        default=DEFAULT_RETRIES,
        help=f"API call retries (default: {DEFAULT_RETRIES})",
    )
    parser.add_argument(
        "--api-timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help=f"API read timeout seconds (default: {DEFAULT_TIMEOUT_S})",
    )
    parser.add_argument(
        "--mode",
        choices=["smart", "off", "edges", "long"],
        default=DEFAULT_MODE,
    )
    parser.add_argument("--locale", default=None, help="ur (default), en, hi, auto")
    parser.add_argument("--skip-before", action="store_true")
    parser.add_argument("--skip-transcribe", action="store_true")
    parser.add_argument(
        "--no-parallel-api",
        action="store_true",
        help="Do not run before/after API calls in parallel within a file",
    )
    parser.add_argument("--threshold-db", type=float, default=DEFAULT_THRESHOLD_DB)
    parser.add_argument("--min-pause", type=float, default=DEFAULT_MIN_PAUSE_S)
    parser.add_argument("--keep-silence", type=float, default=DEFAULT_KEEP_SILENCE_S)
    args = parser.parse_args(argv)

    inputs = collect_inputs(args.inputs)
    if not inputs:
        print("No input files found.", file=sys.stderr)
        return 2

    results = run_batch(
        inputs,
        workers=args.workers,
        job_retries=args.job_retries,
        job_timeout_s=args.job_timeout,
        mode=args.mode,
        locale=args.locale,
        skip_before=args.skip_before,
        skip_transcribe=args.skip_transcribe,
        parallel_api=not args.no_parallel_api,
        api_retries=args.api_retries,
        api_timeout_s=args.api_timeout,
        threshold_db=args.threshold_db,
        min_pause_s=args.min_pause,
        keep_silence_s=args.keep_silence,
    )
    failed = sum(1 for r in results if r.status != "ok")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
