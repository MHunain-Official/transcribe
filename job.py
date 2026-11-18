#!/usr/bin/env python3
"""Single-file preprocess + transcribe job used by run.py and batch.py."""

from __future__ import annotations

import shutil
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from compare import write_comparison
from preprocess import (
    DEFAULT_KEEP_SILENCE_S,
    DEFAULT_MIN_PAUSE_S,
    DEFAULT_MODE,
    DEFAULT_THRESHOLD_DB,
    OUT_DIR,
    format_seconds,
    preprocess,
    probe_duration,
    resolve_tool,
)
from transcribe import TranscribeError, transcribe


def stem_of(path: Path) -> str:
    return path.stem if path.suffix else path.name


def duration_of(path: Path) -> float | None:
    ffmpeg = resolve_tool("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    return probe_duration(path, ffmpeg, ffprobe)


@dataclass
class JobResult:
    input: str
    stem: str
    status: str  # ok | skipped | error | timeout
    job_dir: str | None = None
    error: str | None = None
    error_type: str | None = None
    traceback: str | None = None
    before_duration_s: float | None = None
    after_duration_s: float | None = None
    saved_pct: float | None = None
    preprocess_before_s: float | None = None
    preprocess_after_s: float | None = None
    api_before_s: float | None = None
    api_after_s: float | None = None
    api_wall_s: float | None = None
    total_wall_s: float | None = None
    attempts: int = 1
    started_at: str | None = None
    finished_at: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def process_file(
    input_path: Path,
    *,
    mode: str = DEFAULT_MODE,
    threshold_db: float = DEFAULT_THRESHOLD_DB,
    min_pause_s: float = DEFAULT_MIN_PAUSE_S,
    keep_silence_s: float = DEFAULT_KEEP_SILENCE_S,
    locale: str | None = None,
    skip_transcribe: bool = False,
    skip_before: bool = False,
    parallel_api: bool = True,
    api_retries: int = 3,
    api_timeout_s: float = 300.0,
) -> JobResult:
    """Run one recording end-to-end. Raises on failure (caller may retry/skip)."""
    input_path = Path(input_path)
    stem = stem_of(input_path)
    started = _utc_now()
    t_job0 = time.perf_counter()
    job_dir = OUT_DIR / stem
    before_dir = job_dir / "before"
    after_dir = job_dir / "after"
    before_dir.mkdir(parents=True, exist_ok=True)
    after_dir.mkdir(parents=True, exist_ok=True)

    before_wav = before_dir / f"{stem}.wav"
    after_wav = after_dir / f"{stem}.clean.wav"
    before_dur: float | None = None
    after_dur: float | None = None
    before_txt: Path | None = None
    after_txt: Path | None = None
    preprocess_before_s: float | None = None
    preprocess_after_s: float | None = None
    api_before_s: float | None = None
    api_after_s: float | None = None
    api_wall_s: float | None = None

    if not input_path.is_file():
        raise FileNotFoundError(f"Input not found: {input_path}")

    if not skip_before:
        print(f"[{stem}] preprocess before (convert only)")
        t0 = time.perf_counter()
        preprocess(
            input_path,
            before_wav,
            mode="off",
            threshold_db=threshold_db,
            min_pause_s=min_pause_s,
            keep_silence_s=keep_silence_s,
        )
        preprocess_before_s = time.perf_counter() - t0
        before_dur = duration_of(before_wav)
    else:
        print(f"[{stem}] skip before/")

    print(f"[{stem}] preprocess after ({mode})")
    t0 = time.perf_counter()
    preprocess(
        input_path,
        after_wav,
        mode=mode,
        threshold_db=threshold_db,
        min_pause_s=min_pause_s,
        keep_silence_s=keep_silence_s,
    )
    preprocess_after_s = time.perf_counter() - t0
    after_dur = duration_of(after_wav)
    if before_dur is None:
        before_dur = duration_of(input_path)

    saved_pct = None
    if before_dur is not None and after_dur is not None and before_dur > 0:
        saved_pct = ((before_dur - after_dur) / before_dur) * 100
        print(
            f"[{stem}] duration {format_seconds(before_dur)} → "
            f"{format_seconds(after_dur)} ({saved_pct:.1f}% shorter)"
        )

    timing_fields = dict(
        preprocess_before_s=preprocess_before_s,
        preprocess_after_s=preprocess_after_s,
        api_before_s=api_before_s,
        api_after_s=api_after_s,
        api_wall_s=api_wall_s,
        total_wall_s=time.perf_counter() - t_job0,
    )

    if skip_transcribe:
        return JobResult(
            input=str(input_path),
            stem=stem,
            status="ok",
            job_dir=str(job_dir),
            before_duration_s=before_dur,
            after_duration_s=after_dur,
            saved_pct=saved_pct,
            started_at=started,
            finished_at=_utc_now(),
            extras={"skip_transcribe": True},
            **timing_fields,
        )

    def _tx(wav: Path, out: Path, label: str) -> tuple[str, Path, float]:
        print(f"[{stem}] transcribe {label}")
        t_api = time.perf_counter()
        txt, _ = transcribe(
            wav,
            out,
            locale=locale,
            retries=api_retries,
            timeout_s=api_timeout_s,
        )
        return label, txt, time.perf_counter() - t_api

    if not skip_before and parallel_api:
        t_api0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix=f"api-{stem}") as pool:
            fut_before = pool.submit(_tx, before_wav, before_dir, "before")
            fut_after = pool.submit(_tx, after_wav, after_dir, "after")
            errors: list[BaseException] = []
            for fut in as_completed([fut_before, fut_after]):
                try:
                    label, path, elapsed = fut.result()
                    if label == "before":
                        before_txt = path
                        api_before_s = elapsed
                    else:
                        after_txt = path
                        api_after_s = elapsed
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)
            if errors:
                raise errors[0]
        api_wall_s = time.perf_counter() - t_api0
    else:
        t_api0 = time.perf_counter()
        if not skip_before:
            _, before_txt, api_before_s = _tx(before_wav, before_dir, "before")
        _, after_txt, api_after_s = _tx(after_wav, after_dir, "after")
        api_wall_s = time.perf_counter() - t_api0

    if before_txt and after_txt and before_txt.exists() and after_txt.exists():
        comparison = write_comparison(
            before_txt,
            after_txt,
            job_dir / "comparison.md",
            stem=stem,
            before_duration_s=before_dur,
            after_duration_s=after_dur,
            mode=mode,
        )
        print(f"[{stem}] comparison → {comparison}")

    total_wall_s = time.perf_counter() - t_job0
    print(
        f"[{stem}] timing preprocess="
        f"{(preprocess_before_s or 0) + (preprocess_after_s or 0):.1f}s "
        f"api_wall={api_wall_s or 0:.1f}s total={total_wall_s:.1f}s"
    )

    return JobResult(
        input=str(input_path),
        stem=stem,
        status="ok",
        job_dir=str(job_dir),
        before_duration_s=before_dur,
        after_duration_s=after_dur,
        saved_pct=saved_pct,
        preprocess_before_s=preprocess_before_s,
        preprocess_after_s=preprocess_after_s,
        api_before_s=api_before_s,
        api_after_s=api_after_s,
        api_wall_s=api_wall_s,
        total_wall_s=total_wall_s,
        started_at=started,
        finished_at=_utc_now(),
    )


def run_file_safe(
    input_path: Path,
    *,
    attempts: int = 1,
    **kwargs: Any,
) -> JobResult:
    """Like process_file, but catches failures and returns JobResult(status=error)."""
    input_path = Path(input_path)
    stem = stem_of(input_path)
    started = _utc_now()
    last_exc: BaseException | None = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            result = process_file(input_path, **kwargs)
            result.attempts = attempt
            result.started_at = started
            result.finished_at = _utc_now()
            return result
        except (SystemExit, KeyboardInterrupt):
            raise
        except BaseException as exc:  # noqa: BLE001 — convert to JobResult for batch
            last_exc = exc
            print(f"[{stem}] attempt {attempt}/{attempts} failed: {exc}")
            if attempt < attempts:
                continue

    assert last_exc is not None
    return JobResult(
        input=str(input_path),
        stem=stem,
        status="error",
        error=str(last_exc),
        error_type=type(last_exc).__name__,
        traceback="".join(
            traceback.format_exception(type(last_exc), last_exc, last_exc.__traceback__)
        ),
        attempts=attempts,
        started_at=started,
        finished_at=_utc_now(),
        extras={"note": "skipped after failures; batch continued"},
    )


# Re-export for callers that want typed API errors
__all__ = [
    "JobResult",
    "TranscribeError",
    "process_file",
    "run_file_safe",
    "stem_of",
    "duration_of",
    "OUT_DIR",
    "DEFAULT_MODE",
    "DEFAULT_THRESHOLD_DB",
    "DEFAULT_MIN_PAUSE_S",
    "DEFAULT_KEEP_SILENCE_S",
]
