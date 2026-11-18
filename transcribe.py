#!/usr/bin/env python3
"""Transcribe cleaned audio with Azure Speech MAI-Transcribe-2."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv


OUT_DIR = Path(__file__).resolve().parent / "out"
API_VERSION = "2025-10-15"
MODEL = "MAI-Transcribe-2"
# MAI auto-detect often maps Urdu speech → Hindi (Devanagari). Prefer Urdu script.
DEFAULT_LOCALE = "ur"
DEFAULT_RETRIES = 3
DEFAULT_TIMEOUT_S = 300.0
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}

LOCALE_ALIASES = {
    "auto": None,
    "ur": "ur",
    "urdu": "ur",
    "ur-pk": "ur",
    "ur_pk": "ur",
    "ur-in": "ur",
    "hi": "hi",
    "hindi": "hi",
    "en": "en",
    "english": "en",
}


class TranscribeError(RuntimeError):
    """Raised when Azure transcription fails (after retries if any)."""


def normalize_locale(value: str | None) -> str | None:
    if value is None:
        return DEFAULT_LOCALE
    raw = value.strip().lower()
    if not raw:
        return DEFAULT_LOCALE
    if raw in LOCALE_ALIASES:
        return LOCALE_ALIASES[raw]
    return raw


def load_config() -> tuple[str, str, str | None]:
    load_dotenv(Path(__file__).resolve().parent / ".env")
    key = os.getenv("AZURE_SPEECH_KEY", "").strip()
    endpoint = os.getenv("AZURE_SPEECH_ENDPOINT", "").strip().rstrip("/")
    locale = normalize_locale(os.getenv("AZURE_SPEECH_LOCALE", DEFAULT_LOCALE))
    if not key or not endpoint:
        raise TranscribeError(
            "Missing AZURE_SPEECH_KEY or AZURE_SPEECH_ENDPOINT in .env"
        )
    return key, endpoint, locale


def extract_transcript(payload: dict) -> str:
    """Pull readable text from Fast Transcription response shapes."""
    if isinstance(payload.get("combinedPhrases"), list):
        parts = [
            p.get("text", "").strip()
            for p in payload["combinedPhrases"]
            if isinstance(p, dict) and p.get("text")
        ]
        if parts:
            return "\n".join(parts).strip()

    if isinstance(payload.get("phrases"), list):
        parts = [
            p.get("text", "").strip()
            for p in payload["phrases"]
            if isinstance(p, dict) and p.get("text")
        ]
        if parts:
            return "\n".join(parts).strip()

    text = payload.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()

    return json.dumps(payload, indent=2, ensure_ascii=False)


def build_definition(locale: str | None) -> dict:
    definition: dict = {
        "enhancedMode": {
            "enabled": True,
            "model": MODEL,
            "modelOptions": {
                "transcribeStyle": "clean",
            },
        }
    }
    if locale:
        definition["locales"] = [locale]
    return definition


def _post_once(
    *,
    url: str,
    key: str,
    audio_path: Path,
    definition: dict,
    timeout_s: float,
) -> requests.Response:
    # (connect timeout, read timeout)
    timeout = (30.0, float(timeout_s))
    with audio_path.open("rb") as audio_file:
        return requests.post(
            url,
            params={"api-version": API_VERSION},
            headers={"Ocp-Apim-Subscription-Key": key},
            files={
                "audio": (audio_path.name, audio_file, "audio/wav"),
                "definition": (None, json.dumps(definition), "application/json"),
            },
            timeout=timeout,
        )


def transcribe(
    audio_path: Path,
    out_dir: Path | None = None,
    *,
    basename: str = "transcript",
    locale: str | None = None,
    retries: int = DEFAULT_RETRIES,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> tuple[Path, Path]:
    if not audio_path.is_file():
        raise TranscribeError(f"Audio not found: {audio_path}")

    key, endpoint, env_locale = load_config()
    locale = normalize_locale(locale) if locale is not None else env_locale
    url = f"{endpoint}/speechtotext/transcriptions:transcribe"
    definition = build_definition(locale)

    dest = out_dir or OUT_DIR
    dest.mkdir(parents=True, exist_ok=True)
    json_path = dest / f"{basename}.json"
    txt_path = dest / f"{basename}.txt"

    attempts = max(1, int(retries))
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            response = _post_once(
                url=url,
                key=key,
                audio_path=audio_path,
                definition=definition,
                timeout_s=timeout_s,
            )
            if response.status_code in RETRYABLE_STATUS:
                raise TranscribeError(
                    f"Retryable HTTP {response.status_code}: {response.text[:500]}"
                )
            if response.status_code >= 400:
                raise TranscribeError(
                    f"Transcription failed ({response.status_code}):\n{response.text}"
                )
            try:
                payload = response.json()
            except ValueError as exc:
                raise TranscribeError(f"Non-JSON response:\n{response.text}") from exc

            transcript = extract_transcript(payload)
            json_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            txt_path.write_text(transcript + "\n", encoding="utf-8")

            print(f"Model:      {MODEL}")
            print(f"Locale:     {locale or 'auto'}")
            print(f"Attempt:    {attempt}/{attempts}")
            print(f"JSON:       {json_path}")
            print(f"Transcript: {txt_path}")
            preview = transcript if len(transcript) <= 500 else transcript[:500] + "…"
            print("---")
            print(preview)
            return txt_path, json_path

        except (requests.Timeout, requests.ConnectionError, TranscribeError) as exc:
            last_error = exc
            msg = str(exc)
            is_timeout = isinstance(exc, (requests.Timeout, requests.ConnectionError))
            is_retryable_http = "Retryable HTTP" in msg
            if isinstance(exc, TranscribeError) and not is_retryable_http:
                raise
            if attempt >= attempts or not (is_timeout or is_retryable_http):
                break
            wait = min(60.0, 2.0 ** attempt)
            print(
                f"Transcribe retry {attempt}/{attempts} for {audio_path.name}: "
                f"{exc} — sleeping {wait:.1f}s"
            )
            time.sleep(wait)

    raise TranscribeError(
        f"Transcription failed after {attempts} attempt(s) for {audio_path}: {last_error}"
    ) from last_error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe audio with Azure MAI-Transcribe-2."
    )
    parser.add_argument("audio", type=Path, help="Path to WAV/MP3/FLAC audio")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: out/)",
    )
    parser.add_argument(
        "--locale",
        default=None,
        help="MAI language code: ur (default), en, hi, or auto",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help=f"API retries on timeout/5xx/429 (default: {DEFAULT_RETRIES})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help=f"Per-request read timeout seconds (default: {DEFAULT_TIMEOUT_S})",
    )
    args = parser.parse_args(argv)
    try:
        transcribe(
            args.audio,
            args.out_dir,
            locale=args.locale,
            retries=args.retries,
            timeout_s=args.timeout,
        )
    except TranscribeError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
