#!/usr/bin/env python3
"""Compare before/after transcripts to spot missing content."""

from __future__ import annotations

import argparse
import re
from difflib import SequenceMatcher
from pathlib import Path


MONEY_RE = re.compile(
    r"(?:£|\$|€)\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\s*(?:pounds?|gbp|usd)\b|"
    r"\b\d{2,6}(?:\.\d+)?\s*(?:pounds?|gbp)\b",
    re.I,
)
PERCENT_RE = re.compile(r"\b\d{1,3}(?:\.\d+)?\s*%|\b\d{1,3}\s*percent\b", re.I)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
TIME_RE = re.compile(
    r"\b(?:\d{1,2}:\d{2}\s*(?:am|pm)?|\d{1,2}\s*(?:am|pm)|every\s+\d+\s+months?|"
    r"\d+\s+days?|\d+\s+minutes?)\b",
    re.I,
)
NAME_HINT_RE = re.compile(
    r"\b(?:once you pay|payment plan|reserve|townhouse|apartment|"
    r"national forest|islamabad|airport|savills|one homes)\b",
    re.I,
)


STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "to", "of", "in", "on", "for", "is",
    "are", "was", "were", "be", "been", "being", "it", "this", "that", "you",
    "your", "we", "our", "i", "me", "my", "he", "she", "they", "them", "so",
    "just", "like", "know", "yeah", "yes", "no", "ok", "okay", "um", "uh",
    "right", "well", "look", "say", "said", "going", "gonna", "will", "can",
    "with", "as", "at", "by", "from", "about", "into", "over", "out", "up",
    "do", "does", "did", "have", "has", "had", "if", "then", "than", "very",
}


def normalize(text: str) -> str:
    text = text.lower()
    # Drop ASR stutter fragments: "ha- you" / "you-"
    text = re.sub(r"\b\w{1,3}-\s*", " ", text)
    text = re.sub(r"[^\w\s%@.-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Collapse repeated consecutive words: "go through go through go through"
    parts = text.split()
    collapsed: list[str] = []
    for w in parts:
        if not collapsed or collapsed[-1] != w:
            collapsed.append(w)
    return " ".join(collapsed)


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def words(text: str) -> list[str]:
    return [w for w in normalize(text).split() if w]


def content_words(text: str) -> list[str]:
    return [w for w in words(text) if w not in STOPWORDS and len(w) > 1]


def coverage(needle: str, haystack_norm: str, min_ratio: float = 0.72) -> bool:
    n = normalize(needle)
    if not n:
        return True
    if n in haystack_norm:
        return True
    n_words = n.split()
    if len(n_words) <= 2:
        return all(w in haystack_norm for w in n_words)

    # Content-word bag overlap (handles stutter / mild paraphrase)
    cw = content_words(needle)
    if cw:
        h_set = set(haystack_norm.split())
        hit = sum(1 for w in cw if w in h_set)
        if hit / len(cw) >= 0.8:
            return True

    h_words = haystack_norm.split()
    window = max(len(n_words) + 4, len(n_words))
    for i in range(0, max(1, len(h_words) - len(n_words) + 1)):
        chunk = " ".join(h_words[i : i + window])
        if SequenceMatcher(None, n, chunk).ratio() >= min_ratio:
            return True
    return False


def extract_facts(text: str) -> list[str]:
    facts: list[str] = []
    for rx in (MONEY_RE, PERCENT_RE, EMAIL_RE, TIME_RE):
        facts.extend(m.group(0).strip() for m in rx.finditer(text))
    # Keep distinctive content phrases that often matter on sales calls
    for sent in sentences(text):
        if NAME_HINT_RE.search(sent) and 20 <= len(sent) <= 180:
            facts.append(sent)
    # de-dupe case-insensitively, preserve order
    seen: set[str] = set()
    out: list[str] = []
    for f in facts:
        key = normalize(f)
        if key and key not in seen:
            seen.add(key)
            out.append(f)
    return out


def missing_sentences(before_text: str, after_text: str) -> list[str]:
    after_norm = normalize(after_text)
    missing: list[str] = []
    for sent in sentences(before_text):
        # Skip tiny / filler lines — they create false "missing" noise
        if len(content_words(sent)) < 3:
            continue
        if not coverage(sent, after_norm):
            missing.append(sent)
    return missing


def format_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    total = int(round(seconds))
    m, s = divmod(total, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def word_overlap(before_text: str, after_text: str) -> float:
    bw = set(words(before_text))
    aw = set(words(after_text))
    if not bw:
        return 1.0
    return len(bw & aw) / len(bw)


def compare_transcripts(
    before_text: str,
    after_text: str,
    *,
    stem: str,
    before_duration_s: float | None = None,
    after_duration_s: float | None = None,
    mode: str = "smart",
) -> str:
    before_words = words(before_text)
    after_words = words(after_text)
    overlap = word_overlap(before_text, after_text)
    missing = missing_sentences(before_text, after_text)
    facts = extract_facts(before_text)
    after_norm = normalize(after_text)
    missing_facts = [f for f in facts if not coverage(f, after_norm, min_ratio=0.65)]
    kept_facts = [f for f in facts if f not in missing_facts]

    saved = None
    pct = None
    if before_duration_s and after_duration_s and before_duration_s > 0:
        saved = before_duration_s - after_duration_s
        pct = (saved / before_duration_s) * 100

    lines: list[str] = [
        f"# Before vs After — `{stem}`",
        "",
        "Use this to spot-check whether pause compression dropped important speech.",
        "",
        "## Audio",
        "",
        "| | Duration |",
        "|---|---|",
        f"| Before (convert only) | {format_seconds(before_duration_s)} |",
        f"| After (`{mode}`) | {format_seconds(after_duration_s)} |",
    ]
    if saved is not None and pct is not None:
        lines.append(f"| Saved | {format_seconds(saved)} ({pct:.1f}%) |")
    lines += [
        "",
        "## Transcript size",
        "",
        "| | Words | Chars |",
        "|---|---:|---:|",
        f"| Before | {len(before_words)} | {len(before_text)} |",
        f"| After | {len(after_words)} | {len(after_text)} |",
        f"| Word coverage (before→after) | {overlap * 100:.1f}% | |",
        "",
        "## Verdict",
        "",
    ]

    critical_missing = [
        f
        for f in missing_facts
        if MONEY_RE.search(f) or PERCENT_RE.search(f) or EMAIL_RE.search(f)
    ]
    if not missing and not missing_facts:
        lines.append(
            f"**Looks aligned.** No clear missing sentences or key facts "
            f"(word coverage {overlap * 100:.1f}%). Still skim payment / name details."
        )
    elif critical_missing or (len(missing) >= 5 and overlap < 0.92):
        lines.append(
            "**Review needed.** Some before-content looks weak or absent in after. "
            "Check the lists below (especially money / %). "
            "If critical, re-run with `--mode off`."
        )
    else:
        lines.append(
            "**Mostly aligned** with a few possible gaps — verify the items below."
        )

    lines += [
        "",
        "## Key facts from Before",
        "",
        "`[x]` = found in After (kept). `[ ]` = not found (check manually).",
        "",
    ]
    if not facts:
        lines.append("_No money / % / email / timing facts extracted._")
    else:
        for f in kept_facts:
            lines.append(f"- [x] {f}")
        for f in missing_facts:
            lines.append(f"- [ ] **MISSING?** {f}")

    lines += [
        "",
        "## Possible wording gaps (Before → After)",
        "",
        "These are lines where After may use different wording, not always deleted speech. "
        "Stutters/repeats often look “missing” even when the meaning is kept.",
        "",
    ]
    if not missing:
        lines.append("_None detected._")
    else:
        for i, sent in enumerate(missing, 1):
            lines.append(f"{i}. {sent}")

    lines += [
        "",
        "## How to use",
        "",
        "1. `[x]` facts are already in After — they are **not** missing.",
        "2. Only worry about `[ ]` money/%/email items and any gap that changes a commitment.",
        "3. If a payment / name / commitment line is truly gone, re-run with `--mode off`.",
        "",
    ]
    return "\n".join(lines)


def write_comparison(
    before_txt: Path,
    after_txt: Path,
    out_path: Path,
    *,
    stem: str,
    before_duration_s: float | None = None,
    after_duration_s: float | None = None,
    mode: str = "smart",
) -> Path:
    before_text = before_txt.read_text(encoding="utf-8")
    after_text = after_txt.read_text(encoding="utf-8")
    md = compare_transcripts(
        before_text,
        after_text,
        stem=stem,
        before_duration_s=before_duration_s,
        after_duration_s=after_duration_s,
        mode=mode,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare before/after transcripts.")
    parser.add_argument("before", type=Path, help="Before transcript .txt")
    parser.add_argument("after", type=Path, help="After transcript .txt")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output markdown path (default: next to after as comparison.md)",
    )
    parser.add_argument("--stem", default="audio")
    parser.add_argument("--mode", default="smart")
    parser.add_argument("--before-duration", type=float, default=None)
    parser.add_argument("--after-duration", type=float, default=None)
    args = parser.parse_args(argv)
    out = args.output or (args.after.parent.parent / "comparison.md")
    path = write_comparison(
        args.before,
        args.after,
        out,
        stem=args.stem,
        before_duration_s=args.before_duration,
        after_duration_s=args.after_duration,
        mode=args.mode,
    )
    print(f"Comparison: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
