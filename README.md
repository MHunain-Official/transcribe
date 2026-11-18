# Cost-aware, speech-safe MAI transcription

Convert audio → compress **long dead-air only** (keep all speech) → Azure **MAI-Transcribe-2**.

**Goals (in order):**
1. Keep conversational words / important details  
2. Reduce billed minutes by shortening flat no-speech stretches  

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# ffmpeg on PATH required
```

`.env`:

```env
AZURE_SPEECH_KEY=...
AZURE_SPEECH_REGION=centralindia
AZURE_SPEECH_ENDPOINT=https://centralindia.api.cognitive.microsoft.com/
AZURE_SPEECH_LOCALE=ur
```

`AZURE_SPEECH_LOCALE=ur` forces **Urdu script** for Eng+Urdu calls. Without it, MAI auto-detect often writes the same speech as **Hindi (Devanagari)** (e.g. `एक नया` instead of `ایک نیا`).

## Usage

```bash
# Default: before + after + comparison (2 API calls), locale=ur
python run.py 405

# Explicit Urdu (Eng+Urdu → اردو + English)
python run.py ..\voicenotes\405 --locale ur

# Max accuracy, no cost save
python run.py 405 --mode off

# Preprocess only
python run.py 405 --skip-transcribe

# After only (1 API call)
python run.py 405 --skip-before
```

## Batch (parallel + retries + skip on hang/error)

```powershell
cd C:\OneHomes\demo\transcribe
.\.venv\Scripts\Activate.ps1

# All voicenotes in parallel (3 workers). Retries API/job failures.
# If one file hangs past --job-timeout, it is skipped and logged; others keep going.
python batch.py ..\voicenotes --workers 3 --locale ur

# Safer on rate limits
python batch.py ..\voicenotes --workers 2 --job-retries 3 --api-retries 4

# After-only (1 API call per file)
python batch.py ..\voicenotes --skip-before --workers 3
```

Logs:
- `out/batch_summary.md` — OK / error / timeout table
- `out/batch_results.json` — per-job timings (preprocess/API/total)
- `out/batch_errors.jsonl` — machine-readable skip/failure records
- `out/full_report.md` — full timing + quality report (`python report.py`)

Outputs per file:

```text
out/405/
  before/          # convert only (baseline)
    405.wav
    transcript.txt
    transcript.json
  after/           # selected mode (default: smart)
    405.clean.wav
    transcript.txt
    transcript.json
  comparison.md    # summary: duration save + missing facts/sentences
```

## Modes

| Mode | Behavior | Cost | Risk to words |
|------|----------|------|----------------|
| `smart` (default) | Keep all speech; cap dead-air ≥ 0.6s down to ~0.25s | ~10–15% on talky calls | Low |
| `off` | Convert only | None | None |
| `edges` | Trim start/end dead air only | Small | Very low |
| `long` | Legacy silenceremove | Higher save | Higher — avoid for sales calls |

On `405`, smart saved **~13%** minutes with **all key phrases kept** (including “once you pay 80%… stop”).  
If cleaning removes >30% duration, a **WARNING** is printed — spot-check payment details.

Note: earlier ~50% “savings” came from deleting real speech. True no-conversation dead-air on this call is closer to 10–15% if words are protected.

## What we learned

Aggressive silence deletion saved ~50% time but dropped critical lines
(e.g. “once you pay 80%, you stop”).  
`smart` shortens only detected no-conversation stretches and keeps speech segments intact.
