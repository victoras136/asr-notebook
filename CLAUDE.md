# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ECE22073 — Multilingual Podcast Summarizer. A distributed system where a local Streamlit UI communicates with a GPU-backed Google Colab runtime via Google Drive as a message bus. All ML compute runs on Colab; the local machine only runs the UI.

> **⚠️ Note:** Old session notes and handoffs may reference a `Politakis/` subfolder — that folder no longer exists. The repo was restructured 2026-06-11. Code now lives in `Pipeline/` (backend) and `App/` (UI).

## Running the App

**Local Streamlit UI (Docker — recommended):**
```bash
cd App
cp .env.example .env   # first time only — paste OPENAI_API_KEY
docker compose up
# → http://localhost:8501
```
System deps (`ffmpeg`, `espeak-ng`) and Python packages are pre-installed in the image. Code changes reflect instantly via volume mount — no rebuild needed.

**Local Streamlit UI (native Python):**
```bash
brew install ffmpeg espeak-ng          # system deps, once
pip install -r App/requirements.txt
cd App && streamlit run streamlit_app.py
```
Requires `App/credentials.json` (Google OAuth). `App/token.json` is cached after the first browser OAuth flow (click **Connect Google Drive** in the sidebar).

**Colab watcher (GPU backend) — Cell 4 in `Pipeline/notebook.ipynb`. Use the inline version, NOT `main_loop()`:**
```python
import logging, time, sys, os
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", force=True)
sys.path.insert(0, '/content/asr-notebook/Pipeline')
os.chdir('/content/asr-notebook')
import config, drive_bridge as db, colab_job_watcher as cjw
db.init_drive_structure()
processed_names = set()
while True:
    try:
        for file_info in db.find_new_input_files():
            fname = file_info["name"]
            if fname.lower().endswith((".wav", ".mp3", ".m4a")) and fname not in processed_names:
                processed_names.add(fname)
                cjw._handle_asr_job(file_info)
        for file_info in db.find_new_podcast_jobs():
            fname = file_info["name"]
            if fname not in processed_names:
                processed_names.add(fname)
                cjw._handle_podcast_job(file_info)
    except Exception as e:
        logging.error("Watcher error: %s", e, exc_info=True)
    time.sleep(config.POLL_INTERVAL_SEC)
```

**Run pipeline directly (local or Colab):**
```bash
python3 -c "import sys; sys.path.insert(0, 'Pipeline'); from run_pipeline import run_pipeline; run_pipeline('Samples/sample_podcasts/bilingual_long.wav')"
```

**Evaluation:**
```bash
python3 -c "import sys; sys.path.insert(0, 'Benchmarks'); sys.path.insert(0, 'Pipeline'); from evaluate_real_pipeline import run_real_evaluation; run_real_evaluation()"
```

**Benchmark all ASR models (Colab GPU):**
```bash
python3 Benchmarks/benchmark_all.py Samples/sample_podcasts/bilingual_long.wav Samples/sample_podcasts/bilingual_long_gt.json --normalize
```

## Architecture

### Data Flow

```
[User] → Streamlit (local, App/)
           │ uploads audio → Drive: ece22073/input/{job_id}.wav
           │ writes JSON   → Drive: ece22073/input/podcast_jobs/{job_id}.json
           │ polls status  ← Drive: ece22073/output/{job_id}/status.json
           │ reads results ← Drive: ece22073/output/{job_id}/*.json

[Colab watcher] polls Drive API every 10s
  → finds audio → runs ASR pipeline  → uploads results to output/{job_id}/
  → finds JSON  → runs podcast TTS   → uploads MP3 to output/podcasts/
```

### ASR Pipeline Stages (`Pipeline/run_pipeline.py`)

`audio_processor → asr_pipeline → llm_integration → transcript_normalizer → topic_extraction → summary_generator`

Results: `transcript.json`, `transcript.txt`, `normalized_transcript.txt`, `summary_outputs.json`, `quality_metrics.json`, `processing_time_analysis.json`.

### Streamlit App (`App/streamlit_app.py`)

Single-file app. At module level it: sets page config, injects all CSS, initialises `st_autorefresh` (disabled at ~11 days when idle, enabled at `LOCAL_POLL_INTERVAL_SEC` when processing), bootstraps session state, and conditionally calls `db.authenticate()` only when `App/token.json` already exists (to avoid blocking on autorefresh reruns).

Five pages, routed by `st.session_state["_current_page"]`:

| Key | Function | Description |
|-----|----------|-------------|
| `Upload` | `_page_upload()` | File upload → Drive → transcription polling |
| `Notebook` | `_page_notebook()` | 3-column: Sources \| Chat \| Studio |
| `Summaries` | `_page_summaries()` | TL;DR / Executive / Deep Dive |
| `Podcast` | `_page_podcast()` | TTS podcast generation |
| `Accuracy Check` | `_page_accuracy()` | WER + ROUGE single/bulk comparison |

`comparison_metrics.py` lives in `App/` (not `Pipeline/`) and is imported directly by `streamlit_app.py`.

### CSS Design System

- Fonts: IBM Plex Sans (body) + IBM Plex Mono (labels/chips) from Google Fonts
- Background: `#111113` (app), `#0c0c0e` (sidebar), `#16161a` (cards)
- Accent: `#4a9eff` (blue), `#3dde8f` (green)
- Nav buttons are `st.sidebar.button()` styled via `section[data-testid="stSidebar"] .stButton > button` with `all: unset !important`. Active state is applied post-render via an injected `<script>` that queries buttons by `.innerText`.
- **`_nav_button()` on line ~330 is dead code** — superseded by the `sections` loop in `_sidebar()`. Do not call it; delete when cleaning up.

### Key Design Rules

- **Drive FUSE mount is unreliable** — the watcher must use `db.find_new_input_files()` (Drive API polling), never `os.listdir()` on `/content/drive/...`. This is the root cause of the most common "jobs never picked up" bug.
- **`logging.basicConfig(force=True)` is mandatory** before importing watcher modules in Colab — without it, all errors are silently swallowed because `basicConfig` only runs under `__main__`.
- **Do NOT call `cjw.main_loop()`** when importing as a module — logging isn't configured then. Use the inline watcher loop above.
- **All Drive paths come from `config.py`** — no file hardcodes Drive folder strings.
- **`drive_bridge.py` auto-detects environment** (Colab vs local) — same API surface, different auth path.
- **Job IDs** are derived from filenames (`{job_id}.wav`, `{job_id}.json`), not generated at dispatch time.
- **Failed jobs are not archived** — the input file stays in `input/` so a Colab restart automatically retries.
- **WER always uses raw `transcript.txt`** — the LLM normalization pass never modifies it.

### Sidebar Collapse Bug (open)

`header { visibility: hidden; }` on line 49 hides the entire Streamlit `<header>` element, including the expand arrow that appears when the sidebar is collapsed — making it impossible to reopen the sidebar without a page refresh. Fix: replace `header` in that rule with `[data-testid="stToolbar"]` to target only the deploy button.

### TTS Models (`Pipeline/podcast_pipeline.py`)

Only Kokoro-82M is tested end-to-end. Voices: `af_heart` (Speaker A), `am_michael` (Speaker B). Install: `pip install "kokoro>=0.9.4"`. Other models (Dia-1.6B, Bark, XTTS-v2, F5-TTS) are wired up but untested; Dia-1.6B needs 10 GB VRAM.

## Environment Variables

| Variable | Where | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | Colab Secrets / `.env` | GPT-4o-mini for LLM stages — required |
| `HF_TOKEN` | Colab Secrets | pyannote diarization (accept model terms at huggingface.co/pyannote/speaker-diarization-3.1) |
| `GITHUB_TOKEN` | Colab Secrets | Private repo clone in Colab Cell 1 |
| `NORMALIZATION_MODEL` | env | Override for transcript cleanup model |
| `LLM_MODEL` | env | Override for NER/summary model |
| `LLM_BASE_URL` | env | Override to point at local Ollama |

## Known Gotchas

- **`pip install bark`** and **`pip install dia`** install unrelated packages — install TTS models from git (see `App/requirements_colab.txt`).
- **`scipy`/`numpy` mismatch** after chaotic pip install → `pip install --force-reinstall scipy numpy` + kernel restart.
- **`ImportError: cannot import name 'box_iou'`** → caused by `sys.modules.setdefault` stubs for `transformers.loss.loss_for_object_detection` — remove them.
- **Two `ece22073` folders in Drive** → `get_or_create_folder` picks oldest by `createdTime`. Delete the newer duplicate from Drive UI.
- **`RuntimeError: Event loop is closed`** spam from httpx async cleanup in ASR thread — non-fatal, jobs complete fine.
- **Diarization disabled on Colab** — numpy/pyannote version mismatch on current Colab image.
- **Word timestamps disabled on Colab** — CTranslate2 CUDA alignment crashes on T4; enabled only on local CPU runs.
- **Drive OAuth blocks on autorefresh** — `db.authenticate()` at module level is guarded by `token.json` existence check to avoid spawning a new browser tab on every poll cycle.
