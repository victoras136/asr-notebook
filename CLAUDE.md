# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ECE22073 — Multilingual Podcast Summarizer. A distributed system where a local Streamlit UI communicates with a GPU-backed Google Colab runtime via Google Drive as a message bus. All ML compute runs on Colab; the local machine only runs the UI.

## Running the App

**Local Streamlit UI:**
```bash
pip install -r Politakis/requirements.txt
cd Politakis && streamlit run streamlit_app.py
```
Requires `Politakis/credentials.json` (Google OAuth). `token.json` is cached alongside it after the first browser OAuth flow.

**Colab watcher (runs in notebook.ipynb, Cell 4 — use the inline version, NOT `main_loop()`):**
```python
import logging, time, sys, os
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", force=True)
sys.path.insert(0, '/content/asr-notebook/Politakis')
os.chdir('/content/asr-notebook')
import config, drive_bridge as db, colab_job_watcher as cjw
db.init_drive_structure()
processed_names = set()
while True:
    try:
        for file_info in db.find_new_input_files():
            fname = file_info["name"]
            if fname.lower().endswith(".wav") and fname not in processed_names:
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

**Run pipeline directly (local/Colab):**
```bash
python3 Politakis/run_pipeline.py <path_to_audio.wav>
```

**Evaluation:**
```bash
python3 Politakis/evaluate_real_pipeline.py
```

## Architecture

All code lives in `Politakis/`. The notebook `notebook.ipynb` at the repo root is the Colab entry point (clone + install + auth + run watcher).

### Data Flow

```
[User] → Streamlit (local)
           │ uploads WAV   → Drive: ece22073/input/{job_id}.wav
           │ writes JSON   → Drive: ece22073/input/podcast_jobs/{job_id}.json
           │ polls status  ← Drive: ece22073/output/{job_id}/status.json
           │ reads results ← Drive: ece22073/output/{job_id}/*.json

[Colab watcher] polls Drive API every 10s
  → finds WAV   → runs ASR pipeline  → uploads results to output/{job_id}/
  → finds JSON  → runs podcast TTS   → uploads MP3 to output/podcasts/
```

### ASR Pipeline Stages (run_pipeline.py)

`audio_processor` → `asr_pipeline` → `llm_integration` → `transcript_normalizer` → `topic_extraction` → `summary_generator`

Results land in `Politakis/results/`: `transcript.json`, `transcript.txt`, `normalized_transcript.txt`, `summary_outputs.json`, `quality_metrics.json`, `processing_time_analysis.json`.

### Key Design Rules

- **Drive FUSE mount is unreliable** — the watcher must use `db.find_new_input_files()` (Drive API polling), never `os.listdir()` on `/content/drive/...`. This is the root cause of the most common "jobs never picked up" bug.
- **`logging.basicConfig(force=True)` is mandatory** before importing watcher modules in Colab — without it, all errors are silently swallowed because `basicConfig` only runs under `__main__`.
- **Do NOT call `cjw.main_loop()`** when importing as a module — logging isn't configured then. Use the inline watcher loop above.
- **All Drive paths come from `config.py`** — no file hardcodes Drive folder strings.
- **`drive_bridge.py` auto-detects environment** (Colab vs local) — same API surface, different auth path.
- **Job IDs** are derived from filenames (`{job_id}.wav`, `{job_id}.json`), not generated at dispatch time.
- **Failed jobs are not archived** — the input file stays in `input/` so a Colab restart automatically retries.

### TTS Models (podcast_pipeline.py)

Only Kokoro-82M is tested end-to-end. Voices: `af_heart` (Speaker A), `am_michael` (Speaker B). Install: `pip install "kokoro>=0.9.4"`. Other models (Dia-1.6B, Bark, XTTS-v2, F5-TTS) are wired up but untested.

## Environment Variables

| Variable | Where | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | Colab Secrets | GPT-4o-mini for LLM stages |
| `HF_TOKEN` | Colab Secrets | pyannote diarization (must also accept model terms at huggingface.co/pyannote/speaker-diarization-3.1) |
| `GITHUB_TOKEN` | Colab Secrets | Private repo clone in Cell 1 |

## Known Gotchas

- **`pip install bark`** and **`pip install dia`** install unrelated packages. TTS dependencies must be installed from git (see `requirements_colab.txt`).
- **`scipy`/`numpy` mismatch** after chaotic pip install → `pip install --force-reinstall scipy numpy` + kernel restart.
- **`ImportError: cannot import name 'box_iou'`** → caused by `sys.modules.setdefault` stubs for `transformers.loss.loss_for_object_detection` — remove them.
- **Two `ece22073` folders in Drive** → `get_or_create_folder` picks oldest by `createdTime`. Delete the newer duplicate from Drive UI.
- **`RuntimeError: Event loop is closed`** spam from httpx async cleanup in ASR thread — non-fatal, jobs complete fine.
