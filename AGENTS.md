# AGENTS.md

ECE22073 multilingual podcast summarizer — ASR, NER, speaker diarization, multi-tier summarization, podcast TTS generation.

**⚠️ Continue from here first.** Ignore old `Politakis/` references — the repo was restructured 2026-06-11.

## Quick Start

```bash
# Streamlit UI (Docker — recommended)
cd App && docker compose up
# → http://localhost:8501

# Streamlit UI (native Python)
cd App && streamlit run streamlit_app.py

# Colab notebook
# Open Pipeline/notebook.ipynb in Google Colab — run cells 1-4 in order

# Local pipeline
python3 -c "import sys; sys.path.insert(0, 'Pipeline'); from run_pipeline import run_pipeline; run_pipeline('Samples/sample_podcasts/bilingual_long.wav')"

# Evaluation
python3 -c "import sys; sys.path.insert(0, 'Benchmarks'); sys.path.insert(0, 'Pipeline'); from evaluate_real_pipeline import run_real_evaluation; run_real_evaluation()"

# Benchmark all ASR models (on Colab GPU)
python3 Benchmarks/benchmark_all.py Samples/sample_podcasts/bilingual_long.wav Samples/sample_podcasts/bilingual_long_gt.json --normalize
```

## Architecture

```
asr-notebook/
├── Pipeline/        → Core: config, drive_bridge, colab_job_watcher, run_pipeline,
│                       audio_processor, asr_pipeline, llm_integration,
│                       transcript_normalizer, topic_extraction, summary_generator,
│                       podcast_pipeline, diarize_transcript, strip_newlines
├── App/             → Streamlit UI, Dockerfile, docker-compose.yml, requirements
├── Benchmarks/      → evaluate, evaluate_real_pipeline, benchmark_all + 5 others
├── Foundational/    → real_time_processor, sanity_transcribe
├── Samples/         → sample_podcasts/ (test audio + ground truth)
└── Results/         → Output directory
```

**Data flow**: Streamlit (local) ↔ Google Drive ↔ Colab watcher (T4 GPU)

**ASR pipeline**: `audio_processor → asr_pipeline → llm_integration → transcript_normalizer → topic_extraction → summary_generator`

## Key Commands

| Task | Command |
|------|---------|
| Install deps | `pip install -r App/requirements.txt` |
| Docker build + run | `cd App && docker compose up` |
| Pipeline (local) | `python3 -c "import sys; sys.path.insert(0,'Pipeline'); from run_pipeline import run_pipeline; run_pipeline('<path>')"` |
| Evaluation | `python3 -c "import sys; sys.path.insert(0,'Benchmarks'); sys.path.insert(0,'Pipeline'); from evaluate_real_pipeline import run_real_evaluation; run_real_evaluation()"` |
| Colab watcher | Open `Pipeline/notebook.ipynb` → run cells 1-4 |

## Environment

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENAI_API_KEY` | — | Required for LLM stages |
| `NORMALIZATION_MODEL` | `gpt-5.4-mini-2026-03-17` | Transcript cleanup model |
| `ENABLE_TRANSCRIPT_NORMALIZATION` | `true` | Feature flag |
| `HF_TOKEN` | — | Required for pyannote diarization |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | Override for Ollama |
| `LLM_MODEL` | `gpt-5.4-mini-2026-03-17` | Model for NER/summary |

## Gotchas

- **Read this file first** before doing anything — old `Politakis/` paths no longer exist
- **WER always uses raw `transcript.txt`** — normalization never touches it
- **Drive FUSE is broken on Colab** — watcher must use `db.find_new_input_files()` (Drive API), never `os.listdir()`
- **`logging.basicConfig(force=True)`** mandatory before importing watcher modules in Colab
- **Do NOT call `cjw.main_loop()`** — use inline watcher in notebook Cell 4 (see CLAUDE.md for pattern)
- **`App/credentials.json`** required for Drive OAuth → writes `App/token.json`
- **Colab watcher accepts `.wav`, `.mp3`, `.m4a`** — audio_processor handles all via pydub
- **Diarization disabled on Colab** (numpy/pyannote version mismatch)
- **Word timestamps disabled on Colab** (CTranslate2 CUDA alignment crash on T4)
- **No CI, no tests, no linter** — academic deliverable
- **Terminal hangs**: run pipelines directly in your terminal, not via agent bash tool
- **Two `ece22073` folders in Drive** — delete the newer one (created 12:00)

## ASR Models Benchmarked

| Model | Backend | Greek | Notes |
|-------|---------|-------|-------|
| faster-whisper turbo | CTranslate2 int8 | ✅ | Production baseline, 0.73× real-time on M2 |
| faster-whisper large-v3 | CTranslate2 int8 | ✅ | Slower, similar WER |
| Parakeet TDT 0.6B | Transformers CUDA | ✅ | Needs Colab T4 + git transformers |
| Canary 1B V2 | NeMo CUDA | ✅ | Needs `source_lang=target_lang` for ASR mode |
