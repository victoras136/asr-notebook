# AGENTS.md

ECE22073 academic project: multilingual podcast summarizer with ASR, NER, speaker diarization, and multi-tier summarization.

## Commands

```bash
# Install dependencies (one-time)
pip install -r Politakis/requirements.txt

# Generate multilingual test audio + ground truth (one-time)
python3 setup_bilingual_test.py

# Full pipeline on an audio file
python3 Politakis/run_pipeline.py <path_to_audio.wav>

# Evaluation against rubric gates (WER ≤ 0.08, ROUGE-1 ≥ 0.40, Topic Recall ≥ 0.80)
python3 Politakis/evaluate_real_pipeline.py

# Streamlit dashboard (MUST run from inside Politakis/ — uses relative Path("results"))
cd Politakis && streamlit run streamlit_app.py

# Benchmark all ASR models
python3 Politakis/benchmark_all.py <audio.wav> <ground_truth.json> [--normalize]
```

Pipeline and evaluation scripts inject `Politakis/` into `sys.path` automatically and can run from repo root. The Streamlit app uses `Path("results")` (relative), so it must be launched from within `Politakis/`.

No CI, no tests, no linter config. This is an academic deliverable.

## Architecture

Six sequential modules in `Politakis/`:

```
audio_processor → asr_pipeline → llm_integration → transcript_normalizer
                → topic_extraction → summary_generator
```

| Stage | Module | Responsibility |
|-------|--------|----------------|
| 1 | `audio_processor` | Load any audio format via pydub, normalize to −20 dBFS, resample to 16 kHz mono, split into chunks on silence boundaries using Silero VAD. Outputs generator of chunk dicts. |
| 2 | `asr_pipeline` | Transcribes chunks with `faster-whisper` (CTranslate2 backend), runs pyannote speaker diarization, tags speakers, filters low-confidence segments. Outputs generator of chunk dicts. |
| 3a | `llm_integration` | Sends transcript windows to an LLM (OpenAI-compatible API or local Ollama) every ~2 min for live-ticker NER and segment summarization. Async, non-blocking. |
| 3b | `transcript_normalizer` | LLM-based entity/term repair: corrects proper nouns, org names, technical terms. Feature-flagged via `ENABLE_TRANSCRIPT_NORMALIZATION`. Never affects WER (WER uses raw `transcript.txt`). |
| 3c | `topic_extraction` | Deduplicates and normalizes entity lists from the live ticker into a frequency-ranked registry. Supports both batch and streaming incremental modes. |
| 4 | `summary_generator` | Produces YouTube-style timestamped chapters and three summary tiers (TL;DR, Executive, Deep Dive) from the full transcript and entity registry. Also provides Q&A backend. |
| — | `real_time_processor.py` | Async wrapper emitting live events (`chunk`, `ticker`, `summary`, `done`) for streaming use cases. |
| — | `evaluate.py` | Metric computation library (WER, ROUGE, topic recall, latency, language support). Used by notebooks and evaluation scripts. |

Output directory: `Politakis/results/` — `transcript.json`, `transcript.txt`, `summary_outputs.json`, `quality_metrics.json`.

## Environment prerequisites

- **PyAnnote diarization** requires a HuggingFace token. Export before running:
  ```bash
  export HF_TOKEN="hf_..."
  ```
- **LLM** defaults to `gpt-5.4-mini-2026-03-17` via the OpenAI API (`OPENAI_API_KEY`). For local Ollama:
  ```bash
  export OPENAI_API_KEY="sk-..."
  export LLM_BASE_URL=http://localhost:11434/v1
  export LLM_MODEL=llama3
  ```
- **Transcript normalization** uses `NORMALIZATION_MODEL` (default `gpt-5.4-mini-2026-03-17`). Set `ENABLE_TRANSCRIPT_NORMALIZATION=false` to skip.
- **ffmpeg** is only needed for compressed formats (mp3, ogg, m4a). `.wav` files are parsed natively via Python's `wave` library.

## Code style conventions

### Imports

- Every file begins with `from __future__ import annotations` as the first import.
- Standard library imports first, then third-party, then local/relative. No blank line separators between groups within the same block.
- Heavy ML dependencies (torch, openai, faster-whisper, pyannote) are imported at module level. Lightweight helpers (lazy clients, fallback loaders) use function-level imports with try/except.
- Local module imports use bare names (`import audio_processor as ap`), relying on `sys.path.insert(0, ...)` in the orchestrating script.

### Typing

- **All public functions** must have type annotations on parameters and return type.
- Use `from typing import Any, Generator, AsyncIterator, TypedDict` as needed.
- Use Python 3.10+ union syntax: `str | Path`, `dict | None`, `list[dict]`.
- **TypedDict** for every schema object that crosses module boundaries (entity registry, summary outputs, quality metrics, processing analysis). Document the full JSON schema in the docstring block above the TypedDict.
- Module-level constants are type-annotated: `TARGET_SAMPLE_RATE: int = 16_000`.
- Inline variable annotations for numpy/torch types: `samples_f32: np.ndarray = ...`.

### Naming

- `snake_case` for variables, functions, methods, and modules.
- `PascalCase` for classes: `SileroVAD`, `AccumulatedTranscript`, `EntityRegistry`, `ResourceMonitor`.
- `UPPER_CASE` for module-level constants and thresholds.
- Private methods and module-level helpers are prefixed with `_` (e.g., `_call_llm`, `_normalise_entity`, `_rank_entities`).

### Error handling

- Defensive dict access with `.get(key, default)` throughout — never rely on dict keys being present.
- LLM JSON responses are parsed with try/except `json.JSONDecodeError`; fall back to empty defaults rather than crashing.
- Optional dependencies (diarization, GPU acceleration) fail gracefully with `logger.warning()` and return empty results.
- Pipeline stages in `run_pipeline.py` use the `_run_stage()` wrapper that catches all exceptions and returns fallback payloads.

### Functions and generators

- Streaming stages (`audio_processor`, `asr_pipeline`) use **generator functions** (`yield` chunk dicts) to avoid buffering entire audio in RAM.
- Async LLM calls (`llm_integration`, `summary_generator`) use `asyncio.create_task()` to fire background calls while the stream continues.
- Every async function has a synchronous wrapper (e.g., `generate_summary()` wraps `generate_summary_async()` via `asyncio.run()`).
- Keyword-only arguments use `*` separator for tuneable parameters (e.g., `*, vad_threshold=0.5, min_chunk_sec=25.0`).

### JSON and serialization

- All data crossing module boundaries is parsed JSON dict — never raw multi-line strings.
- `json.dumps(payload, ensure_ascii=False)` for multilingual text (Greek + English).
- `round(value, 4)` on all floats before embedding in dicts — ensures clean JSON and prevents NumPy subtypes.
- NumPy numeric types (float32, int64) must be cast with `float()` or `int()` before JSON serialization.
- File I/O uses `Path.read_text(encoding="utf-8")` / `Path.write_text()` and `json.dump(..., indent=2, ensure_ascii=False)`.

### Comments and docstrings

- Extensive **"why" comments** on every non-trivial block — explain the rationale, not what the code does.
- Module docstrings describe responsibility and document the output schema.
- Sections separated with `# ═══ Section Name ═══` dividers.
- Section-numbering comments in longer modules (e.g., `# 1. LLM Client`, `# 4. Q&A`).

### Logging

- Every module creates its own logger: `logger = logging.getLogger(__name__)`.
- Logging format: `"%(asctime)s | %(name)s | %(levelname)s | %(message)s"`.
- Use `logger.info()` for key pipeline events, `logger.warning()` for non-fatal issues, `logger.error()` for failures.
- Emoji prefixes in orchestration logs (`🎙️`, `🧠`, `🗂️`, `📝`) — not used in library modules.

## Key files

| File | Role |
|------|------|
| `Politakis/run_pipeline.py` | End-to-end orchestrator (main entrypoint) |
| `Politakis/evaluate_real_pipeline.py` | Rubric-gate verification against ground truth |
| `Politakis/evaluate.py` | Metric computation library (WER, ROUGE, recall, latency) |
| `Politakis/streamlit_app.py` | Interactive web dashboard |
| `Politakis/real_time_processor.py` | Async streaming wrapper with live event emission |
| `Politakis/transcript_normalizer.py` | LLM-based ASR error correction (feature-flagged) |
| `Politakis/requirements.txt` | All Python dependencies |
| `Politakis/results/` | Output directory (transcripts, summaries, metrics) |
| `Politakis/sample_podcasts/` | Test audio files and generators |
| `setup_bilingual_test.py` | Generates bilingual test audio + ground truth JSON |
| `Politakis/benchmark_all.py` | Multi-model ASR benchmark with normalization support |

## Gotchas

- **ROUGE-1 key**: evaluation uses `rouge1_f1`, not `rouge1`. `evaluate_real_pipeline.py:165` was patched for this.
- **NumPy → JSON**: `audio_processor.py` casts RMS values with `float()` before JSON serialization. Any new numeric feature in chunk dicts must be cast to native Python types.
- **PyAnnote API version**: `asr_pipeline.py` checks for `.speaker_diarization` attribute on the result; modern PyAnnote wraps `Annotation` inside `DiarizeOutput`.
- **Ground truth files**: `evaluate_real_pipeline.py` looks for `sample_podcasts/<stem>_gt.json` matching the source audio filename, falling back to `results/ground_truth.json`. Generate proper ground truth with `setup_bilingual_test.py`.
- **Silero VAD** is stateful — `vad_chunker()` calls `vad.reset()` at end-of-stream.
- **Chunks < 0.3 s** are silently skipped by `vad_chunker()` as trailing silence.
- **Transcript normalization**: uses `ENABLE_TRANSCRIPT_NORMALIZATION` env var (default `true`). Never affects WER — raw `transcript.txt` is the immutable reference. Entity re-extraction runs on normalized text and replaces ticker entities.
- **Long-running pipelines**: run directly in your terminal, not via agent bash tools. Pipeline runs may produce no visible output for extended periods.
- No CI, no tests, no linter config. This is an academic deliverable.
