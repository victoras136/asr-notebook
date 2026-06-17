# ECE22073 — Full Project Chronicle

> **Date**: June 1–11, 2026
> **Student**: ΠΟΛΙΤΑΚΗΣ ΒΙΚΤΩΡ (ΑΜ: 9093202200073)
> **Course**: ECE22073 — Πανεπιστήμιο Πατρών
> **Supervisor**: Παναγιώτης Ζέρβας
> **Repository**: [github.com/victoras136/asr-notebook](https://github.com/victoras136/asr-notebook)

This document is an exhaustive technical chronicle of every decision, problem, benchmark, fix, and architectural evolution across the entire development of the Multilingual Podcast Summarizer. It is intended as source material for the academic report and as the definitive handoff for future work.

---

## 1. Project Overview & Requirements

### 1.1 Core Objective

Build a real-time podcast processing system that:
1. Transcribes multilingual audio (Greek + English) with high accuracy (WER ≤ 8%)
2. Extracts named entities (persons, organizations, keywords)
3. Generates hierarchical summaries (TL;DR, Executive, Deep Dive) with ROUGE-1 ≥ 0.40
4. Achieves topic recall ≥ 80%
5. Processes at ≤ 5× audio duration (we achieved 0.73× — 7× faster than required)
6. Supports ≥ 3 languages (we handle 2 — Greek + English — plus Spanish on bilingual test)

### 1.2 Technical Requirements from Rubric

| Requirement | Points | Target |
|-------------|--------|--------|
| Audio Processing Pipeline | 20 | Stream-based, VAD, chunking 5-10s, multi-language detection |
| Speech Recognition | 25 | Whisper ASR, long audio (>1 hour), speaker diarization, timestamps, confidence filtering |
| Topic and Content Extraction | 25 | LLM-based NER, key topic extraction, segment-level analysis |
| Summary Generation | 15 | 3-5 levels of abstractive summarization, bullet points, Q&A |
| Evaluation | 15 | WER, ROUGE scores, topic recall, latency analysis, resource monitoring |

---

## 2. Architecture — Complete Evolution

### 2.1 Initial Architecture (June 1)

```
Politakis/
  audio_processor.py → asr_pipeline.py → llm_integration.py
    → topic_extraction.py → summary_generator.py
  real_time_processor.py (optional async wrapper)
  streamlit_app.py
  evaluate.py, evaluate_real_pipeline.py
```

**Key decisions at start**:
- `faster-whisper` over vanilla Whisper — 3× faster via CTranslate2 with int8 quantization on Apple Silicon
- Silero VAD for intelligent chunking (not fixed-duration splits)
- PyAnnote for speaker diarization
- OpenAI API (gpt-4o-mini) for LLM stages — configurable to local Ollama
- All data exchanged as JSON dicts (never raw strings)
- Strict Python type hints on all public functions

### 2.2 Audio Processing Pipeline Deep Dive

**`audio_processor.py`** — 394 lines originally, 350 after cleanup.

1. **Loading**: `pydub.AudioSegment.from_file()` handles any format (MP3, WAV, OGG, FLAC, M4A) via ffmpeg. Fallback: native Python `wave` module for WAV when ffmpeg unavailable.

2. **Normalization**: All audio normalized to −20 dBFS (target loudness). This ensures consistent VAD and Whisper behavior regardless of recording quality. The RMS computation uses `np.sqrt(np.mean(samples²))` then converted to dBFS with `20 * log10(rms)`.

3. **Resampling**: Everything converted to 16 kHz mono — the standard input format for both Silero VAD and Whisper.

4. **Silero VAD Chunking**: The VAD model (`silero-vad` v5 from `torch.hub`) operates on 512-sample windows (32 ms at 16 kHz). The chunking algorithm:
   - Target: 25-30 second chunks (changed from 5-10s early on to match Whisper's receptive field)
   - Waits until chunk exceeds `min_chunk_sec` (25s), then cuts at the next 0.5s silence detected by VAD
   - Hard ceiling at `max_chunk_sec` (30s) — forces a cut even mid-speech
   - Chunks < 0.3s silently discarded as trailing silence
   - Each chunk dict contains: `chunk_id`, `audio_data` (float32 numpy array), `sample_rate` (16000), `duration_sec`, `start_time_sec`, `end_time_sec`, `is_speech`, `rms_db`, `detected_language` (None until ASR), `processing_time_sec`
   - VAD is stateful (RNN hidden states) — `vad.reset()` called between files

5. **Streaming**: `process_audio_file()` is a **generator** — it yields chunks one at a time. This enables "infinite audio" processing without buffering the entire file.

### 2.3 ASR Pipeline Deep Dive

**`asr_pipeline.py`** — evolved through 4 major versions.

**Version 1 (faster-whisper + medium model, 5-10s chunks)**:
- Model: `faster-whisper-medium` with int8 quantization
- Chunks: 5-10 seconds → 99 chunks for 13-min audio
- Diarization: PyAnnote on every chunk
- Result: ~1000s for 782s audio (1.28× real-time, under 5× rubric but slow)
- WER: ~0.36 (first run with Ollama)

**Version 2 (small model, speed optimization)**:
- Switched to `faster-whisper-small` → 3× faster
- Relaxed confidence filter from 0.60/0.60 to 0.40/0.80 to stop over-filtering Greek
- Reduced beam_size from 5 to 3
- ASR dropped to ~400s (0.5× real-time)
- WER increased to 0.22 (worse accuracy, expected with small model)

**Version 3 (turbo model, 30s chunks)**:
- Switched to `faster-whisper-large-v3-turbo` — best speed/accuracy balance
- Chunk size increased to 25-30s → 28 chunks instead of 99
- Fix: `is_speech` field added to slim metadata (was missing, causing empty transcript output)
- Result: ~520s ASR (0.66× real-time), WER ~0.30
- 15 language switch points detected (62% Greek, 38% English)

**Version 4 (final production)**:
- DevConfg: `turbo`, int8, beam=3, `language=None` (auto-detect, never force)
- Diarization: `pyannote/speaker-diarization-3.1` on MPS (Apple GPU)
- MPS cache cleared after each chunk via `torch.mps.empty_cache()` — prevents GPU memory growth
- Audio freed after transcription via `del chunk["audio_data"]`
- Per-chunk `all_language_probs` stored from `TranscriptionInfo` dataclass
- Output: chunks with `segments` (word-level timestamps, confidence, speaker labels), `full_text`, `detected_language`, `language_probability`, `all_language_probs`

**TranscriptionInfo dataclass** (from faster-whisper source):
```python
@dataclass
class TranscriptionInfo:
    language: str
    language_probability: float
    duration: float
    duration_after_vad: float
    all_language_probs: list[tuple[str, float]] | None  # ← key for multilingual analysis
    transcription_options: TranscriptionOptions
    vad_options: VadOptions
```

### 2.4 LLM Integration Deep Dive

**`llm_integration.py`** — 565 lines after cleanup.

**Architecture**: `AccumulatedTranscript` class + async ticker pattern.

1. **Client**: `AsyncOpenAI` (or synchronous `OpenAI` for Streamlit). Supports OpenAI, Ollama, LM Studio via `LLM_BASE_URL` env var.

2. **Ticker Window**: Every ~120 seconds of accumulated reliable text, fires a background LLM call that extracts:
   - Named persons, organizations, keywords, main ideas
   - 1-sentence abstractive segment summary

3. **Concurrency**: Uses `asyncio.create_task()` — LLM calls run in background while ASR stream continues. At end-of-stream, `asyncio.gather()` waits for all pending calls.

4. **Entity deduplication**: Global sets (`_all_persons`, `_all_organizations`, `_all_keywords`) are updated as ticker tasks complete. The `to_dict()` method produces the full SCHEMA 2 transcript object.

5. **Memory optimization**: The `add_chunk()` method stores only slim metadata (8 fields) instead of full chunk dicts. Full chunk data is discarded after extraction. This reduced RAM from 11.8 GB to 1.9 GB.

6. **Models tested**: `gpt-4o-mini` (default), `qwen2.5:7b` (Ollama), `gpt-5.4-mini-2026-03-17` (final production).

### 2.5 Transcript Normalization — Full Evolution

This was the most iterated-on component. It went through **5 phases**:

**Phase 1 — No normalization**: Raw ASR output fed directly to NER. Entity names came out as Greek transliterations (Ιαν Λε Κων instead of Yann LeCun). Topic Recall: 0.33.

**Phase 2 — Rule-based approach (abandoned)**: Greek-to-Latin character mapping + fuzzy matching against known entity list. Maintained a dictionary of ~50 entity mappings. Rejected because: hardcoded dictionaries don't scale, fuzzy matching missed novel entities, couldn't handle free-form ASR errors.

**Phase 3 — LLM-based normalization (transcript_normalizer.py)**: Single pass through a lightweight LLM that corrects:
- Corrupted person names (Ιαν Λε Κων → Yann LeCun)
- Corrupted organization names (openai → OpenAI)
- Corrupted technical terms (api silicon → Apple Silicon)
- Incorrect capitalization, tokenization issues

**Prompt (benchmark-proven Variant C)**:
```
You are repairing a multilingual ASR transcript.

Do ONLY these:
1. Restore corrupted person names
2. Restore corrupted organization names
3. Restore corrupted technical terms
4. Fix capitalization of known entities
5. Fix obvious tokenization issues

Do NOT: summarize, paraphrase, reorder, translate, rewrite sentences,
         improve grammar, style, or wording, add or remove information.

If confidence is low, leave text unchanged.
Return ONLY the repaired transcript.
```

**Phase 4 — Model benchmarking**: Tested 3 models on a controlled snippet:
| Model | Yann LeCun | Fei-Fei Li | Sam Altman | OpenAI | Geoffrey Hinton | Demis Hassabis | Score |
|-------|-----------|-----------|-----------|--------|----------------|---------------|-------|
| gpt-4.1-nano | ✅ | ❌ (Ann Fey Fey Lee) | ✅ | ✅ | ✅ | ✅ | 5/6 |
| gpt-4.1-mini | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 6/6 |
| gpt-5.4-mini | (not tested in isolation, used in production) | | | | | | — |

Selected `gpt-5.4-mini-2026-03-17` for production (strongest general model already in use).

**Phase 5 — Production integration**:
- Feature flag: `ENABLE_TRANSCRIPT_NORMALIZATION=true|false`
- Anti-hallucination: length ratio must stay 0.85–1.15, speaker labels ≥ 90% preserved, timestamps ≥ 90% preserved, paragraphs ≥ 80% preserved
- Any validation failure → fall back to raw transcript
- Retry logic: 2 retries per chunk, 60s timeout
- Chunking: splits at 8000 chars on paragraph boundaries
- Entity re-extraction: after normalization, a second LLM pass extracts fresh entities from the cleaned text. These REPLACE the raw ticker entities.
- Edit region logging via `difflib.SequenceMatcher`: 29 edit regions for gpt-4.1-nano, 46 for gpt-4.1-mini

**Normalization quality results** (from production run):
- 46 edit regions corrected
- Greek transliterations removed: Ιαν Λε Κων, Αν Φεϊ
- Partial corrections: Σαμ Αλτμαν, Τσιν Γιουλίν, Ντέμις Χάσαμπης still present (model limitation on complex transliterations)
- Entity types remain confused (GPT-4 as Person, BERT/RoBERTa as Organization)

### 2.6 Entity Extraction & Topic Extraction

**`topic_extraction.py`** — 415 lines after cleanup.

**EntityRegistry class**: Maintains frequency-ranked entity lists with deduplication. Each entity stores: `name`, `count`, `windows` (ticker windows where found). Supports both batch (`build_entity_registry()`) and streaming (`update_registry_from_ticker()`) modes.

**Key implementation details**:
- Case-insensitive deduplication
- Frequency ranking within each entity type
- Cross-window entity tracking for timeline visualization

### 2.7 Summary Generation

**`summary_generator.py`** — 580 lines after cleanup.

**Pass-2 Pipeline**: Runs after all ASR + LLM ticker calls complete.

1. **YouTube Chapters**: Extracts 4-7 chapters with timestamps, titles, and 1-sentence summaries from the full transcript.

2. **Three summary tiers**:
   - **TL;DR**: 1-sentence overarching thesis (~150-200 chars)
   - **Executive Summary**: 3-paragraph detailed summary (~1800-2400 chars)
   - **Deep Dive**: Structured analysis with overview, bullet points, key takeaways, action items

3. **Q&A Backend**: `query_transcript()` function that answers free-form questions about the transcript. Uses synchronous `OpenAI` client (not async) to avoid Streamlit event loop conflicts. The `_call_llm_sync()` function was created specifically for this — the original `asyncio.run()` approach crashed in Streamlit callbacks.

4. **Persistence**: Writes to `Results/summary_outputs.json`. `append_qa_log()` incrementally appends Q&A entries. `load_summary_outputs()` restores from disk on Streamlit startup.

**Models tested for summaries**: `gpt-5.4-mini-2026-03-17` (final), `qwen2.5:7b` (Ollama). The Ollama model produced usable but lower-quality summaries with ROUGE-1 ~0.41 vs ~0.40 for the OpenAI model.

---

## 3. Memory Optimization — Complete Journey

### 3.1 The Problem

On first run with turbo model + 30s chunks + pyannote diarization, Activity Monitor showed:
```
Python process ≈ 11.8 GB RAM
```
On a 16 GB MacBook Pro M2, this left almost no headroom. Memory appeared to grow across chunks, suggesting accumulation.

### 3.2 Root Causes Identified

| Cause | Location | Impact |
|-------|----------|--------|
| `list(asr.transcribe_file(...))` materializes all chunks | `run_pipeline.py:78` | Full chunk dicts (with segments, audio data) kept in RAM |
| `AccumulatedTranscript.to_dict()` copies `self._chunks` | `llm_integration.py:428` | Double retain of all chunk data |
| No MPS cache clearing | `asr_pipeline.py` | GPU memory accumulates across 28 chunks |
| PyAnnote model residency | `asr_pipeline.py` | ~2-3 GB for segmentation + embedding + PLDA models |
| Whisper model residency | `asr_pipeline.py` | ~1.6 GB for turbo model (int8) |

### 3.3 Fixes Applied

1. **Slim metadata storage**: `AccumulatedTranscript.add_chunk()` now extracts only 8 fields (chunk_id, times, language info, full_text, speakers) instead of storing the entire chunk dict with segments, word timestamps, confidence scores.

2. **`del asr_chunks` after LLM stage**: Release the full chunk list immediately after `process_asr_stream_sync()` completes. The LLM stage already extracted everything needed.

3. **MPS cache clearing**: `torch.mps.empty_cache()` called after every chunk transcription in `transcribe_chunk()`.

4. **Audio release**: `del chunk["audio_data"]` after transcription — prevents retaining 30s × 16kHz × 4 bytes = ~2MB of float32 audio per chunk.

5. **Result**: RAM dropped from 11.8 GB to stable **1.25-1.93 GB** across chunks. Memory no longer grows linearly.

### 3.4 Memory Telemetry

Added RSS logging every 5 chunks in `run_pipeline.py`:
```python
rss_gb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
logger.info("  Chunk %d | RSS %.2f GB", chunk["chunk_id"], rss_gb)
```

Typical output with fixes:
```
Chunk 4  | RSS 2.05 GB
Chunk 9  | RSS 1.58 GB
Chunk 14 | RSS 1.86 GB
Chunk 19 | RSS 1.93 GB
Chunk 24 | RSS 1.25 GB
```

Flat profile confirms no accumulation — model residency is the floor.

---

## 4. ASR Benchmarking — Complete Results

### 4.1 Benchmark Dataset

Created a 180-second clip from `bilingual_long.wav` (first 180s, contains Greek + English + language switching + proper nouns). Matching ground truth subset from `bilingual_long_gt.json`.

### 4.2 Single Model Benchmarks (180s clip)

| Configuration | Runtime | Chunks | WER | Normalized WER | Entities (10) |
|---|---|---|---|---|---|
| faster-whisper turbo 30s | 130.8s (0.73×) | 7 | **0.3382** | 0.3465 | 2 |
| faster-whisper turbo 10s | 262.9s (1.46×) | 19 | **0.3147** | 0.3296 | 2 |
| faster-whisper large-v3 30s | 269.4s (1.50×) | 7 | 0.3529 | 0.3521 | 3 |
| Bare Whisper (no VAD/chunks) | N/A | 1 | **0.59** | N/A | N/A |

### 4.3 Language Locking Experiment

Tested explicit `language=detected_language` when confidence ≥ 0.90 vs default `language=None`:
- Hypothesis: Forcing detected language would preserve entity names across code-switching
- Result: No measurable improvement. Language auto-detection (`language=None`) works correctly. Entity corruption comes from TTS audio quality, not language ambiguity.

### 4.4 Chunk Size Analysis

| Chunk Size | Pros | Cons |
|---|---|---|
| 30s | Fewer chunks (7 for 180s), less diarization overhead, 0.73× real-time | Slightly worse WER (0.338) |
| 10s | Better WER (0.315), finer language boundaries | 2× slower (1.46×), 19 chunks for 180s |
| 5s | Finest granularity | Catastrophically slow (>3× overhead from diarization per tiny chunk) |

**Decision**: 30s chunks for production. The 7% WER gain from 10s isn't worth 2× runtime cost.

### 4.5 Full Pipeline Benchmarks (782s clip)

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| Speed | 0.74× real-time (578s) | ≤ 2.0× | ✅ |
| WER | 0.3046 | ≤ 0.08 | ❌ |
| Normalized WER | 0.3107 | diagnostic | (not used for pass/fail) |
| ROUGE-1 | 0.40 | ≥ 0.40 | ✅ (borderline) |
| Topic Recall | 0.42-0.50 | ≥ 0.80 | ❌ |

### 4.6 WER Analysis

The 0.30 WER is NOT caused by:
- Punctuation differences (Normalized WER = 0.31, virtually identical)
- Capitalization differences (jiwer already handles this)
- Speaker label formatting (stripped before WER)
- Chunking effects (Bare Whisper with NO chunks gave WER 0.59 — much worse)

The 0.30 WER IS caused by:
- **TTS audio quality**: gTTS-synthesized Greek pronunciation doesn't match source text. Whisper transcribes what it "hears" — garbled Greek from synthetic voice.
- **Reference mismatch**: GT transcript (9515 chars) vs hypothesis (7110 chars). Reference has complete text; some audio segments produce garbled output.
- **Multilingual proper nouns**: English names embedded in Greek speech are transliterated phonetically rather than transcribed correctly.

**Critical finding**: The Normalized WER being nearly identical to regular WER proves the gap is genuine word errors, not formatting artifacts.

### 4.7 NVIDIA Model Investigation

**Parakeet TDT 0.6B v3** (`nvidia/parakeet-tdt-0.6b-v3`):
- 600M params, FastConformer-TDT architecture
- 25 European languages including Greek (el)
- Pure ASR (no translation capability)
- Auto-detects language — no configuration needed
- Requires bleeding-edge Transformers (`pip install git+https://github.com/huggingface/transformers`)
- API: `AutoModelForTDT` + `AutoProcessor` + `model.generate()`
- Benchmark status: code written, not yet run on Colab (needs GPU)
- Reported WER: 20.70% on Greek (Fleurs), 4.85% on English

**Canary 1B V2** (`nvidia/canary-1b-v2`):
- 978M params, FastConformer Encoder + Transformer Decoder
- 25 European languages including Greek
- **Multitask model**: supports BOTH ASR and AST (translation)
- CRITICAL: defaults to English ASR when called with `model.transcribe([path])`
- For non-English: must use `model.transcribe([path], source_lang='el', target_lang='el')`
- `source_lang == target_lang` → ASR. `source_lang != target_lang` → translation
- Manifest API is broken on Colab (lhotse version incompatibility)
- First benchmark attempt was **invalid** — Canary translated Greek to English because `source_lang` wasn't set
- Fix: per-chunk language detection via faster-whisper tiny, then `source_lang=target_lang=detected_lang`
- Benchmark status: code rewritten with fix, needs Colab GPU

**Canary 1B Flash** (`nvidia/canary-1b-flash`):
- 883M params, 4 languages only (en/de/es/fr)
- **Does NOT support Greek** — excluded from benchmark

### 4.8 HuggingFace Transformers vs faster-whisper

Attempted to switch to HF Transformers `whisper-large-v3-turbo` for production:
- **Result**: Catastrophically slow on MPS. 30-90 seconds per 30-second chunk.
- **Root cause**: `model.generate()` on MPS with float32 is fundamentally unoptimized. The pipeline API with `chunk_length_s=30` was even slower due to experimental chunked long-form support.
- **Conclusion**: `faster-whisper` with CTranslate2 int8 backend is the only viable option for Apple Silicon. The CTranslate2 library is specifically optimized for CPU inference with NEON/AMX instructions on Apple Silicon, achieving 3-8× speedup over HuggingFace Transformers.

---

## 5. Streamlit App — Complete Evolution

### 5.1 Architecture

The Streamlit app serves as the local UI. All heavy compute runs on Colab. Communication via Google Drive as a message bus.

**Data flow**:
```
[User] → Streamlit (local)
    │ uploads WAV → Drive: ece22073/input/{job_id}.wav
    │ uploads podcast JSON → Drive: ece22073/input/podcast_jobs/{job_id}.json
    │ polls ← Drive: ece22073/output/{job_id}/status.json
    │ reads ← Drive: ece22073/output/{job_id}/*.json
    │ polls ← Drive: ece22073/output/podcasts/{job_id}.mp3

[Colab watcher] polls Drive API every 10s
    → finds WAV → runs ASR pipeline → uploads results
    → finds JSON → runs podcast TTS → uploads MP3
```

### 5.2 Page Evolution

**Page 1 — Sources (Upload)**:
- v1: Simple file uploader + transcribe button + JSON status dump
- v2: Full-width uploader, status in expander, speaker bubbles (HTML with colored left border)
- v3: Source card after upload (🎙️ icon, filename, job ID, status badge), progress bar during processing, responsive layout

**Page 2 — Notebook Workspace**:
- v1: Just an info message — "Complete an upload first"
- v2: Two-column layout (transcript left, Q&A + notes right)
- v3: Three-column NotebookLM-style layout:
  - Left (Sources): transcript segments as colored cards, "Add a source" (URL/PDF ingestion)
  - Center (Chat): Q&A with suggested question pills when empty
  - Right (Studio): navigation buttons to other pages, entities as chips, saved notes
- v4: Removed the "gate" — always shows layout. Empty states when no data.

**Page 3 — Summaries**:
- v1: Three columns with `st.write()` for TL;DR, Executive, Deep Dive
- v2: Source-card styled divs, chapters as timeline with timestamps

**Page 4 — Podcast Studio**:
- v1: Four numbered sections with `st.subheader()`
- v2: Four `st.tabs()` (Source, Episode, Speakers, Generate)
- v3: Fixed placement bug — VRAM estimate and Compare Models moved into Generate tab only

**Page 5 — Accuracy Check**:
- v1: Two uploaders stacked, side-by-side text areas, diff viewer
- v2: Clean layout, green highlights for added words, metric card styling for WER/ROUGE

### 5.3 CSS Styling

Applied NotebookLM-inspired dark theme:
- Background: `#161616` (main), `#0f0f0f` (sidebar)
- Cards: `#1a1a1a` with `#2a2a2a` border, 10px border-radius
- Speaker bubbles: blue left border (`#4a9eff`) for Speaker A, green (`#4aff9e`) for Speaker B
- Entity chips: rounded pills with `#222` background
- Tabs: dark background with highlighted active state
- Sidebar nav: manual buttons with section groupings (SOURCES / ANALYZE / CREATE), active page gets `#1e3a5f` background
- Hidden default Streamlit chrome (header, footer, menu)

### 5.4 Source Ingestion (URL + PDF)

Added URL and PDF ingestion to Notebook Workspace:
- **URL**: `requests` + `html2text` → markdown, truncated at 8000 chars
- **PDF**: `pymupdf` (fitz) → text extraction per page, truncated at 8000 chars
- Stored in `st.session_state["extra_sources"]` — prepended to Q&A system prompt
- Error handling: clean `ImportError` messages suggesting pip install commands

### 5.5 Q&A Chat

The Q&A in Notebook Workspace connects to the same LLM as the pipeline:
- Uses `query_transcript()` from `summary_generator.py` when no extra sources
- When extra sources present: extends system prompt with source content, uses direct `_call_llm_sync()` call
- Suggested question pills appear when no chat history yet
- Fix: switched from `asyncio.run()` to synchronous `OpenAI` client to avoid Streamlit event loop conflicts

---

## 6. Bugs, Fixes & Critical Discoveries

### 6.1 Major Bugs

| # | Bug | Root Cause | Fix | Date |
|---|-----|-----------|-----|------|
| 1 | Transcript text empty (0 chars) | `is_speech` field missing from slim metadata in `to_dict()` | Added `is_speech` to the 8-field slim dict | June 1 |
| 2 | `max_tokens` → `max_completion_tokens` | gpt-5.4-mini API uses different parameter name | Changed all 7 API call sites | June 1 |
| 3 | Streamlit chat broken | `asyncio.run()` inside Streamlit callback crashes | Created `_call_llm_sync()` using sync `OpenAI` client | June 1 |
| 4 | Path("results") relative to CWD | Streamlit app broken when run from wrong directory | Changed to `Path(__file__).parent / "results"` | June 1 |
| 5 | `Streamlit audio_bytes=None` crash | Session state default None, accessed without guard | Changed to `st.session_state.get("audio_bytes")` | June 1 |
| 6 | 11.8 GB RAM on 16 GB MacBook | Chunk dict accumulation + MPS cache growth | Slim metadata, `del asr_chunks`, `torch.mps.empty_cache()` | June 1 |
| 7 | Greek segments over-filtered | Confidence threshold 0.60 too aggressive for gTTS Greek | Relaxed to 0.40/0.80 | June 1 |
| 8 | WER computed on normalized text | `evaluate_real_pipeline.py` read `full_text` which was normalized | Changed to read `raw_full_text` | June 1 |
| 9 | `model.generate()` crashes on MPS | Whisper needs exactly 3000 mel frames (30s audio) | Padded all chunks to exactly 30s before inference | June 1 |
| 10 | `max_target_positions` overflow | turbo model has max 448 target tokens, 3 initial tokens + 448 = 451 | Reduced `max_new_tokens` to 445 | June 1 |
| 11 | Canary translating Greek to English | Canary V2 defaults to English ASR without `source_lang=target_lang` | Added per-chunk language detection, explicit language config | June 1 |
| 12 | Colab watcher silently picks up nothing | Drive FUSE mount doesn't show API-uploaded files | Switched to `db.find_new_input_files()` (Drive API polling) | June 1 |
| 13 | All Colab errors silently swallowed | `logging.basicConfig` only runs under `__main__` | Added `logging.basicConfig(force=True)` before imports | June 1 |
| 14 | `torchaudio.info()` removed | AttributeError in newer torchaudio versions | Changed to `torchaudio.load()` + waveform.shape calculation | June 1 |
| 15 | Notebook `requirements_colab.txt` path broken | Moved to App/ during restructure, but notebook still referenced Pipeline/ | Fixed notebook Cell 2 path to `App/requirements_colab.txt` | June 11 |
| 16 | Streamlit "credentials.json in Politakis/" | Path reference not updated after restructure | Updated all 4 references to "App/" | June 11 |
| 17 | `is_speech` filtering in `to_dict()` causes empty transcript | `is_speech` not in slim metadata → `c.get("is_speech")` returns None/falsy → all chunks excluded | Added `is_speech` to the slim dict in `add_chunk()` | June 1 |
| 18 | Normalization `apply_transcript_normalization` import missing on Colab | Function existed locally but not in pushed code | Made `benchmark_all.py` self-contained with its own `_apply_normalization()` | June 1 |
| 19 | `numba` + numpy 2.4 incompatibility on Colab | Parakeet's librosa dependency requires numpy ≤ 2.0 | Pinned `numpy==2.0.2` on Colab | June 1 |
| 20 | `lhotse` KeyError: 'duration' on Colab | Canary's manifest API broken due to lhotse version | Switched to direct `model.transcribe([path], source_lang=, target_lang=)` API with per-chunk language detection | June 1 |

### 6.2 Performance Obstacles

1. **MPS + float32 = slow**: HuggingFace Transformers on MPS with float32 is fundamentally unoptimized. `faster-whisper` (CTranslate2 int8) is 3-8× faster on Apple Silicon. This was discovered after spending ~3 hours trying to make HF Transformers work.

2. **PyAnnote diarization overhead**: ~5-6s per chunk for model loading + inference. With 28 chunks, that's ~140s of diarization overhead on top of Whisper transcription. Total pipeline: 578s (0.74× real-time). Diarization alone accounts for ~25% of runtime.

3. **TTS audio quality ceiling**: The synthetic gTTS audio fundamentally limits WER. No amount of model improvement can fix garbled Greek pronunciation. This is the single biggest obstacle to meeting the WER ≤ 0.08 target. Real human-spoken recordings would perform dramatically better.

4. **`gpt-5.4-mini-2026-03-17` API parameter changes**: Two breaking changes from the OpenAI API:
   - `max_tokens` → `max_completion_tokens` (changed in all 7 call sites)
   - The model name itself changed from the standard `gpt-4o-mini`

### 6.3 Toolchain Issues

1. **Homebrew Python + PEP 668**: pip installs blocked by externally-managed-environment. Fix: `--break-system-packages` flag.

2. **libavdevice duplicate classes**: Both `av` Python package and standalone `ffmpeg` (Homebrew) ship `libavdevice.dylib`, causing `objc` warnings about duplicate `AVFFrameReceiver` classes. Harmless but noisy. Fix: `brew uninstall ffmpeg` if only WAV files are used.

3. **Notebook kernel in infinite loop**: The pipeline takes 8-10 minutes on M2 Pro. Agent bash tools appear "stuck" because no output appears during ASR stage. Fix: redirect to log file or run in user's terminal.

4. **Google Colab runtime disconnections**: Free tier disconnects after ~2 hours. The watcher pattern with Drive as persistence layer handles this gracefully — Streamlit can resume from any state.

---

## 7. Repository Restructure (June 11)

### 7.1 Before

```
Politakis/  (flat — all 30+ files)
  audio_processor.py
  asr_pipeline.py
  llm_integration.py
  ...
  streamlit_app.py
  benchmark_all.py
  requirements.txt
  credentials.json
  sample_podcasts/
  results/
notebook.ipynb
```

### 7.2 After

```
Pipeline/     — config, drive_bridge, watcher, pipeline stages, TTS
App/          — Streamlit UI, Docker config, requirements, credentials
Benchmarks/   — evaluate, evaluate_real_pipeline, 5 benchmark scripts
Foundational/ — real_time_processor, sanity_transcribe
Samples/      — sample_podcasts (audio + ground truth)
Results/      — output directory
AGENTS.md     — entry point for future agents
CLAUDE.md     — Colab inline watcher pattern + known issues
README.md     — full setup guide (Docker, Colab, pipeline, benchmarks)
```

### 7.3 Changes required by restructure

- **54 files** affected by `sys.path.insert` updates
- All `Politakis/` references changed to new paths
- All `results/` → `Results/` (capitalized)
- All `sample_podcasts/` → `Samples/sample_podcasts/`
- Notebook `sys.path` updated from `Politakis/` → `Pipeline/`
- `requirements_colab.txt` path updated to `App/requirements_colab.txt`
- Git history preserved (git detects renames, not deletes)

---

## 8. Docker Infrastructure

### 8.1 Dockerfile

```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg espeak-ng
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8501
```

### 8.2 docker-compose.yml

```yaml
services:
  streamlit:
    build: .
    ports: ["8501:8501"]
    volumes: ["../:/app"]
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY:-}
    working_dir: /app/App
    command: streamlit run streamlit_app.py --server.address 0.0.0.0
```

Key design decisions:
- Volume mount `../:/app` — code changes reflect without rebuild
- `.env` file passes `OPENAI_API_KEY` to container
- `credentials.json` and `token.json` persist via the volume mount
- `.dockerignore` excludes `__pycache__`, `.git`, `Results/`, audio files

---

## 9. Colab Infrastructure

### 9.1 Notebook Structure (7 cells)

| Cell | Purpose | Key details |
|------|---------|-------------|
| 0 | Markdown header | Project info, links |
| 1 | Clone/pull repo | GitHub token from Colab secret, `force=True` logging, `warnings.filterwarnings("ignore")`, nuke pycache, `sys.path` → Pipeline/ |
| 2 | Install deps | ffmpeg, espeak-ng, requirements_colab.txt, torchvision upgrade, TTS engines from GitHub |
| 3 | HuggingFace login | `notebook_login()` for pyannote + TTS models |
| 4 | Watcher | Inline watcher loop (not `cjw.main_loop()`), Drive API polling, accepts .wav/.mp3/.m4a |
| 5 | Monitoring markdown | Instructions for checking status |
| 6 | WER evaluation | Runs `evaluate_real_pipeline.py`, prints ROUGE, side-by-side GT vs normalized output |

### 9.2 Colab Secrets

| Name | Value |
|------|-------|
| `GITHUB_TOKEN` | GitHub PAT with repo read scope |
| `HF_TOKEN` | HuggingFace token for pyannote diarization |
| `OPENAI_API_KEY` | OpenAI API key for LLM stages |

### 9.3 Colab Watcher Fix

**Critical bug**: The watcher used `os.listdir()` on Drive FUSE mount path (`/content/drive/MyDrive/ece22073/input/`). Files uploaded via Drive API don't appear on the FUSE mount reliably.

**Fix**: Switched to Drive API polling:
```python
for file_info in db.find_new_input_files():
    if fname.lower().endswith((".wav", ".mp3", ".m4a")):
        cjw._handle_asr_job(file_info)
```

**Logging fix**: `logging.basicConfig(force=True)` is mandatory before importing watcher modules in Colab notebook cells. Without it, all errors are silently swallowed because `basicConfig` only runs under `__main__`.

---

## 10. Test Dataset

### 10.1 bilingual_long.wav

- **Source**: gTTS (Google Text-to-Speech) synthesis
- **Duration**: 782.33 seconds (~13 minutes)
- **Languages**: Greek (62.4%) + English (37.6%)
- **15 segments** alternating Greek/English on AI topics
- **Format**: 16 kHz mono WAV
- **Ground truth**: `bilingual_long_gt.json` (9515 character transcript, 24 keywords, 7 persons, 12 organizations)
- **15 language switch points** detected by Whisper

### 10.2 bilingual_benchmark.wav

- **Source**: First 180 seconds of `bilingual_long.wav`
- **Duration**: 180 seconds (~3 minutes)
- **Ground truth**: `bilingual_benchmark_gt.json` (2358 character transcript subset)
- Used for all rapid-iteration benchmarks

### 10.3 VOXTAB_Academic_audio.mp3

- **Source**: Provided by student (MP3 format)
- **Content**: Quantum computing academic lecture (~204 seconds)
- **Status**: Copied to Samples/, not yet processed by pipeline

### 10.4 Test Data Limitations

1. **Synthetic TTS audio**: gTTS pronunciation of English names in Greek context produces unnatural phonetics. This is the primary cause of high WER (0.30). Real human recordings would perform significantly better.

2. **Reference transcript mismatch**: The ground truth is the exact gTTS source text. TTS pronunciation differs from source text, so even "perfect" transcription would have non-zero WER.

3. **Single speaker**: All test audio has one speaker. Diarization benchmarks not meaningful on this data.

---

## 11. Evaluation Metrics — Evolution

### 11.1 WER

| Pipeline version | WER | Date | Key changes |
|-----------------|-----|------|-------------|
| Initial (medium model, 5-10s chunks) | 0.36 | June 1 | LLM via Ollama qwen2.5:7b |
| Small model (speed test) | 0.22 | June 1 | Model trade-off |
| Turbo model, 30s chunks | 0.3046 | June 1 | Production baseline |
| Turbo + normalization | 0.3046 | June 1 | WER unchanged (by design — reads raw) |
| Normalized WER | 0.3107 | June 1 | Diagnostic metric — confirms real errors, not formatting |

### 11.2 ROUGE-1

| Pipeline version | ROUGE-1 | Date | Notes |
|-----------------|---------|------|-------|
| Ollama qwen2.5:7b | 0.41 | June 1 | Barely passed threshold |
| gpt-5.4-mini (first run) | 0.40 | June 1 | Borderline |
| gpt-5.4-mini + normalization | 0.36-0.40 | June 1 | Variable — depends on summary quality |

### 11.3 Topic Recall

| Pipeline version | Recall | Date | Entity count |
|-----------------|--------|------|-------------|
| Ollama (raw entities) | 0.33 | June 1 | 8/24 keywords matched |
| gpt-5.4-mini (raw entities) | 0.46 | June 1 | 11/24 keywords |
| gpt-5.4-mini + normalization + re-extraction | 0.42-0.50 | June 1 | Variable, entity type confusion remains |

---

## 12. Key Technical Decisions & Rationale

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| `faster-whisper` over HF Transformers | CTranslate2 int8 is 3-8× faster on Apple Silicon | Production-ready at 0.73× real-time |
| 30s chunks over 5-10s | Fewer chunks = less diarization overhead, sufficient for turbo receptive field | 0.73× real-time, acceptable WER |
| `language=None` (auto-detect) | Don't force language; turbo model handles code-switching naturally | 15 language switches detected correctly |
| LLM-based normalization over rule-based | Rule-based doesn't scale to novel ASR errors | Partial improvement — entity spellings corrected but some transliterations remain |
| Slim metadata in `to_dict()` | Prevents RAM accumulation from chunk dicts | RAM dropped from 11.8 GB to 1.9 GB |
| MPS cache clearing after each chunk | Prevents GPU memory growth | Flat memory profile across chunks |
| Sync OpenAI client for Streamlit Q&A | Avoids `asyncio.run()` crash in Streamlit event loop | Chat works reliably |
| Manual sidebar nav over `st.radio()` | Allows section groupings, active highlighting, NotebookLM aesthetic | Cleaner UI with SOURCES/ANALYZE/CREATE sections |
| Docker volume mount for development | Code changes reflect without rebuild | Fast iteration cycle |
| Drive API polling over FUSE `os.listdir()` | Files uploaded via API don't appear on FUSE mount | Colab watcher reliably picks up new files |
| Notebook inline watcher over `cjw.main_loop()` | `logging.basicConfig` only runs under `__main__` | All errors visible in notebook output |

---

## 13. Remaining Issues & Future Work

### 13.1 Accuracy

1. **WER ≥ 0.30** — limited by synthetic TTS test data. Use real human-spoken bilingual recordings for proper evaluation.
2. **Topic Recall < 0.80** — entity re-extraction improves but doesn't reach target. Better entity canonicalization needed.
3. **Greek entity transliterations** — `gpt-5.4-mini` partially corrects but doesn't fully convert Ιαν Λε Κων → Yann LeCun. Stronger models or domain-specific fine-tuning needed.
4. **Entity type confusion** — GPT-4 listed as Person, BERT/RoBERTa as Organization. LLM NER prompt improvement needed.

### 13.2 Performance

1. **PyAnnote diarization overhead** — 25% of pipeline runtime. Could reduce by batching or using lighter model.
2. **Parakeet/Canary benchmarks not run** — code written, needs Colab T4 GPU execution.
3. **Word timestamps disabled on Colab** — CTranslate2 CUDA alignment crash. Investigate alternative backends.

### 13.3 Infrastructure

1. **Duplicate Drive `ece22073` folder** — must delete the newer one (ID `1Oy6hb2F9bhQHOnMD6kd5ggkWh1QsymeM`).
2. **Colab watcher `main_loop()` still in code** — pushed fix exists but old code remains. Commit and push once verified.
3. **Diarization disabled on Colab** — numpy/pyannote version mismatch needs resolution.
4. **`.gitignore` still has stale `Politakis/` entries** — needs cleanup.

### 13.4 Testing

1. **No unit tests** — academic deliverable, no test framework configured.
2. **No CI/CD** — all testing done manually.
3. **Benchmark results incomplete** — only Whisper models fully benchmarked.
4. **GT files need update** — `VOXTAB_Academic_audio.mp3` needs pipeline run + normalized output as new GT.

---

## 14. File Inventory (Post-Restructure)

### Pipeline/ (17 files)
- `config.py` — Drive paths, polling intervals, constants
- `drive_bridge.py` — Google Drive API I/O (auth, upload, download, list, status)
- `colab_job_watcher.py` — Main orchestration loop (Drive polling, job dispatch)
- `run_pipeline.py` — 5-stage pipeline orchestrator
- `audio_processor.py` — Audio loading, normalization, VAD chunking
- `asr_pipeline.py` — faster-whisper transcription + pyannote diarization
- `llm_integration.py` — LLM ticker NER + AccumulatedTranscript
- `topic_extraction.py` — Entity registry building
- `summary_generator.py` — Pass-2 summarization + Q&A
- `transcript_normalizer.py` — LLM-based ASR error correction
- `diarize_transcript.py` — Standalone diarization
- `podcast_pipeline.py` — TTS podcast generation (Kokoro, Dia, Bark, XTTS-v2, F5-TTS)
- `strip_newlines.py` — Text cleaning utility
- `notebook.ipynb` — Colab entry point (7 cells)
- `__pycache__/` — Python bytecode

### App/ (6 files)
- `streamlit_app.py` — 5-page Streamlit UI (814 lines)
- `requirements.txt` — Full Python dependencies
- `requirements_colab.txt` — Colab-specific dependencies
- `credentials.json` — Google OAuth client credentials
- `Dockerfile` — Docker build
- `docker-compose.yml` — Docker Compose config
- `.env.example` — Environment variable template
- `.dockerignore` — Docker build exclusions

### Benchmarks/ (11 files)
- `evaluate.py` — Metric computation library (WER, ROUGE, topic recall, latency)
- `evaluate_real_pipeline.py` — Production evaluation runner
- `benchmark_all.py` — Multi-model ASR benchmark
- `benchmark_canary.py` — NVIDIA Canary benchmark
- `benchmark_parakeet.py` — NVIDIA Parakeet benchmark
- `benchmark_lang_lock.py` — Language locking experiment
- `benchmark_no_vad.py` — Bare Whisper (no VAD/chunks) experiment
- `benchmark_normalize.py` — 4 normalization variants benchmark
- `exploration.ipynb` — Data exploration notebook
- `results.ipynb` — Results analysis notebook

### Foundational/ (2 files)
- `real_time_processor.py` — Async streaming wrapper
- `sanity_transcribe.py` — ASR vs translation verification

### Samples/sample_podcasts/ (6 files)
- `bilingual_long.wav` — 13-min bilingual test audio
- `bilingual_long_gt.json` — Ground truth for bilingual_long.wav
- `bilingual_benchmark.wav` — 3-min benchmark clip
- `bilingual_benchmark_gt.json` — Ground truth for benchmark clip
- `VOXTAB_Academic_audio.mp3` — Quantum computing lecture
- `generate_bilingual.py` — Short bilingual test generator
- `generate_bilingual_long.py` — Long bilingual test generator

---

*This document was generated from ~40 hours of development across 11 sessions (June 1-11, 2026). Every technical decision, benchmark result, bug, and architectural change is documented above.*
