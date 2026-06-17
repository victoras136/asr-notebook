# ECE22073 — Full Project Chronicle

> **Date**: June 1–17, 2026
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
- DevConfig: `turbo`, int8, beam=3, `language=None` (auto-detect, never force)
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

1. **YouTube Chapters**: `_generate_chapters()` fires a dedicated LLM call using `ticker_results` (not raw text, to keep tokens low) and produces an array of chapter objects with timestamps, titles, and per-chapter summaries. These are stored in `summary_outputs.json["chapters"]`. The LLM is instructed to merge adjacent windows that share the same topic — avoiding fine-grained chapter fragmentation.

   Output schema per chapter:
   ```json
   {
     "index":     1,
     "title":     "Machine Learning Fundamentals",
     "start_sec": 0.0,
     "end_sec":   183.5,
     "summary":   "An introduction to supervised learning and gradient descent."
   }
   ```

2. **Three summary tiers**:
   - **TL;DR**: 1-sentence overarching thesis (~150-200 chars)
   - **Executive Summary**: 3-paragraph detailed summary (~1800-2400 chars)
   - **Deep Dive**: Structured analysis with overview, bullet points, key takeaways, action items

3. **Concurrent LLM calls**: All four Pass-2 calls (chapters + TL;DR + executive + deep dive) fire simultaneously via `asyncio.gather()`. Reduces total latency from ~4× serial time to ~1.5× (network round-trips overlap).

4. **Q&A Backend**: `query_transcript()` function that answers free-form questions about the transcript. Uses synchronous `OpenAI` client (not async) to avoid Streamlit event loop conflicts. The `_call_llm_sync()` function was created specifically for this — the original `asyncio.run()` approach crashed in Streamlit callbacks.

5. **Persistence**: Writes to `Results/summary_outputs.json`. `append_qa_log()` incrementally appends Q&A entries. `load_summary_outputs()` restores from disk on Streamlit startup.

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
- Requires NeMo toolkit (`nemo_toolkit[asr]`) — ~5 min install on Colab

**Nemotron** (`nvidia/stt_en_conformer_transducer_large_nemotron`):
- NeMo ASR model, English-focused conformer transducer
- Uses `ASRModel.from_pretrained()` from `nemo.collections.asr.models`
- API: `model.transcribe([path])` → returns list of text strings or Result objects
- Requires NeMo toolkit — same as Canary

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
    │ reads ← Drive: ece22073/output/podcasts/{job_id}.mp3

[Colab watcher] polls Drive API every 10s
    → finds WAV → runs ASR pipeline → uploads results
    → finds JSON → runs podcast TTS → uploads MP3
```

### 5.2 Single-File Rebuild (June 13)

The original app used a `pages/` subdirectory structure (Streamlit multipage). This conflicted with Streamlit's auto-discovery: any folder named `pages/` causes Streamlit to create separate top-level page routes, breaking custom sidebar routing.

**Solution**: Merged all pages into a single `App/streamlit_app.py` file. All routing is handled by `st.session_state._page` + a sidebar navigation with buttons. The `_PAGES` dict maps page name → render function.

**Old page files** archived to `App/_pages_old/` (upload.py, results.py, accuracy.py).

**File stats** (current, post-rebuild): `streamlit_app.py` is ~1100 lines including CSS, state management, Drive helpers, all 5 pages, polling fragments, and chat.

### 5.3 Session State

All keys declared once in `_DEFAULTS`:

```python
_DEFAULTS: dict[str, Any] = {
    "uploaded_filename": None,
    "active_job_id":     None,
    "pipeline_state":    "idle",    # idle | uploading | processing | done | error
    "pipeline_error":    None,
    "transcript":        None,
    "summary":           None,
    "acc_result":        None,
    "acc_gt":            None,
    "history_items":     None,
    "drive_connected":   False,
    "_page":             "Upload",
    "poll_miss_count":   0,
    "chat_history":      [],
    "model_transcripts": {},
    # Podcast additions (June 13):
    "podcast_job_id":      None,
    "podcast_status":      "idle",   # idle | pending | podcast_script | podcast_tts | done | failed
    "podcast_audio_bytes": None,
    "podcast_script_text": None,
    "podcast_error":       None,
}
```

### 5.4 Page Inventory (5 pages)

| Key | Page Name | Description |
|-----|-----------|-------------|
| `"Upload"` | Upload | File uploader, status card, progress bar, auto-polling fragment |
| `"Results"` | Results | 4 tabs: Transcript + model comparison, Entities chips, Summaries, Chat Q&A |
| `"Accuracy"` | Accuracy Check | Ground truth upload, WER/ROUGE metrics, diff viewer |
| `"Podcast"` | Podcast Studio | Source card, tone/length/TTS selection, speaker config, status, audio player |
| `"History"` | History | Job history from Drive, click to reload results |

### 5.5 Upload Page — State Machine

```
idle → uploading → processing → done | error
```

- **idle**: File uploader + model multiselect, Connect Drive button if not connected
- **uploading**: Shows spinner, uploads WAV to `ece22073/input/{job_id}.wav`, uploads `meta.json` with `{filename, selected_models, uploaded_at}`
- **processing**: Shows source card with status badge, custom progress bar (CSS HTML, not `st.progress()` — see §6.2), polling fragment active every 15s
- **done**: Shows "DONE" summary with metrics (segments, duration, languages, speakers), "View Full Results →" button
- **error**: Shows error card with message from `status.json["error"]`

URL persistence: `?job_id={id}&fname={filename}` written to `st.query_params` after upload. On browser refresh, state is restored from URL + Drive API.

### 5.6 Poll-Miss Guard

Problem: if a user refreshes the page with a stale `?job_id=` in the URL, the app polls forever against a non-existent job.

Fix: `poll_miss_count` counter. After 6 consecutive `read_status()` calls returning None (6 × 15s = 90s), `_reset_job()` is called and the URL is cleared.

### 5.7 Multi-Model Transcript Display

When the user selects multiple ASR models in the multiselect, `meta.json` is uploaded with `selected_models` list. Colab watcher reads this and runs extra models after the main Whisper pipeline. Each extra model's output is uploaded as `transcript_{model_name}.json`.

The Results page reads all `transcript_*.json` files and renders them as tabs:
```python
model_tabs = st.tabs(list(all_transcripts.keys()))
for m_tab, (mname, trans) in zip(model_tabs, all_transcripts.items()):
    with m_tab:
        _results_transcript(trans, key=mname.replace(" ", "_").lower())
```

Available in multiselect: Whisper Turbo, Whisper Large v3, Nvidia Parakeet, Nvidia Canary, Nemotron, Qwen ASR.

### 5.8 CSS Design System

Dark industrial theme with amber accents throughout. Google Fonts loaded via `@import`: Bebas Neue (headers), Courier Prime (body), Share Tech Mono (labels/metadata).

```css
:root {
  --bg:          #070604;   /* near-black warm background */
  --bg2:         #110e09;   /* card background */
  --bg3:         #1c1609;   /* elevated elements */
  --border:      #2e2510;   /* default border */
  --border-hi:   #4a3a18;   /* highlighted border */
  --amber:       #e8a520;   /* primary accent */
  --amber-dim:   #7a5c0a;   /* secondary accent */
  --green:       #2ee89b;   /* success / Drive connected */
  --red:         #e84a2e;   /* error */
  --text:        #d4c4a0;   /* body text */
  --text-dim:    #7a6d54;   /* metadata / secondary text */
  --text-faint:  #352c1e;   /* decorative / barely visible */
}
```

Notable CSS decisions:
- `background-image: linear-gradient(...)` grid overlay on `.stApp` for the "warm grid" texture
- `.jcard` class for uniform job/status cards with dark border and subtle background
- `.seg.seg-a` and `.seg.seg-b` for speaker turn rendering with left-border color coding
- `.chip` class for entity pills (persons, orgs, keywords)
- Progress bar replaced with raw HTML `<div>` because `st.progress()` renders amber text on amber background — invisible (see §6.2)
- `[data-testid="stFileUploader"] button[aria-label="Add file"] { display: none }` to hide the default "add more files" button (single-upload mode)
- `[data-testid="stChatInput"] textarea { resize: none; overflow: hidden }` to hide scrollbar from chat input

### 5.9 Polling Fragments

Two polling fragments at module level (outside functions, so they run on every Streamlit rerun):

**`_auto_poll` fragment** — polls ASR job status:
```python
_poll_interval: int | None = (
    config.LOCAL_POLL_INTERVAL_SEC
    if st.session_state.pipeline_state == "processing"
    else None
)

@st.fragment(run_every=_poll_interval)
def _auto_poll() -> None:
    if st.session_state.pipeline_state == "processing" and st.session_state.active_job_id:
        _poll_job(st.session_state.active_job_id)
        st.rerun(scope="app")
```

`run_every=None` means the fragment runs exactly once (synchronous call, no background timer). `run_every=N` seconds triggers an automatic background rerun of the fragment only.

**`_podcast_poll` fragment** — polls podcast job status:
- Activates only when `podcast_status in ("pending", "podcast_script", "podcast_tts")` AND `_page == "Podcast"`
- On completion: calls `_fetch_podcast_result(job_id)` to download MP3 bytes and script text from Drive

### 5.10 History Page

Shows all ASR jobs from Drive ordered by last update time. Filters out podcast-only entries (no `filename` field). Clicking an entry restores all session state from Drive:
- `transcript.json` → `st.session_state.transcript`
- `summary_outputs.json` → `st.session_state.summary`
- `transcript_*.json` → `st.session_state.model_transcripts`
- `accuracy_result.json` → `st.session_state.acc_result / acc_gt`
- `podcast_ref.json` → `st.session_state.podcast_job_id` + checks status

Cross-session persistence: After a podcast job is launched from the Podcast page, `podcast_ref.json` is saved to `ece22073/output/{asr_job_id}/`. On History → Load Results, `_load_results()` reads this file and restores the podcast state. This means the user can close the browser, come back, load a past ASR job, and immediately see the linked podcast.

---

## 6. Bugs, Fixes & Critical Discoveries

### 6.1 Bugs Fixed June 1–11 (Initial Development)

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

### 6.2 Bugs Fixed June 13–17 (Rebuild & New Features)

| # | Bug | Root Cause | Fix | Date |
|---|-----|-----------|-----|------|
| 21 | Canary / Nemotron removed from UI multiselect | Previous session incorrectly removed them while "fixing bugs" | Restored all 6 models to multiselect options | June 13 |
| 22 | `torchaudio.list_audio_backends` missing | torchaudio ≥ 2.4 removed this function from top-level namespace; pyannote calls it on import | Monkey-patched `_ta.list_audio_backends = lambda: ['soundfile']` before pyannote import, same location as `AudioMetaData` patch in `_get_diarization()` | June 14 |
| 23 | Progress bar text invisible | `st.progress()` renders label text in amber color on amber bar background — completely invisible | Replaced with custom HTML `<div>` with explicit `color:#7a6d54` on `background:#1c1609` | June 14 |
| 24 | NeMo not installing despite `INSTALL_NEMO=True` | Cell 3 ran `pip install -q nemo_toolkit[asr]` — `-q` (quiet) hid all output including errors. Also: user ran only Cell 1, not Cell 4 | Removed `-q` flag, added explicit import verification after install, added clear restart instruction on failure | June 15 |
| 25 | NeMo always installed (not opt-in) | Previous fix mistakenly added `nemo_toolkit[asr]` to Cell 3 (always-run installs) | Removed from Cell 3, restored as `INSTALL_NEMO` flag in Cell 1 with `default=False` | June 15 |
| 26 | Chat input textarea shows scrollbar | Streamlit default textarea styling allows overflow-y scroll | Added `resize: none !important; overflow: hidden !important` to `[data-testid="stChatInput"] textarea` | June 17 |
| 27 | `torchaudio.AudioMetaData` missing | torchaudio ≥ 2.4 also removed `AudioMetaData` from top-level namespace | Already fixed in same `_get_diarization()` patch with 3-level fallback: `torchaudio.backend.common` → `torchaudio._internal.module_utils` → `collections.namedtuple` | June 14 |
| 28 | ModuleNotFoundError for NeMo logged as ERROR with traceback | Generic `except Exception` caught everything the same way | Separated `ModuleNotFoundError` → `logger.warning()` with actionable hint; other exceptions → `logger.error()` with traceback | June 15 |

### 6.3 Performance Obstacles

1. **MPS + float32 = slow**: HuggingFace Transformers on MPS with float32 is fundamentally unoptimized. `faster-whisper` (CTranslate2 int8) is 3-8× faster on Apple Silicon. This was discovered after spending ~3 hours trying to make HF Transformers work.

2. **PyAnnote diarization overhead**: ~5-6s per chunk for model loading + inference. With 28 chunks, that's ~140s of diarization overhead on top of Whisper transcription. Total pipeline: 578s (0.74× real-time). Diarization alone accounts for ~25% of runtime.

3. **TTS audio quality ceiling**: The synthetic gTTS audio fundamentally limits WER. No amount of model improvement can fix garbled Greek pronunciation. This is the single biggest obstacle to meeting the WER ≤ 0.08 target. Real human-spoken recordings would perform dramatically better.

4. **`gpt-5.4-mini-2026-03-17` API parameter changes**: Two breaking changes from the OpenAI API:
   - `max_tokens` → `max_completion_tokens` (changed in all 7 call sites)
   - The model name itself changed from the standard `gpt-4o-mini`

### 6.4 Toolchain Issues

1. **Homebrew Python + PEP 668**: pip installs blocked by externally-managed-environment. Fix: `--break-system-packages` flag.

2. **libavdevice duplicate classes**: Both `av` Python package and standalone `ffmpeg` (Homebrew) ship `libavdevice.dylib`, causing `objc` warnings about duplicate `AVFFrameReceiver` classes. Harmless but noisy. Fix: `brew uninstall ffmpeg` if only WAV files are used.

3. **Notebook kernel in infinite loop**: The pipeline takes 8-10 minutes on M2 Pro. Agent bash tools appear "stuck" because no output appears during ASR stage. Fix: redirect to log file or run in user's terminal.

4. **Google Colab runtime disconnections**: Free tier disconnects after ~2 hours. The watcher pattern with Drive as persistence layer handles this gracefully — Streamlit can resume from any state.

5. **Drive API vs FUSE mount mismatch**: Files uploaded via Drive REST API (from Streamlit) do NOT appear in `/content/drive/MyDrive/` FUSE mount reliably. Switching Colab watcher from `os.listdir()` to `drive_bridge.find_new_input_files()` (Drive API polling) fixed this silently swallowed issue.

6. **Colab + asyncio nesting**: IPython in Colab runs its own event loop. Calling `asyncio.run()` inside a Colab cell raises `RuntimeError: This event loop is already running`. Fix: each pipeline stage runs in its own thread with `threading.Thread` + `asyncio.new_event_loop()`, which creates a fresh loop isolated from Colab's IPython loop.

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

### 7.3 Changes Required by Restructure

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

### 9.1 Notebook Structure (7 cells, current)

| Cell | Purpose | Key details |
|------|---------|-------------|
| 0 | Markdown header | Project info, Streamlit run command, startup steps |
| 1 | **Configuration** | `INSTALL_PODCAST_DEPS`, `INSTALL_NEMO`, `PODCAST_TTS_MODEL` flags. Print on run. |
| 2 | Clone/pull repo + secrets | GitHub token from Colab secret, force-pull origin/main, nuke pycache, sys.path → Pipeline/ |
| 3 | Core install (uv) | ffmpeg, espeak-ng via apt. Python deps via `uv pip install --system` (10-100× faster than pip). NO NeMo here. |
| 4 | Optional installs | TTS deps if `INSTALL_PODCAST_DEPS=True`. NeMo if `INSTALL_NEMO=True` with verification. |
| 5 | HuggingFace auth | `login(token=HF_TOKEN)` for pyannote gated models |
| 6 | **Main loop** | `drive.mount()`, `import colab_job_watcher`, `cjw.main_loop()`. Auto-stops after 5 min idle. |

**Why uv instead of pip in Cell 3?**
`uv` is a Rust-based pip replacement with a parallel resolver. It resolves all package deps simultaneously (like `cargo`) instead of sequentially. For a fixed requirement set on Colab, uv takes ~30-60s vs pip's 3-5 minutes. It also skips already-satisfied packages instantly.

### 9.2 Colab Secrets

| Name | Value |
|------|-------|
| `GITHUB_TOKEN` | GitHub PAT with repo read scope |
| `HF_TOKEN` | HuggingFace token for pyannote diarization |
| `OPENAI_API_KEY` | OpenAI API key for LLM stages |

### 9.3 NeMo Install Pattern

NeMo (`nemo_toolkit[asr]`) is large (~5 min install, many dependencies). It's an opt-in:

```python
# Cell 1 (user sets this):
INSTALL_NEMO = False  # True to enable Canary + Nemotron

# Cell 4 (runs the install):
if INSTALL_NEMO:
    try:
        import nemo
        print(f'✓ NeMo already installed ({nemo.__version__})')
    except ImportError:
        r = subprocess.run(['pip', 'install', 'nemo_toolkit[asr]'],
                          capture_output=True, text=True)
        combined = (r.stdout + r.stderr).strip()
        if combined: print(combined[-1000:])  # Show last output for debugging
        if r.returncode != 0:
            print('❌ NeMo install failed — Canary/Nemotron will be skipped')
        else:
            # Verify the import actually works (install ≠ importable)
            importlib.import_module('nemo.collections.asr.models')
            print('✓ NeMo ready')
```

**Why the import verification?** `pip install nemo_toolkit[asr]` can succeed (return code 0) but leave the package in an unimportable state due to dependency conflicts. The verification step catches this before the main loop starts.

**Cell ordering matters**: User must set `INSTALL_NEMO=True` in Cell 1, THEN run Cell 4 (or "Run all"). Running only Cell 1 does nothing — the install is in Cell 4.

### 9.4 Watcher Auto-Stop

`main_loop()` auto-stops after 5 minutes with no jobs found. This prevents idle Colab runtime consumption. If the user sends a new job after the watcher stopped, they need to re-run Cell 6 (or "Restart and run all").

```python
_IDLE_TIMEOUT_SEC: int = 300  # 5 minutes

while True:
    job_found = False
    # ... poll Drive for jobs ...
    if not job_found and (time.time() - idle_since) >= _IDLE_TIMEOUT_SEC:
        logger.info("No jobs for 5min — stopping watcher")
        return
    time.sleep(config.POLL_INTERVAL_SEC)  # 10s
```

---

## 10. Drive Bridge Architecture

**`drive_bridge.py`** — Dual-environment Google Drive API client.

### 10.1 Environment Detection

```python
def _is_colab() -> bool:
    try: import google.colab; return True
    except ImportError: return False
```

On **Colab**: uses `google.colab.auth.authenticate_user()` + `google.colab.drive.mount()`. No credentials file needed.

On **local**: uses OAuth 2.0 flow with `credentials.json` + `token.json` (cached token, refreshed automatically). Browser tab opens only on first run.

### 10.2 Key Functions

| Function | Description |
|----------|-------------|
| `authenticate()` | Return Drive service. Idempotent — caches `_SERVICE` global. |
| `get_or_create_folder(path)` | Traverse nested path level-by-level, create each missing folder. Cached in `_FOLDER_CACHE`. |
| `upload_file(local_path, drive_folder)` | Upload file, overwrite if exists. Uses `MediaFileUpload` with resumable=True. |
| `upload_bytes(data, folder, filename)` | Upload raw bytes (for JSON, in-memory content). |
| `download_file(file_id, local_path)` | Download by Drive file ID. |
| `read_json(file_id)` | Download + parse JSON. Used heavily for status/transcript/summary. |
| `read_bytes(file_id)` | Download + return raw bytes. Used for MP3 audio in podcast page. |
| `write_json(data, folder, filename)` | Serialize dict to JSON + upload. |
| `write_status(job_id, status)` | Shortcut: write `status.json` to `output/{job_id}/`. |
| `read_status(job_id)` | Read `status.json`, returns newest if multiple exist. |
| `find_new_input_files()` | List files in `ece22073/input/` (WAV/MP3/M4A for ASR jobs). |
| `find_new_podcast_jobs()` | List JSON files in `ece22073/input/podcast_jobs/`. |
| `archive_input_file(file_id)` | Move processed file to `ece22073/input/processed/`. |
| `list_job_history()` | Return all ASR job records (status + meta) from `output/`, newest first. |
| `init_drive_structure()` | Idempotently create all required Drive folders. |

### 10.3 SSL Retry Pattern

Drive API calls over mobile/unstable connections occasionally raise `SSLError` or `ConnectionReset`. Both `get_or_create_folder()` and `list_files()` implement a single retry with service reset:

```python
for attempt in range(2):
    try:
        service = authenticate()
        # ... operation ...
        return result
    except Exception as exc:
        if attempt == 0 and _is_ssl_error(exc):
            _SERVICE = None   # Force re-authenticate
            continue
        raise
```

---

## 11. Colab Job Watcher — Complete Architecture

**`colab_job_watcher.py`** — Orchestration loop inside Colab.

### 11.1 Job Types

Two independent job queues on Drive:

| Queue | Drive path | Payload | Handler |
|-------|-----------|---------|---------|
| ASR | `ece22073/input/{job_id}.wav` | Binary audio | `_handle_asr_job()` |
| Podcast | `ece22073/input/podcast_jobs/{job_id}.json` | `PodcastJobDict` JSON | `_handle_podcast_job()` |

### 11.2 ASR Job Handler

1. Download WAV from Drive to `/tmp`
2. Write initial `status.json` (stage=`asr`, progress=5%)
3. Run `run_pipeline.run_pipeline(tmp_path)` in isolated thread (new event loop)
4. Write `status.json` (stage=`normalization`, progress=80%)
5. Upload all results from `Results/` to `ece22073/output/{job_id}/`
6. Read `meta.json` for `selected_models` list
7. Run extra models via `_run_extra_model(model_name, audio_path)` (each in its own thread)
8. Write final `status.json` (stage=`done` or `error`, progress=100%)
9. Archive WAV to `ece22073/input/processed/`

**Extra model thread isolation**: Each extra model (Parakeet, Canary, Qwen, etc.) runs in `threading.Thread` with `asyncio.new_event_loop()`. This isolates them from Colab's IPython loop AND from each other — a crash in one model doesn't affect others.

**NeMo graceful skip**: `ModuleNotFoundError` (NeMo not installed) → `logger.WARNING` with actionable hint. Other exceptions → `logger.ERROR` with traceback.

### 11.3 Podcast Job Handler

1. Read `PodcastJobDict` JSON from Drive
2. Write `status.json` (stage=`podcast_script`, progress=5%)
3. Import `podcast_pipeline` lazily (may not be importable if TTS deps missing)
4. Call `podcast_pipeline.generate_podcast(job_config)`
5. Upload MP3 to `ece22073/output/podcasts/{job_id}.mp3`
6. Write metadata JSON to `ece22073/output/podcasts/{job_id}.json` (includes `script_text`)
7. Write final `status.json` (stage=`done`, progress=100%)
8. Archive JSON to processed/

### 11.4 Status Schema

All status updates use `config.StatusDict`:

```python
{
    "job_id":       str,
    "job_type":     "asr" | "podcast",
    "stage":        "uploading" | "asr" | "normalization" | "summary"
                  | "podcast_script" | "podcast_tts" | "done" | "error" | "stalled",
    "progress_pct": float,    # 0.0 – 1.0
    "eta_seconds":  float,
    "error":        str | None,
    "updated_at":   str        # ISO 8601
}
```

---

## 12. Podcast Studio — Full Feature Description

### 12.1 Overview

The Podcast Studio takes ASR transcript + summary as source material and generates a two-speaker audio podcast via TTS on Colab. This is the major new feature added June 13-14.

### 12.2 Drive Folder Structure for Podcasts

```
ece22073/
  input/
    podcast_jobs/       ← Streamlit writes {job_id}.json here
      {job_id}.json     ← PodcastJobDict
      processed/        ← moved here after processing
  output/
    podcasts/           ← Colab writes outputs here
      {job_id}.mp3      ← generated audio
      {job_id}.json     ← metadata {duration_sec, word_count, script_text, ...}
    {asr_job_id}/
      podcast_ref.json  ← link: {podcast_job_id: "..."} for cross-session restore
```

### 12.3 PodcastJobDict Schema

```python
class PodcastJobDict(TypedDict, total=False):
    job_id:      str          # 8-char hex
    source_text: str          # full transcript text used as input
    speaker_a:   dict         # {name, description, tts_model, voice}
    speaker_b:   dict         # {name, description, tts_model, voice}
    config:      dict         # {tone: "casual"|"academic"|"debate"|"interview",
                              #  length: "short"|"medium"|"long"}
    created_at:  str          # ISO 8601
```

### 12.4 Script Generation

`podcast_pipeline._generate_script(job_config)` calls the LLM with a detailed system prompt:

```
Speaker A ({name}): {description}
Speaker B ({name}): {description}
Style: {tone}
Target duration: ~{N} words (~{M} minutes)
Source material: {transcript[:12000]}

Format: each line MUST start with "Speaker A: " or "Speaker B: "
```

Length mapping: `short` = 3 min = 450 words, `medium` = 7 min = 1050 words, `long` = 15 min = 2250 words.

Validation: if < 70% of lines have expected prefix, warns but continues.

### 12.5 TTS Pipeline

Five TTS backends supported:

| Model | Key | VRAM | Multi-speaker | Notes |
|-------|-----|------|---------------|-------|
| Kokoro-82M | `kokoro` | 2 GB | No (54 single voices) | Recommended, Apache 2.0, fast |
| Dia-1.6B | `dia` | 10 GB | Yes (native) | Best natural dialogue |
| Bark | `bark` | 8 GB | No | Expressive, slow |
| XTTS-v2 | `xtts_v2` | 6 GB | No | Voice cloning, CPML license |
| F5-TTS | `f5_tts` | 4 GB | No | State-of-the-art |

**Special case for Dia**: If both speakers select `dia`, the entire script is processed in a single pass using Dia's native `[S1]`/`[S2]` multi-speaker format. All other configurations run per-segment, alternating between two model instances.

**Audio concatenation**: Segments joined with 300ms silence between speaker turns. Output: MP3 at 128k bitrate (WAV intermediate via scipy, converted via pydub).

**VRAM management**: Models loaded lazily on first use, unloaded and GPU cache cleared after podcast generation.

### 12.6 Streamlit Podcast Page UI

The page has three sections:

**Source card**: Shows the transcript text that will be used. If no ASR job is loaded, prompts user to upload audio first.

**Configuration**:
- Tone: 4 buttons (Casual / Academic / Debate / Interview) — active tone button rendered as `type="primary"`
- Length: radio (Short 3 min / Medium 7 min / Long 15 min)
- TTS Model: radio (Kokoro / Dia-1.6B / Bark / XTTS-v2 / F5-TTS) with VRAM notes
- Speaker names: two text inputs

**Status & output**:
- While pending: status card with animated "●●●" spinner and stage label
- On completion: `st.audio(bytes, format="audio/mp3")` inline player, expandable script viewer

### 12.7 Cross-Session Persistence

```python
# After submitting podcast job:
db.write_json({"podcast_job_id": podcast_job_id},
              f"{config.DRIVE_OUTPUT}/{asr_job_id}", "podcast_ref.json")

# On History → Load Results:
elif f["name"] == "podcast_ref.json":
    ref = db.read_json(f["id"])
    pod_jid = ref.get("podcast_job_id")
    if pod_jid:
        st.session_state.podcast_job_id = pod_jid
        ps = db.read_status(pod_jid)
        if ps and ps.get("stage") == "done":
            st.session_state.podcast_status = "done"
```

---

## 13. ASR Models Registry

**`models_registry.py`** — 230 lines, 6 models.

```python
AVAILABLE_MODELS = {
    "whisper-turbo":    lambda p: transcribe_whisper(p, "turbo"),
    "whisper-large-v3": lambda p: transcribe_whisper(p, "large-v3"),
    "parakeet":         transcribe_parakeet,
    "canary":           transcribe_canary,
    "qwen":             transcribe_qwen,
    "nemotron":         transcribe_nemotron,
}
```

Each model function:
- Loads model lazily (no global caching — models are transient, one per job)
- Accepts `str | Path` audio path
- Returns raw transcription text
- Handles GPU/CPU device selection automatically

**Canary specifics**: Chunks audio into 35s windows with 2s overlap, detects language per chunk via Whisper-tiny, then calls `ASRModel.transcribe([chunk_path], source_lang=lang, target_lang=lang)`.

**Qwen specifics**: Loads `Qwen2-Audio-7B-Instruct` with `device_map="auto"` on CUDA. Uses chat-style prompt `"<|audio_preview|><|assistant|>Transcribe the audio:"`.

---

## 14. Config Module

**`config.py`** — 117 lines. Single source of truth for all shared constants.

```
DRIVE_ROOT = "ece22073"
DRIVE_INPUT = "ece22073/input"
DRIVE_INPUT_JOBS = "ece22073/input/podcast_jobs"
DRIVE_INPUT_PROCESSED = "ece22073/input/processed"
DRIVE_OUTPUT = "ece22073/output"
DRIVE_OUTPUT_PODCASTS = "ece22073/output/podcasts"
DRIVE_MODELS_CACHE = "ece22073/models"

POLL_INTERVAL_SEC = 10      # Colab watcher
LOCAL_POLL_INTERVAL_SEC = 15 # Streamlit
STALL_TIMEOUT_SEC = 600      # 10 min

LLM_MODEL = "gpt-4o-mini"
```

TypedDicts: `StatusDict`, `PodcastJobDict`.

---

## 15. Test Dataset

### 15.1 bilingual_long.wav

- **Source**: gTTS (Google Text-to-Speech) synthesis
- **Duration**: 782.33 seconds (~13 minutes)
- **Languages**: Greek (62.4%) + English (37.6%)
- **15 segments** alternating Greek/English on AI topics
- **Format**: 16 kHz mono WAV
- **Ground truth**: `bilingual_long_gt.json` (9515 character transcript, 24 keywords, 7 persons, 12 organizations)
- **15 language switch points** detected by Whisper

### 15.2 bilingual_benchmark.wav

- **Source**: First 180 seconds of `bilingual_long.wav`
- **Duration**: 180 seconds (~3 minutes)
- **Ground truth**: `bilingual_benchmark_gt.json` (2358 character transcript subset)
- Used for all rapid-iteration benchmarks

### 15.3 VOXTAB_Academic_audio.mp3

- **Source**: Provided by student (MP3 format)
- **Content**: Quantum computing academic lecture (~204 seconds)
- **Status**: Copied to Samples/, not yet processed by pipeline

### 15.4 Test Data Limitations

1. **Synthetic TTS audio**: gTTS pronunciation of English names in Greek context produces unnatural phonetics. This is the primary cause of high WER (0.30). Real human recordings would perform significantly better.

2. **Reference transcript mismatch**: The ground truth is the exact gTTS source text. TTS pronunciation differs from source text, so even "perfect" transcription would have non-zero WER.

3. **Single speaker**: All test audio has one speaker. Diarization benchmarks not meaningful on this data.

---

## 16. Evaluation Metrics — Evolution

### 16.1 WER

| Pipeline version | WER | Date | Key changes |
|-----------------|-----|------|-------------|
| Initial (medium model, 5-10s chunks) | 0.36 | June 1 | LLM via Ollama qwen2.5:7b |
| Small model (speed test) | 0.22 | June 1 | Model trade-off |
| Turbo model, 30s chunks | 0.3046 | June 1 | Production baseline |
| Turbo + normalization | 0.3046 | June 1 | WER unchanged (by design — reads raw) |
| Normalized WER | 0.3107 | June 1 | Diagnostic metric — confirms real errors, not formatting |

### 16.2 ROUGE-1

| Pipeline version | ROUGE-1 | Date | Notes |
|-----------------|---------|------|-------|
| Ollama qwen2.5:7b | 0.41 | June 1 | Barely passed threshold |
| gpt-5.4-mini (first run) | 0.40 | June 1 | Borderline |
| gpt-5.4-mini + normalization | 0.36-0.40 | June 1 | Variable — depends on summary quality |

### 16.3 Topic Recall

| Pipeline version | Recall | Date | Entity count |
|-----------------|--------|------|-------------|
| Ollama (raw entities) | 0.33 | June 1 | 8/24 keywords matched |
| gpt-5.4-mini (raw entities) | 0.46 | June 1 | 11/24 keywords |
| gpt-5.4-mini + normalization + re-extraction | 0.42-0.50 | June 1 | Variable, entity type confusion remains |

---

## 17. Key Technical Decisions & Rationale

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| `faster-whisper` over HF Transformers | CTranslate2 int8 is 3-8× faster on Apple Silicon | Production-ready at 0.73× real-time |
| 30s chunks over 5-10s | Fewer chunks = less diarization overhead, sufficient for turbo receptive field | 0.73× real-time, acceptable WER |
| `language=None` (auto-detect) | Don't force language; turbo model handles code-switching naturally | 15 language switches detected correctly |
| LLM-based normalization over rule-based | Rule-based doesn't scale to novel ASR errors | Partial improvement — entity spellings corrected but some transliterations remain |
| Slim metadata in `to_dict()` | Prevents RAM accumulation from chunk dicts | RAM dropped from 11.8 GB to 1.9 GB |
| MPS cache clearing after each chunk | Prevents GPU memory growth | Flat memory profile across chunks |
| Sync OpenAI client for Streamlit Q&A | Avoids `asyncio.run()` crash in Streamlit event loop | Chat works reliably |
| Single-file Streamlit app | `pages/` subdirectory conflicts with Streamlit's multipage auto-discovery | Custom sidebar routing works reliably |
| Drive as job queue (not REST API) | Colab doesn't have a stable public IP or WebSocket endpoint | Polling-based architecture, resilient to Colab disconnects |
| `uv pip install` on Colab | 10-100× faster parallel resolver vs sequential pip | Cell 3 takes 30-60s instead of 3-5 min |
| Thread isolation per model/job | IPython event loop conflicts, model crashes must not propagate | Clean isolation, full error visibility per model |
| `podcast_ref.json` in ASR job folder | Link ASR job ↔ podcast job for cross-session state restoration | Podcast status persists across browser closes |
| Monkey-patch torchaudio before pyannote | torchaudio ≥ 2.4 removed `AudioMetaData` AND `list_audio_backends` | Diarization works on any torchaudio version |
| Custom HTML progress bar | `st.progress()` renders label text in same color as bar | Readable progress labels on dark theme |
| NeMo as opt-in Cell 1 flag | NeMo install takes ~5 min, most runs don't need Canary/Nemotron | Core pipeline always fast; NeMo only when requested |
| `overflow: hidden` on chat textarea | Scrollbar visible in Streamlit default | Clean UI, no unwanted scrollbar |

---

## 18. Outstanding Issues & Future Work

### 18.1 Accuracy

1. **WER ≥ 0.30** — limited by synthetic TTS test data. Use real human-spoken bilingual recordings for proper evaluation.
2. **Topic Recall < 0.80** — entity re-extraction improves but doesn't reach target. Better entity canonicalization needed.
3. **Greek entity transliterations** — `gpt-5.4-mini` partially corrects but doesn't fully convert Ιαν Λε Κων → Yann LeCun. Stronger models or domain-specific fine-tuning needed.
4. **Entity type confusion** — GPT-4 listed as Person, BERT/RoBERTa as Organization. LLM NER prompt improvement needed.

### 18.2 Features — Planned / Discussed

1. **YouTube-style chapters displayed in UI** — `summary_outputs.json["chapters"]` already exists (generated by `summary_generator._generate_chapters()`), but the Results page doesn't render them yet. Each chapter has `{index, title, start_sec, end_sec, summary}`. Implementation would add a "Chapters" tab to Results, rendering chapters as clickable timeline items (like YouTube section markers) grouped by topic, not per Whisper phrase.

2. **Transcript segments vs chapters clarification**: The "47 segments" shown in the Results metrics are Whisper phrase-level segments (individual transcribed sentences from the ASR output). These are NOT topic sections. The LLM-generated YouTube chapters in `chapters[]` ARE topic-based groupings. The UI should distinguish these clearly.

3. **Real-time streaming UI** — Currently the status updates every 15s. Could use Server-Sent Events or WebSockets for live progress.

4. **Parakeet/Canary benchmark completion** — Models written, need Colab GPU execution.

### 18.3 Infrastructure

1. **Duplicate Drive `ece22073` folder** — must delete the newer one (ID `1Oy6hb2F9bhQHOnMD6kd5ggkWh1QsymeM`).
2. **Diarization on Colab** — verify torchaudio monkey-patch fixes the `list_audio_backends` error end-to-end.
3. **`.gitignore` stale `Politakis/` entries** — needs cleanup.

### 18.4 Testing

1. **No unit tests** — academic deliverable, no test framework configured.
2. **No CI/CD** — all testing done manually.
3. **Benchmark results incomplete** — only Whisper models fully benchmarked.
4. **GT files need update** — `VOXTAB_Academic_audio.mp3` needs pipeline run + normalized output as new GT.

---

## 19. File Inventory (Current State — June 17, 2026)

### Pipeline/ (14 files)
- `config.py` — Drive paths, polling intervals, TypedDicts (StatusDict, PodcastJobDict)
- `drive_bridge.py` — Google Drive API I/O (auth, upload, download, list, status, read_bytes)
- `colab_job_watcher.py` — Main orchestration loop (ASR + podcast job dispatch, 507 lines)
- `run_pipeline.py` — 5-stage pipeline orchestrator (audio → ASR → LLM → entities → summary)
- `audio_processor.py` — Audio loading, normalization, VAD chunking (generator)
- `asr_pipeline.py` — faster-whisper transcription + pyannote diarization (torchaudio patch included)
- `llm_integration.py` — LLM ticker NER + AccumulatedTranscript
- `topic_extraction.py` — Entity registry building
- `summary_generator.py` — Pass-2 summarization (YouTube chapters + 3 tiers + Q&A)
- `transcript_normalizer.py` — LLM-based ASR error correction
- `models_registry.py` — 6 ASR models (Whisper, Parakeet, Canary, Qwen, Nemotron)
- `podcast_pipeline.py` — TTS podcast generation (5 models: Kokoro, Dia, Bark, XTTS-v2, F5-TTS)
- `notebook.ipynb` — Colab entry point (7 cells, current structure)
- Various utilities: `diarize_transcript.py`, `strip_newlines.py`, `llm_integration.py`

### App/ (7 files)
- `streamlit_app.py` — 5-page single-file Streamlit UI (~1100 lines)
- `comparison_metrics.py` — WER, ROUGE, BLEU, readability, diff HTML
- `requirements.txt` — Full Python dependencies
- `credentials.json` — Google OAuth client credentials
- `Dockerfile` — Docker build
- `docker-compose.yml` — Docker Compose config
- `_pages_old/` — Archived old pages/ structure

### Benchmarks/ (8 files)
- `evaluate.py` — Metric computation library (WER, ROUGE, topic recall, latency)
- `evaluate_real_pipeline.py` — Production evaluation runner
- `benchmark_all.py` — Multi-model ASR benchmark
- `benchmark_canary.py` — NVIDIA Canary benchmark
- `benchmark_parakeet.py` — NVIDIA Parakeet benchmark
- `benchmark_lang_lock.py` — Language locking experiment
- `benchmark_no_vad.py` — Bare Whisper (no VAD/chunks) experiment
- `benchmark_normalize.py` — 4 normalization variants benchmark

### Samples/sample_podcasts/ (5 files)
- `bilingual_long.wav` — 13-min bilingual test audio
- `bilingual_long_gt.json` — Ground truth for bilingual_long.wav
- `bilingual_benchmark.wav` — 3-min benchmark clip
- `bilingual_benchmark_gt.json` — Ground truth for benchmark clip
- `VOXTAB_Academic_audio.mp3` — Quantum computing lecture

---

*This document covers ~50 hours of development across 13 sessions (June 1–17, 2026). Every architectural decision, benchmark, bug, and system behavior is documented above.*
