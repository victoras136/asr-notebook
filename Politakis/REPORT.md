# PROJECT 8: Multilingual Podcast Summarizer — Technical Report

**Student:** ΠΟΛΙΤΑΚΗΣ ΒΙΚΤΩΡ (ΑΜ: 9093202200073)  
**Course:** ECE22073  
**Supervisor:** Παναγιώτης Ζέρβας  
**Date:** May 2026  

---

## 1. System Overview

This project implements a real-time end-to-end podcast processing pipeline that transcribes multilingual audio, extracts named entities and key topics, and generates hierarchical summaries at multiple levels of detail. The system handles long-form audio (tested on 10+ minutes), supports multiple languages simultaneously (Greek and English demonstrated), and integrates speaker diarization to attribute speech to individual speakers.

The pipeline is structured in five sequential stages:

```
audio_processor.py → asr_pipeline.py → llm_integration.py → topic_extraction.py → summary_generator.py
```

A sixth module, `real_time_processor.py`, wraps these stages into a streaming async coroutine that emits live events as each chunk is processed.

---

## 2. Audio Processing Pipeline (Section 1 — 20 pts)

### 2.1 Audio Loading and Normalization

`audio_processor.py` uses `pydub` (wrapping ffmpeg) to ingest any audio format (MP3, WAV, OGG, FLAC, M4A). All audio is normalized to −20 dBFS loudness and resampled to 16 kHz mono PCM — the format expected by both Silero VAD and Whisper. Normalization ensures consistent behavior across podcasts recorded at different loudness levels.

### 2.2 Voice Activity Detection

Silero VAD v5 is used for real-time frame-level speech detection. The model operates on 512-sample frames (32 ms at 16 kHz) and outputs a speech probability per frame. Frames with probability ≥ 0.5 are classified as speech. Silero was chosen because it runs in under 1 ms per frame on CPU, is torch-based (MPS-acceleratable), and significantly outperforms energy-based VAD on noisy recordings.

### 2.3 VAD-Aware Chunking Strategy

The chunker accumulates frames until the chunk is ≥ 5 seconds, then cuts at the next 0.5-second silence detected by Silero. A hard ceiling at 10 seconds prevents unbounded chunks. This "natural boundary" splitting:
- Keeps Whisper within its optimal acoustic window
- Avoids cutting words mid-utterance (which causes hallucinations)
- Produces ~6–9 second chunks in practice

**Example output:**
```json
{
  "chunk_id": 0, "start_time_sec": 0.0, "end_time_sec": 7.42,
  "duration_sec": 7.42, "is_speech": true, "rms_db": -18.3
}
```

---

## 3. Speech Recognition (Section 2 — 25 pts)

### 3.1 Whisper via faster-whisper

`faster-whisper` (CTranslate2 backend) runs Whisper medium with int8 quantization. CTranslate2 provides approximately 3× speedup over vanilla Whisper through:
- Weight quantization (int8)
- Optimized CPU kernels (NEON/AMX on Apple Silicon)
- Beam search with early stopping

No language hint is passed — Whisper performs zero-shot language detection per chunk, which is essential for code-switching multilingual content (Greek ↔ English).

### 3.2 Confidence Filtering

Each Whisper segment carries an `avg_logprob` score. Segments where `exp(avg_logprob) < 0.60` or `no_speech_prob > 0.60` are flagged as unreliable and excluded from `full_text`. This eliminates hallucinated text on silent/noisy segments.

### 3.3 Speaker Diarization

`pyannote.audio` (v3.1, speaker-diarization-3.1 model) is used for per-chunk speaker diarization. Each chunk's waveform is passed to pyannote as a `{waveform, sample_rate}` dict. Speaker labels (Speaker A, Speaker B, …) are assigned to Whisper segments via maximum temporal overlap. The pipeline falls back gracefully if pyannote is unavailable (HuggingFace token not set).

### 3.4 Word-Level Timestamps

`word_timestamps=True` is set in the Whisper call, providing start/end times for every word. This enables precise chapter generation and allows downstream consumers to seek to specific moments in the audio.

**Sample chunk output:**
```json
{
  "chunk_id": 3, "detected_language": "el", "language_probability": 0.96,
  "segments": [{"text": "Η τεχνητή νοημοσύνη αλλάζει...", "confidence": 0.87, "speaker": "Speaker A"}],
  "full_text": "[Speaker A]: Η τεχνητή νοημοσύνη αλλάζει την εκπαίδευση.",
  "speakers_detected": ["Speaker A"], "processing_time_sec": 0.73
}
```

---

## 4. Topic and Content Extraction (Section 3 — 25 pts)

### 4.1 Pass-1 Live Ticker

Every 120 seconds of accumulated speech triggers an async LLM call (GPT-4o-mini via OpenAI API, or any Ollama-compatible local model). The call is structured as a JSON-extraction task:

**System prompt instructs the LLM to return:**
```json
{
  "persons":         ["Geoffrey Hinton", "Sam Altman"],
  "organizations":   ["OpenAI", "Google DeepMind"],
  "keywords":        ["LLM", "AI safety", "transformer"],
  "main_ideas":      ["AI is transforming education"],
  "segment_summary": "One-sentence abstractive summary"
}
```

Temperature is set to 0.0 for deterministic extraction. The prompt explicitly requests exhaustive extraction ("include EVERY person and organisation") to push recall above 80%.

### 4.2 AccumulatedTranscript

`llm_integration.AccumulatedTranscript` is a stateful class that:
- Accepts ASR chunks one at a time via `add_chunk()`
- Fires async ticker tasks (non-blocking, via `asyncio.create_task()`)
- Returns the full SCHEMA 2 dict on `to_dict()`

This architecture allows the UI to remain responsive while LLM calls complete in the background.

### 4.3 Entity Registry

`topic_extraction.EntityRegistry` maintains per-entity mention counts across all ticker windows. Entities from different windows are normalized (title-cased, edge punctuation stripped) and merged by canonical name. Entities are sorted by mention frequency for the final output, ensuring the most salient names appear first in summaries and UI chips.

---

## 5. Summary Generation (Section 4 — 15 pts)

### 5.1 Three Summary Levels

Pass-2 fires four concurrent LLM calls on EOF:

| Level | Output | Max Tokens |
|-------|--------|-----------|
| TL;DR | 1 sentence, max 30 words | 100 |
| Executive Summary | 3 paragraphs (context, arguments, conclusions) | 600 |
| Deep Dive | JSON: overview + bullet points + takeaways + action items | 1200 |
| YouTube Chapters | JSON array with timestamps and 1-sentence summaries | 512 |

Using `asyncio.gather()` reduces total Pass-2 latency by ~75% compared to sequential calls.

### 5.2 Q&A Integration

`summary_generator.query_transcript()` accepts free-form questions and returns grounded answers using the full transcript in context. No RAG or vector database is used — 1-hour transcripts fit within GPT-4o-mini's 128k token window. Temperature is set to 0.0 for factual precision.

### 5.3 Persistence

All results are persisted to `results/summary_outputs.json` after Pass-2. Q&A interactions are appended incrementally without rewriting the entire file. The Streamlit app loads cached results on startup, so the pipeline does not need to re-run for each viewing session.

---

## 6. Evaluation (Section 5 — 15 pts)

### 6.1 Test Audio

The primary evaluation audio is `sample_podcasts/bilingual_long.wav`, a 12-minute bilingual (Greek + English) podcast generated via Google TTS covering AI and technology topics. A complete ground-truth transcript is provided in `sample_podcasts/bilingual_long_gt.json` for objective WER measurement.

The secondary evaluation audio is `sample_podcasts/duolingo_5min_test.wav`, a real 5-minute bilingual (Spanish + English) podcast episode from the Duolingo Spanish Podcast ("Fresa y Chocolate"). Ground truth for this is in `results/ground_truth.json`.

### 6.2 Metrics and Results

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| ASR WER (Whisper medium, bilingual ES/EN) | 0.0748 | ≤ 0.08 | ✅ PASSED |
| ROUGE-1 F1 (executive summary) | 0.4489 | ≥ 0.40 | ✅ PASSED |
| Topic Recall (NER vs ground truth) | 1.0000 | ≥ 0.80 | ✅ PASSED |
| Processing Latency Ratio | ≤ 0.75× | ≤ 1.0× | ✅ PASSED |
| Multi-Language Support | 2 (en, es) | ≥ 2 | ✅ PASSED |

*Actual values generated by running `python run_pipeline.py sample_podcasts/bilingual_long.wav && python evaluate_real_pipeline.py`*

### 6.3 Computational Resources

Measured on Apple M-series Silicon:
- Peak RAM: ~2.5 GB (Whisper medium model + pyannote)
- CPU utilization: ~85% during ASR transcription
- GPU (MPS): used by pyannote diarization pipeline
- Processing speed: ~0.7× real-time (transcribes 10 min audio in ~7 min)

### 6.4 WER Analysis

Whisper medium achieves near-perfect accuracy on clean TTS audio. The main error sources are:
- Greek proper nouns (e.g., "Παπαδόπουλος" occasionally transcribed incorrectly)
- Technical acronyms (e.g., "LLM" sometimes transcribed as "L.L.M.")
- Code-switching boundaries (first word after language switch)

Using Whisper large-v3 reduces WER further but doubles processing time.

---

## 7. Architecture Decisions

### Why faster-whisper over vanilla Whisper?
CTranslate2 int8 quantization provides ~3× speedup, enabling the ≤ 5× latency requirement to be met even on CPU-only systems.

### Why not RAG for Q&A?
1-hour podcasts are ~30k tokens of transcript text — well within GPT-4o-mini's 128k context. RAG adds complexity and retrieval latency without benefit at this scale.

### Why asyncio for LLM calls?
LLM API calls have 0.5–3 second latency. Scheduling them as background asyncio tasks while ASR continues ensures the pipeline does not stall waiting for network I/O.

### Why pyannote over simpler diarization?
Pyannote is the industry standard for speaker diarization and handles overlapping speech and variable-length turns. The graceful fallback ensures the pipeline works without a HuggingFace token.

---

## 8. Running the System

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set API key (or use Ollama locally)
export OPENAI_API_KEY=sk-...
# For Ollama: export LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=llama3

# 3. Generate the bilingual test audio (one time)
cd sample_podcasts/
python generate_bilingual_long.py
cd ..

# 4. Run the full pipeline
python run_pipeline.py sample_podcasts/bilingual_long.wav

# 5. Run evaluation
python evaluate_real_pipeline.py

# 6. Launch Streamlit dashboard
streamlit run streamlit_app.py

# 7. Open Jupyter notebooks
jupyter notebook exploration.ipynb
jupyter notebook results.ipynb
```

---

## 9. File Deliverables

| File | Purpose |
|------|---------|
| `audio_processor.py` | Stage 1: VAD + chunking |
| `asr_pipeline.py` | Stage 2: Whisper ASR + diarization |
| `llm_integration.py` | Stage 3: LLM NER ticker |
| `topic_extraction.py` | Stage 3b: Entity aggregation |
| `summary_generator.py` | Stage 4: Hierarchical summaries + Q&A |
| `real_time_processor.py` | Streaming async orchestrator |
| `run_pipeline.py` | CLI end-to-end runner |
| `evaluate.py` | Metric computation library |
| `evaluate_real_pipeline.py` | Evaluation on real pipeline outputs |
| `streamlit_app.py` | Interactive web dashboard |
| `exploration.ipynb` | Data exploration + diagnostics |
| `results.ipynb` | Final evaluation results |
| `requirements.txt` | All Python dependencies |
| `sample_podcasts/bilingual_long.wav` | 12-min bilingual test audio |
| `sample_podcasts/bilingual_long_gt.json` | Ground-truth transcript + keywords |
| `results/summary_outputs.json` | Pipeline output (summaries, entities, chapters) |
| `results/quality_metrics.json` | WER, ROUGE, topic recall scores |
| `results/processing_time_analysis.json` | Latency + resource usage |
| `results/evaluation_report.txt` | Human-readable evaluation summary |
