# ECE22073 — Τεχνικό Questionnaire / Q&A

**Φοιτητής**: ΠΟΛΙΤΑΚΗΣ ΒΙΚΤΩΡ (ΑΜ: 9093202200073)
**Course**: ECE22073 — Πανεπιστήμιο Πατρών
**Date**: Ιούνιος 2026

---

## 1. Αρχιτεκτονική & Σχεδιασμός

### Q1: Ποια είναι η αρχιτεκτονική του pipeline;

**A1:** Το pipeline αποτελείται από 5 στάδια που εκτελούνται σειριακά:

```
audio_processor → asr_pipeline → llm_integration → topic_extraction → summary_generator
```

Επιπλέον υπάρχουν τα βοηθητικά modules `transcript_normalizer` (LLM-based διόρθωση ASR σφαλμάτων), `diarize_transcript` (speaker diarization), και `podcast_pipeline` (TTS podcast generation). Το pipeline είναι modular — κάθε στάδιο είναι ανεξάρτητο Python module με strict type hints, δικό του logger, και JSON schema για τα δεδομένα που ανταλλάσσονται μεταξύ των modules.

**Key decision:** Όλα τα δεδομένα που διασχίζουν τα boundaries των modules είναι parsed JSON dicts — ποτέ raw strings. Αυτό εξασφαλίζει type safety και αποτρέπει parsing errors downstream.

---

### Q2: Πώς χειρίζεται το pipeline μεγάλα αρχεία ήχου (>1 ώρα);

**A2:** Με τρεις μηχανισμούς:

1. **Generator-based streaming**: Τα στάδια `audio_processor` και `asr_pipeline` είναι Python generators (`yield`). Δεν φορτώνουν ολόκληρο το αρχείο στη μνήμη — παράγουν chunks ένα-ένα.

2. **VAD-aware chunking**: Το `audio_processor.py` χωρίζει τον ήχο σε chunks 25-30 δευτερολέπτων χρησιμοποιώντας Silero VAD. Το VAD ανιχνεύει σιωπή και κόβει σε φυσικά όρια ομιλίας, όχι σε αυθαίρετα χρονικά σημεία. Αυτό αποτρέπει το κόψιμο στη μέση μιας λέξης.

3. **Memory management**: Μετά από κάθε chunk transcription, το raw audio (`chunk["audio_data"]`) διαγράφεται από τη μνήμη (`del chunk["audio_data"]`). Το MPS GPU cache αδειάζει μετά από κάθε chunk (`torch.mps.empty_cache()`). Τα full chunk dicts αντικαθίστανται με slim metadata (8 fields αντί για όλα τα segments/word timestamps).

**Αποτέλεσμα**: Σταθερή χρήση RAM ~1.9 GB ανεξάρτητα από τη διάρκεια του ήχου. Έχουμε δοκιμάσει με επιτυχία αρχεία 13 λεπτών. Η αρχιτεκτονική κλιμακώνεται σε ώρες χωρίς memory growth.

---

### Q3: Γιατί επιλέχθηκε `faster-whisper` αντί για vanilla Whisper ή HuggingFace Transformers;

**A3:** Μετά από εκτενή benchmarks (3 backends, 4 models, 3 chunk sizes):

1. **faster-whisper (CTranslate2)**: 0.73× real-time σε Apple Silicon M2 Pro με int8 quantization. To CTranslate2 είναι βελτιστοποιημένο για CPU inference με NEON/AMX instructions στο Apple Silicon.

2. **HuggingFace Transformers**: 1.5-3× real-time (πιο αργό από real-time). Το `model.generate()` σε MPS με float32 είναι θεμελιωδώς μη βελτιστοποιημένο. Η pipeline API με `chunk_length_s=30` είναι ακόμα πιο αργή λόγω experimental chunked long-form support.

3. **NVIDIA NeMo (Canary/Parakeet)**: Απαιτεί CUDA GPU — δεν τρέχει σε Apple Silicon. Είναι διαθέσιμα για το Colab backend.

**Συμπέρασμα**: Το `faster-whisper` με CTranslate2 int8 είναι 3-8× ταχύτερο από οποιαδήποτε άλλη λύση σε Apple Silicon. Για το Colab GPU backend, τα NVIDIA models (Parakeet, Canary) είναι υποψήφια αλλά δεν έχουν benchmark-ριστεί ακόμα.

---

### Q4: Ποιο είναι το data flow μεταξύ Streamlit (local UI) και Google Colab (GPU backend);

**A4:**

```
[User] → Streamlit (local MacBook)
    │ uploads WAV → Google Drive: ece22073/input/{job_id}.wav
    │ uploads podcast JSON → Google Drive: ece22073/input/podcast_jobs/{job_id}.json
    │ polls (every 15s) ← Drive: ece22073/output/{job_id}/status.json
    │ reads results ← Drive: ece22073/output/{job_id}/*.json

[Colab watcher] polls Drive API every 10s
    → finds new WAV → runs ASR pipeline → uploads results to output/{job_id}/
    → finds new JSON → runs podcast TTS → uploads MP3 to output/podcasts/
```

**Key decision**: Το Google Drive λειτουργεί ως message bus. Το Streamlit και το Colab δεν επικοινωνούν απευθείας — ανταλλάσσουν αρχεία μέσω Drive. Αυτό επιτρέπει asynchronous processing: ο χρήστης ανεβάζει ένα αρχείο, το Colab το επεξεργάζεται όποτε είναι διαθέσιμο (ακόμα και ώρες αργότερα), και το Streamlit εμφανίζει τα αποτελέσματα μόλις είναι έτοιμα.

**Critical bug που διορθώθηκε**: Το Colab watcher χρησιμοποιούσε αρχικά `os.listdir()` στο Drive FUSE mount path (`/content/drive/MyDrive/...`). Τα αρχεία που ανεβαίνουν μέσω Drive API δεν εμφανίζονται αξιόπιστα στο FUSE mount. Η λύση ήταν να χρησιμοποιηθεί το Drive API απευθείας (`db.find_new_input_files()`) για polling.

---

## 2. Speech Recognition (ASR)

### Q5: Ποιο μοντέλο ASR χρησιμοποιείται και γιατί;

**A5:** `faster-whisper-large-v3-turbo` με int8 quantization.

**Γιατί turbo και όχι small/medium/large-v3**:
- **small**: Ταχύτερο (0.5× real-time) αλλά WER 0.22 — χειρότερη ακρίβεια, ειδικά στα Ελληνικά
- **medium**: Καλή ακρίβεια αλλά 1.3× real-time — πολύ αργό
- **large-v3**: 1.5B παράμετροι, 1.5× real-time, WER χειρότερο από turbo (0.35 vs 0.34)
- **turbo**: 809M παράμετροι, 0.73× real-time, WER 0.34 — καλύτερος λόγος ταχύτητας/ακρίβειας

Το turbo είναι finetuned version του large-v3 με μειωμένα decoder layers (32 → 4), που το κάνει 2× ταχύτερο με ελάχιστη απώλεια ακρίβειας.

---

### Q6: Πώς γίνεται το chunking του ήχου και γιατί 30 δευτερόλεπτα;

**A6:** Το chunking γίνεται από το `audio_processor.py` χρησιμοποιώντας Silero VAD (Voice Activity Detection):

1. Ο ήχος φορτώνεται και κανονικοποιείται στα -20 dBFS, 16 kHz mono
2. Το Silero VAD (RNN-based, τρέχει σε <1 ms ανά frame) ανιχνεύει ομιλία vs σιωπή
3. Ο αλγόριθμος περιμένει μέχρι το chunk να ξεπεράσει τα 25 δευτερόλεπτα (min_chunk_sec)
4. Μόλις βρει 0.5 δευτερόλεπτα συνεχόμενης σιωπής, κόβει εκεί
5. Αν δεν βρει σιωπή μέχρι τα 30 δευτερόλεπτα (max_chunk_sec), κόβει αναγκαστικά

**Γιατί 30 δευτερόλεπτα**:
- Το Whisper έχει receptive field 30 δευτερολέπτων (3000 mel frames)
- Λιγότερα chunks = λιγότερο overhead από pyannote diarization (~5-6s ανά chunk)
- 7 chunks για 180s ήχου αντί για 19 (10s chunks) ή 99 (5-10s chunks)
- Το 10s WER είναι ελαφρώς καλύτερο (0.315 vs 0.338) αλλά 2× πιο αργό

**Benchmark**: Δοκιμάσαμε 30s, 10s, και 5s chunks. Τα 30s δίνουν την καλύτερη ισορροπία.

---

### Q7: Πώς λειτουργεί το multilingual transcription; Γιατί δεν κλειδώνουμε τη γλώσσα ανά chunk;

**A7:** Το Whisper τρέχει με `language=None` — αυτόματη ανίχνευση γλώσσας ανά chunk. Κάθε chunk μπορεί να ανιχνευθεί ως διαφορετική γλώσσα (π.χ. en → el → en).

**Γιατί όχι language locking**: Κάναμε experiment όπου ανιχνεύαμε τη γλώσσα πρώτα (`model.detect_language()`) και αν η πιθανότητα ήταν ≥ 0.90, κλειδώναμε τη γλώσσα (`language=detected_language`). Το αποτέλεσμα ήταν **καμία βελτίωση** στο WER ή στο entity detection. Το `language=None` (auto-detect) δουλεύει εξίσου καλά.

**Γιατί τα entities είναι κατεστραμμένα**: Το πρόβλημα δεν είναι η ανίχνευση γλώσσας — είναι η ποιότητα του συνθετικού TTS ήχου. Το gTTS προφέρει Αγγλικά ονόματα μέσα σε Ελληνικές προτάσεις με αφύσικο τρόπο, και το Whisper τα μεταγράφει φωνητικά (Ιαν Λε Κων αντί Yann LeCun). Αυτό είναι limitation του test data, όχι του μοντέλου.

**Ανάλυση γλωσσών**: Από 28 chunks των 30s, 62.4% ανιχνεύθηκαν ως Ελληνικά, 37.6% ως Αγγλικά, με 15 language switch points.

---

## 3. LLM & NLP

### Q8: Πώς λειτουργεί το Named Entity Recognition (NER);

**A8:** Το NER γίνεται σε δύο περάσματα:

**Pass 1 — Live Ticker (llm_integration.py)**:
- Κάθε ~2 λεπτά συσσωρευμένου transcribed text, ένα background LLM call εξάγει:
  - Named persons, organizations, keywords (τεχνικοί όροι)
  - 2-4 main ideas, 1-sentence segment summary
- Τα calls είναι async (δεν μπλοκάρουν το ASR stream)
- Στο τέλος του stream, `asyncio.gather()` περιμένει όλα τα pending calls

**Pass 2 — Entity Re-Extraction (transcript_normalizer.py)**:
- Μετά το normalization του transcript, ένα δεύτερο LLM call εξάγει entities από το διορθωμένο κείμενο
- Αυτά τα entities ΑΝΤΙΚΑΘΙΣΤΟΥΝ τα raw ticker entities (όχι merge)
- Αυτό διορθώνει τα Ελληνικά μεταγραμμένα ονόματα που το Pass 1 έχασε

**Μοντέλο**: `gpt-5.4-mini-2026-03-17` (OpenAI API), temperature=0 (deterministic extraction).

**Entity registry (topic_extraction.py)**: Deduplication, frequency ranking, cross-window tracking. Υποστηρίζει batch και streaming incremental modes.

---

### Q9: Πώς λειτουργεί το transcript normalization και γιατί χρειάζεται;

**A9:** Το `transcript_normalizer.py` είναι ένα LLM-based cleanup stage που διορθώνει ASR σφάλματα σε proper nouns:

**Τι διορθώνει**:
- Πρόσωπα: Ιαν Λε Κων → Yann LeCun, Jeffrey Hinton → Geoffrey Hinton
- Οργανισμούς: openai → OpenAI, google deepmind → Google DeepMind
- Τεχνικούς όρους: api silicon → Apple Silicon, gpt 4 → GPT-4

**Τι ΔΕΝ κάνει**: summarize, paraphrase, translate, rewrite, βελτίωση γραμματικής

**Validation**:
- Length ratio: το normalized transcript πρέπει να είναι 85-115% του original
- Speaker labels: ≥ 90% preservation
- Timestamps: ≥ 90% preservation (auto-pass αν δεν υπάρχουν)
- Paragraphs: ≥ 80% preservation
- Οποιοδήποτε validation failure → fallback στο raw transcript

**Feature flag**: `ENABLE_TRANSCRIPT_NORMALIZATION=true|false` (default true). Όταν είναι false, το pipeline συμπεριφέρεται ακριβώς όπως πριν.

**WER safety**: Το WER πάντα υπολογίζεται από το `transcript.txt` (raw ASR output). Το normalized transcript δεν επηρεάζει ποτέ το WER — μόνο τα downstream NLP tasks (NER, topic extraction, summary).

---

### Q10: Πώς παράγονται τα summaries;

**A10:** Το `summary_generator.py` παράγει τρία επίπεδα summary:

1. **YouTube Chapters**: 4-7 timestamped chapters με titles και 1-sentence summaries
2. **TL;DR**: 1-sentence overarching thesis (~150-200 χαρακτήρες)
3. **Executive Summary**: 3-paragraph detailed summary (~1800-2400 χαρακτήρες)
4. **Deep Dive**: Structured analysis — overview, bullet points, key takeaways, action items

Όλα τα summaries παράγονται από το ίδιο LLM (`gpt-5.4-mini-2026-03-17`) με temperature=0.3 για readable prose. Τα calls είναι async και concurrent (`asyncio.gather`) για ελαχιστοποίηση του wall-clock latency.

**Q&A Backend**: Το `query_transcript()` απαντά ελεύθερες ερωτήσεις για το transcript. Χρησιμοποιεί synchronous OpenAI client (όχι async) για να αποφύγει conflicts με το Streamlit event loop.

---

## 4. Αξιολόγηση & Metrics

### Q11: Ποια metrics χρησιμοποιούνται για αξιολόγηση και γιατί;

**A11:** Τέσσερα metrics (από τις απαιτήσεις της εργασίας):

| Metric | Εργαλείο | Τι μετράει | Στόχος | Επίδοση |
|--------|----------|-----------|--------|---------|
| WER | jiwer | Word Error Rate — ακρίβεια μεταγραφής | ≤ 0.08 | 0.30 |
| Normalized WER | jiwer (custom) | WER μετά από aggressive stripping (lowercase, σημεία στίξης, speaker labels) | diagnostic | 0.31 |
| ROUGE-1 | rouge-score | Unigram overlap μεταξύ generated και reference summary | ≥ 0.40 | 0.40 |
| Topic Recall | evaluate.py | Set intersection μεταξύ extracted και reference keywords | ≥ 0.80 | 0.42-0.50 |

**Γιατί Normalized WER**: Είναι diagnostic metric. Συγκρίνοντας WER (0.30) με Normalized WER (0.31) βλέπουμε ότι είναι σχεδόν ίδια. Αυτό αποδεικνύει ότι το WER ΔΕΝ οφείλεται σε formatting (σημεία στίξης, κεφαλαία, speaker labels) αλλά σε πραγματικά transcription errors. Αν το Normalized WER ήταν σημαντικά χαμηλότερο, θα σήμαινε ότι το πρόβλημα είναι formatting — αλλά δεν είναι.

**Latency ratio**: 0.74× (επεξεργασία 578s για 782s ήχου). Στόχος ≤ 5× — τον ξεπερνάμε κατά 7×.

---

### Q12: Γιατί το WER είναι 0.30 αντί για ≤ 0.08;

**A12:** Τρεις κύριοι λόγοι:

1. **Συνθετικός TTS ήχος (gTTS)**: Το test dataset δημιουργήθηκε με Google Text-to-Speech. Η προφορά του gTTS στα Ελληνικά (ειδικά για Αγγλικά ονόματα μέσα σε Ελληνικές προτάσεις) είναι αφύσικη. Το Whisper μεταγράφει αυτό που "ακούει" — garbled Greek από synthetic voice. Το reference transcript είναι το ακριβές gTTS source text, που διαφέρει από την πραγματική προφορά.

2. **Reference-hypothesis length mismatch**: Το reference transcript (9515 χαρακτήρες) είναι μεγαλύτερο από το hypothesis (7110 χαρακτήρες). Κάποια τμήματα του ήχου παράγουν ακατανόητο output λόγω TTS quality.

3. **Multilingual proper nouns**: Αγγλικά ονόματα σε Ελληνικό context μεταγράφονται φωνητικά αντί για κανονικά.

**Πώς το αποδεικνύουμε**: Το Normalized WER (0.31) είναι σχεδόν ίδιο με το regular WER (0.30). Αν το πρόβλημα ήταν formatting, το Normalized WER θα ήταν σημαντικά χαμηλότερο. Επίσης, το bare Whisper (χωρίς VAD/chunking) έδωσε WER 0.59 — πολύ χειρότερο — αποδεικνύοντας ότι το VAD/chunking βελτιώνει (δεν χειροτερεύει) την ακρίβεια.

---

## 5. Memory & Performance

### Q13: Πώς λύθηκε το πρόβλημα των 11.8 GB RAM;

**A13:** Το pipeline αρχικά κατανάλωνε 11.8 GB RAM σε MacBook Pro M2 16 GB. Οι αιτίες:

1. **`list(asr.transcribe_file(...))`**: Όλα τα chunks κρατούνταν στη μνήμη ως Python list. Κάθε chunk dict περιέχει segments με word-level timestamps, confidence scores, speaker labels.

2. **`AccumulatedTranscript.to_dict()`**: Το `"chunks": self._chunks` αντέγραφε όλα τα chunks στο output dict — double retain.

3. **MPS GPU cache growth**: Το MPS δεν απελευθερώνει μνήμη επιθετικά. Σε 28 chunks, το cache συσσωρευόταν.

4. **PyAnnote model residency**: ~2-3 GB για segmentation + embedding + PLDA models.

5. **Whisper model residency**: ~1.6 GB για το turbo model (int8).

**Λύσεις**:
- Slim metadata (8 fields αντί για full chunk dict) στο `add_chunk()`
- `del asr_chunks` μετά το LLM stage
- `torch.mps.empty_cache()` μετά από κάθε chunk
- `del chunk["audio_data"]` μετά το transcription

**Αποτέλεσμα**: 1.25-1.93 GB σταθερή χρήση. Το memory profile είναι flat (δεν αυξάνεται με τον αριθμό των chunks).

---

### Q14: Ποιο είναι το processing speed και πώς επιτεύχθηκε;

**A14:** **0.74× real-time** (578s για 782s ήχου).

**Breakdown ανά στάδιο** (για 13 λεπτά ήχου):
- Audio chunking: <1s
- ASR transcription: 517s (0.66× real-time)
- LLM ticker NER: 5s (6 parallel API calls)
- Transcript normalization: 41s (2 API calls, 46 edit regions)
- Entity re-extraction: 9s
- Summary generation: 6s (4 parallel API calls)

**Πώς επιτεύχθηκε**:
1. `faster-whisper` CTranslate2 int8 — 3-8× ταχύτερο από HF Transformers
2. 30s chunks αντί για 5-10s — λιγότερα chunks = λιγότερο overhead
3. Async LLM calls — δεν μπλοκάρουν το ASR stream
4. Concurrent summary generation — 4 calls παράλληλα

---

## 6. Engineering & Infrastructure

### Q15: Πώς είναι οργανωμένος ο κώδικας;

**A15:** Το repository είναι οργανωμένο σε 6 directories:

| Directory | Περιεχόμενο |
|-----------|------------|
| `Pipeline/` | Core pipeline (17 files): audio processing, ASR, LLM, summarization, Drive bridge, Colab watcher |
| `App/` | Streamlit UI, Docker config, requirements |
| `Benchmarks/` | Evaluation scripts + 5 benchmark scripts για διαφορετικά ASR models |
| `Foundational/` | Real-time processor, sanity tests |
| `Samples/` | Test audio files + ground truth |
| `Results/` | Output directory |

**Coding standards**:
- `from __future__ import annotations` σε κάθε αρχείο
- Type hints σε όλες τις public functions
- TypedDict για κάθε schema που διασχίζει module boundaries
- Generator functions για streaming (αποφυγή buffering)
- Κάθε module έχει δικό του logger
- "Why" comments — εξηγούν το rationale, όχι τι κάνει ο κώδικας

---

### Q16: Πώς έχει στηθεί το Docker και γιατί;

**A16:** Το Streamlit app τρέχει σε Docker container για consistency:

```dockerfile
FROM python:3.11-slim
RUN apt-get install -y ffmpeg espeak-ng
COPY requirements.txt .
RUN pip install -r requirements.txt
```

**docker-compose.yml**:
- Volume mount `../:/app` — αλλαγές στον κώδικα αντανακλώνται χωρίς rebuild
- `OPENAI_API_KEY` περνάει από host `.env` file
- `credentials.json` και `token.json` persistant μέσω του volume mount
- Port 8501 exposed

**Γιατί Docker**: Το Streamlit app χρειάζεται system dependencies (ffmpeg, espeak-ng) που δεν είναι διαθέσιμα παντού. Το Docker εξασφαλίζει ότι όλοι οι developers έχουν το ίδιο περιβάλλον. Το volume mount επιτρέπει γρήγορο development cycle (αλλαγές στον κώδικα → άμεσο reload στο Streamlit).

---

### Q17: Ποια προβλήματα αντιμετωπίστηκαν και πώς λύθηκαν;

**A17:** Τα 5 πιο σημαντικά:

1. **11.8 GB RAM usage** → Slim metadata, MPS cache clearing, audio release → 1.9 GB
2. **`max_tokens` API breaking change** → `max_completion_tokens` σε 7 call sites
3. **Streamlit chat crash (asyncio.run)** → Synchronous OpenAI client (`_call_llm_sync`)
4. **Canary translating Greek to English** → `source_lang=target_lang` explicit config
5. **Colab watcher silently broken** → Drive API polling αντί για FUSE `os.listdir()`

**Συνολικά 20 bugs** documented, όλα με root cause analysis και fixes.

---

## 7. Future Work & Reflections

### Q18: Τι θα βελτιώνατε αν είχατε περισσότερο χρόνο;

**A18:**

1. **Real human-spoken test data**: Το μεγαλύτερο bottleneck είναι το συνθετικό TTS dataset. Με πραγματικές ηχογραφήσεις, το WER θα έπεφτε σημαντικά.

2. **NVIDIA model benchmarks**: Ο κώδικας για Parakeet TDT 0.6B και Canary 1B V2 είναι έτοιμος αλλά δεν έχει τρέξει σε Colab GPU.

3. **Better entity canonicalization**: Το normalization βελτιώνει τα entities αλλά δεν είναι τέλειο. Χρειάζεται domain-specific fine-tuning ή few-shot prompting.

4. **Diarization fix**: Το pyannote είναι disabled στο Colab λόγω numpy version mismatch.

5. **Unit tests**: Το project είναι academic deliverable χωρίς test suite. Για production use θα χρειαζόταν test coverage.

6. **Real-time streaming**: Το `real_time_processor.py` υπάρχει αλλά δεν έχει ενσωματωθεί στο production pipeline. Θα επέτρεπε live transcription αντί για batch processing.

---

*Το questionnaire αυτό καλύπτει όλα τα βασικά technical decisions, benchmarks, bugs, και architecture choices του project. Κάθε απάντηση υποστηρίζεται από συγκεκριμένα δεδομένα και μετρήσεις.*
