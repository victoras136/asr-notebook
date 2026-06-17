# Session Handoff — 17 Ιουνίου 2026

> **Σκοπός:** Εξαντλητική καταγραφή όλων των ενεργειών, ευρημάτων, δυσκολιών και αποφάσεων της συνεδρίας. Θα χρησιμοποιηθεί ως πηγή δεδομένων για το report της εργασίας ECE22073.

---

## 1. Επισκόπηση Συνεδρίας

Η συνεδρία είχε τρεις φάσεις:

1. **Ενημέρωση του `AGENTS.md`** — επαλήθευση και βελτίωση του αρχείου οδηγιών για μελλοντικά AI coding sessions.
2. **Επίλυση προβλήματος `OPENAI_API_KEY`** — το chat tab του Streamlit δεν λειτουργούσε, εμφάνιζε προειδοποίηση.
3. **Οδηγίες macOS** — εμφάνιση hidden files.

---

## 2. Φάση 1: Ενημέρωση `AGENTS.md`

### 2.1 Τι είναι το AGENTS.md

Το `AGENTS.md` είναι ένα instruction file που διαβάζεται από AI coding agents (όπως το OpenCode) πριν ξεκινήσουν να δουλεύουν στο repo. Στόχος του είναι να αποτρέπει λάθη και να επιταχύνει το ramp-up. Κάθε γραμμή πρέπει να απαντά στο ερώτημα: *"Θα το έχανε ένας agent χωρίς βοήθεια;"*

### 2.2 Μεθοδολογία Διερεύνησης

Για την ενημέρωση του `AGENTS.md` ακολουθήθηκε η εξής διαδικασία:

1. Ανάγνωση των υψηλότερης αξίας πηγών πρώτα:
   - `README.md`
   - `AGENTS.md` (υπάρχον)
   - `CLAUDE.md`
   - `.gitignore`
   - `docker-compose.yml`
   - `Dockerfile`
   - `requirements.txt` + `requirements_colab.txt`
   - `Pipeline/config.py`
   - `Pipeline/notebook.ipynb`
   - `App/.streamlit/config.toml`
   - `App/.env.example`
   - `GOOGLE_CREDENTIALS_SETUP.md`

2. Διασταύρωση όλων των ισχυρισμών του AGENTS.md με τον πραγματικό κώδικα:
   - Έλεγχος δομής καταλόγων (directory listing του Pipeline/, App/, Benchmarks/, Foundational/)
   - Έλεγχος notebook cells (το notebook έχει 10 cells: 0–9, εκ των οποίων cells 1–6 είναι τα runtime cells)
   - Έλεγχος environment variables στο config.py και streamlit_app.py
   - Έλεγχος git tracking για sensitive files (credentials.json, token.json, .env)

3. Εντοπισμός κενών — τι ξέρει ο agent από τα αρχεία αλλά δεν είναι γραμμένο στο AGENTS.md.

### 2.3 Αλλαγές που Έγιναν στο AGENTS.md

#### 2.3.1 Διόρθωση αριθμού cells στο notebook

**Πριν:** `run cells 1-4 in order`  
**Μετά:** `run cells 1-6 in order`

**Αιτία:** Το notebook έχει 6 code cells που πρέπει να τρέξουν πριν τον watcher:
- Cell 1: Configuration
- Cell 2: Clone repo + pip install
- Cell 3: Drive authentication
- Cell 4: TTS dependencies (προαιρετικό, ελέγχεται από `INSTALL_PODCAST_DEPS`)
- Cell 5: HuggingFace authentication (για pyannote diarization)
- Cell 6: Watcher loop (το `while True` που κάνει poll το Drive)

Το παλιό `1-4` ήταν λάθος — θα έχανε τα cells 5 και 6 που είναι κρίσιμα για το HF auth και τον watcher.

#### 2.3.2 Προσθήκη `_pages_old/` στην αρχιτεκτονική

Το directory `App/_pages_old/` περιέχει παλιά page implementations (`upload.py`, `results.py`, `accuracy.py`) που προϋπήρχαν του refactoring σε monolithic `streamlit_app.py`. Είναι stale code — δεν χρησιμοποιείται πουθενά. Ένας agent που θα τα έβλεπε μπορεί να προσπαθούσε να τα τροποποιήσει αντί για το `streamlit_app.py`.

#### 2.3.3 Διαχωρισμός requirements files

**Πριν:** Ένα entry "Install deps" → `pip install -r App/requirements.txt`  
**Μετά:** Δύο entries:
- `pip install -r App/requirements.txt` — Streamlit UI μόνο, ΧΩΡΙΣ ML dependencies (torch, transformers, faster-whisper, pyannote)
- `pip install -r App/requirements_colab.txt` — Colab GPU, με ΟΛΑ τα ML dependencies

**Αιτία:** Το `requirements.txt` περιέχει μόνο streamlit, pandas, plotly, google-auth, openai, rouge-score, jiwer, sacrebleu. Το `requirements_colab.txt` περιέχει torch, torchaudio, faster-whisper, transformers, pyannote.audio, pydub, psutil, nest_asyncio, huggingface_hub. Αν κάποιος κάνει `pip install -r App/requirements.txt` σε Colab, θα λείπουν ΟΛΑ τα ML dependencies.

#### 2.3.4 Docker compose — GitHub URL build context

**Προσθήκη gotcha:** Το `docker-compose.yml` χρησιμοποιεί `context: https://github.com/victoras136/asr-notebook.git` αντί για local build context. Αυτό σημαίνει:
- Το Docker κατεβάζει το repo από το GitHub (συγκεκριμένα τον default branch)
- Αλλαγές σε τοπικά αρχεία (Pipeline/, Dockerfile) **δεν εμφανίζονται** στο Docker χωρίς git push
- Για local development πρέπει να χρησιμοποιηθεί native Python (`streamlit run streamlit_app.py`)

Αυτό είναι κρίσιμο — ένας agent θα μπορούσε να κάνει αλλαγές, να τρέξει `docker compose up` και να αναρωτιέται γιατί δεν βλέπει τις αλλαγές.

#### 2.3.5 Προσθήκη missing environment variables

**Προστέθηκαν:**
- `MAX_NORMALIZATION_CHARS` (default: `8000`) — chunk threshold για μεγάλα transcripts κατά το normalization
- `GITHUB_TOKEN` — Colab Secret για clone private repo στο Cell 1

**Επαληθεύτηκαν όλα τα υπάρχοντα:**
- `OPENAI_API_KEY`, `NORMALIZATION_MODEL`, `ENABLE_TRANSCRIPT_NORMALIZATION`, `HF_TOKEN`, `LLM_BASE_URL`, `LLM_MODEL`

#### 2.3.6 Βελτίωση gotchas

**Προστέθηκαν:**
- **Two separate requirements files** — δες 2.3.3
- **Docker compose builds from GitHub URL** — δες 2.3.4
- **`credentials.json`** required for Drive OAuth — τοποθετείται στο `App/`, είναι gitignored. Το `App/token.json` δημιουργείται αυτόματα μετά το πρώτο browser OAuth. Οδηγίες στο `GOOGLE_CREDENTIALS_SETUP.md`.
- **`_pages_old/`** — stale, ignore
- **`pip install bark` / `pip install dia`** — εγκαθιστούν άσχετα work-diary packages, όχι TTS μοντέλα. Τα TTS μοντέλα εγκαθίστανται από git URLs (βλ. `requirements_colab.txt` lines 42-45).
- **Streamlit config** — `App/.streamlit/config.toml`: CORS disabled, dark theme, toolbar viewer mode
- **CLAUDE.md** — reference για deeper notes (CSS design system, sidebar collapse bug, TTS model details)

**Αφαιρέθηκε:**
- **"Two `ece22073` folders in Drive"** — ήταν time-specific operational note (δημιουργήθηκε 12:00), δεν είναι χρήσιμο για agents.

**Διατηρήθηκαν όλα τα υπόλοιπα gotchas** που επαληθεύτηκαν:
- WER uses raw `transcript.txt` (normalization never touches it)
- Drive FUSE broken on Colab → use `db.find_new_input_files()` (Drive API)
- `logging.basicConfig(force=True)` mandatory before importing watcher modules
- Do NOT call `cjw.main_loop()` → use inline watcher
- Colab watcher accepts `.wav`, `.mp3`, `.m4a` via pydub
- Diarization disabled on Colab (numpy/pyannote version mismatch)
- Word timestamps disabled on Colab (CTranslate2 CUDA alignment crash on T4)
- No CI, no tests, no linter (academic deliverable)
- Terminal hangs: run pipelines directly in terminal, not via agent bash tool

### 2.4 Ευρήματα από τη Διασταύρωση Αρχείων

#### Discrepancy: README/CLAUDE.md λένε "auto-stops after 5 min of idle"

Το notebook cell 6 (watcher) είναι ένα `while True` loop χωρίς **κανέναν** μηχανισμό auto-stop. Το `STALL_TIMEOUT_SEC = 600` (10 λεπτά) στο `config.py` χρησιμοποιείται από το **Streamlit** για να μαρκάρει jobs ως "stalled" — όχι για να σταματήσει το Colab runtime.

#### Discrepancy: README λέει "Cell 4 polls Drive..."

Στην πραγματικότητα το drive mount γίνεται στο Cell 3, και ο watcher loop είναι στο **Cell 6**. Το Cell 4 είναι TTS dependencies.

#### Επαλήθευση gitignore

- `credentials.json` — gitignored (ισχύει για root και App/)
- `token.json` — gitignored (root pattern, καλύπτει και App/)
- `.env` — gitignored (root pattern, καλύπτει και App/)
- Τα υπάρχοντα `credentials.json` σε root και `App/` είναι untracked — ασφαλή

#### Επαλήθευση δομής καταλόγων

| Directory | Files | Επαληθεύτηκε |
|-----------|-------|-------------|
| Pipeline/ | 14 Python files + notebook.ipynb | ✅ |
| App/ | streamlit_app.py, comparison_metrics.py, requirements, Docker config | ✅ |
| Benchmarks/ | 7 Python files + 2 notebooks | ✅ |
| Foundational/ | real_time_processor.py, sanity_transcribe.py | ✅ |
| Samples/sample_podcasts/ | test audio + ground truth | ✅ |
| Results/ | Κενό (output directory) | ✅ |
| App/_pages_old/ | upload.py, results.py, accuracy.py, __init__.py | ⚠️ Stale |
| App/.streamlit/ | config.toml | ✅ |

### 2.5 Κατάσταση Αρχείων που ΔΕΝ Υπάρχουν (επιβεβαιώθηκε)

- Δεν υπάρχει `opencode.json` (project-level)
- Δεν υπάρχει `.cursor/rules/` ή `.cursorrules`
- Δεν υπάρχει `pyproject.toml`, `Makefile`, ή CI workflows (`.github/workflows/`)
- Δεν υπάρχουν tests, linter config, ή typechecker config

---

## 3. Φάση 2: Επίλυση `OPENAI_API_KEY` στο Chat Tab

### 3.1 Το Πρόβλημα

Το chat tab στο Streamlit (`📝 Notebook Workspace` page) εμφάνιζε:
```
Set OPENAI_API_KEY in your environment to enable chat.
```

### 3.2 Διάγνωση

Εντοπίστηκε η πηγή στο `App/streamlit_app.py:826-828`:

```python
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    st.warning("Set OPENAI_API_KEY in your environment to enable chat.")
    return
```

Ο κώδικας διαβάζει το `OPENAI_API_KEY` από environment variables. Υπάρχουν δύο τρόποι να στηθεί:

1. **Docker:** Το `docker-compose.yml` έχει `env_file: - .env` (line 28), που διαβάζει το `App/.env`.
2. **Native Python:** Το `os.environ.get()` διαβάζει από το shell environment ή από `.env` αν χρησιμοποιείται python-dotenv (αλλά το `requirements.txt` δεν περιλαμβάνει python-dotenv — άρα για native Python χρειάζεται export στο shell).

### 3.3 Η Λύση

1. Το `App/.env` **δεν υπήρχε** — γι' αυτό το Docker δεν έβρισκε το key.
2. Το `App/.env.example` υπήρχε ως template.
3. Εκτελέστηκε: `cp App/.env.example App/.env`
4. Το αρχείο περιέχει: `OPENAI_API_KEY=sk-your-key-here` — ο χρήστης πρέπει να αντικαταστήσει με το πραγματικό του key.

### 3.4 Gitignore

Το `.env` είναι **ήδη** στο `.gitignore` (γραμμή 6). Το pattern χωρίς path prefix καλύπτει το `.env` σε οποιοδήποτε directory. Δεν χρειάστηκε καμία αλλαγή.

```gitignore
# Γραμμή 6 του .gitignore:
.env
```

Επαληθεύτηκε ότι το `App/.env` είναι untracked (δεν έχει γίνει ποτέ commit).

### 3.5 Docker Workflow

Για να λειτουργήσει το chat tab με Docker:

```bash
cd App
cp .env.example .env          # Πρώτη φορά μόνο
# Επεξεργασία .env → βάλε το OPENAI_API_KEY
docker compose up
```

Το `docker-compose.yml`:
- `env_file: - .env` — φορτώνει το `App/.env` ως environment variables στο container
- `volumes: - ./credentials.json:/repo/App/credentials.json:ro` — mount για Google OAuth
- `volumes: - ./token.json:/repo/App/token.json` — mount για cached OAuth token
- `healthcheck:` — ελέγχει `http://localhost:8501/_stcore/health` κάθε 30s

---

## 4. Φάση 3: Εμφάνιση Hidden Files στο macOS

- **Finder:** `Cmd + Shift + .` (Command + Shift + Period)
- **Terminal:** `ls -la`

---

## 5. Συνολική Κατανόηση της Αρχιτεκτονικής

### 5.1 Data Flow

```
[Χρήστης] → Streamlit UI (App/, τοπικό μηχάνημα)
               │ ανέβασμα audio → Drive: ece22073/input/{job_id}.wav
               │ ανέβασμα JSON  → Drive: ece22073/input/podcast_jobs/{job_id}.json
               │ polling status ← Drive: ece22073/output/{job_id}/status.json
               │ ανάγνωση αποτελεσμάτων ← Drive: ece22073/output/{job_id}/*.json
               
[Colab watcher] (T4 GPU) — κάνει poll το Drive API κάθε 10s
  → βρίσκει audio → τρέχει ASR pipeline → ανεβάζει results στο output/{job_id}/
  → βρίσκει JSON  → τρέχει podcast TTS   → ανεβάζει MP3 στο output/podcasts/
```

### 5.2 ASR Pipeline Stages

```
audio_processor → asr_pipeline → llm_integration → transcript_normalizer
                → topic_extraction → summary_generator → podcast_pipeline
```

**Output files** ανά job:
- `transcript.json` — raw transcript με timestamps, confidence scores
- `transcript.txt` — raw transcript (αυτό χρησιμοποιείται για WER)
- `normalized_transcript.txt` — LLM-καθαρισμένο transcript
- `summary_outputs.json` — TL;DR, executive summary, deep dive, entities
- `quality_metrics.json` — WER, ROUGE scores
- `processing_time_analysis.json` — χρόνοι ανά stage

### 5.3 Streamlit Pages

| Page Key | Λειτουργία |
|----------|-----------|
| `Upload` | File upload → Drive → transcription polling |
| `Notebook` | 3-column: Sources \| Chat \| Studio |
| `Summaries` | TL;DR / Executive / Deep Dive views |
| `Podcast` | TTS podcast generation |
| `Accuracy Check` | WER + ROUGE diff viewer |

### 5.4 Environment Variables (πλήρης λίστα)

| Variable | Default | Πού χρειάζεται | Σκοπός |
|----------|---------|---------------|--------|
| `OPENAI_API_KEY` | — | Colab Secrets + App/.env | LLM stages (NER, summary, normalization, chat) |
| `HF_TOKEN` | — | Colab Secrets | pyannote speaker diarization (gated model) |
| `GITHUB_TOKEN` | — | Colab Secrets | Clone private repo στο Cell 1 |
| `NORMALIZATION_MODEL` | `gpt-5.4-mini-2026-03-17` | env | Μοντέλο για transcript cleanup |
| `ENABLE_TRANSCRIPT_NORMALIZATION` | `true` | env | Feature flag για LLM cleanup |
| `MAX_NORMALIZATION_CHARS` | `8000` | env | Chunk threshold για μεγάλα transcripts |
| `LLM_MODEL` | `gpt-5.4-mini-2026-03-17` | env | Μοντέλο για NER + summary |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | env | Override για local Ollama |

### 5.5 Drive Folder Structure

```
ece22073/
├── input/              ← .wav/.mp3/.m4a files (ο χρήστης ανεβάζει εδώ)
│   └── podcast_jobs/   ← JSON configs για podcast generation
├── output/
│   ├── {job_id}/       ← results ανά job
│   └── podcasts/       ← generated .mp3 files
└── models/             ← model cache
```

### 5.6 Job State Machine

Τα jobs περνούν από στάδια (`Pipeline/config.py:JOB_STAGES`):
```
uploading → asr → normalization → summary → podcast_script → podcast_tts → done
                                                                           → error
                                                                           → stalled
```

- **Terminal stages** (`done`, `error`, `stalled`): ο Colab watcher δεν τα ξαναπιάνει
- **Non-terminal stages**: ο watcher θα τα ξαναρχίσει σε restart (resume)
- **Stall detection**: αν ένα job δεν έχει update για 10 λεπτά (`STALL_TIMEOUT_SEC=600`), το Streamlit το μαρκάρει ως stalled

### 5.7 ASR Models

| Model | Backend | Greek | WER | Notes |
|-------|---------|-------|-----|-------|
| faster-whisper turbo | CTranslate2 int8 | ✅ | 0.34 | Production baseline |
| faster-whisper large-v3 | CTranslate2 int8 | ✅ | 0.35 | Slower |
| Parakeet TDT 0.6B | Transformers CUDA | ✅ | pending | Colab T4 + git transformers |
| Canary 1B V2 | NeMo CUDA | ✅ | pending | `source_lang=target_lang` |

---

## 6. Key Gotchas και Δυσκολίες (Πλήρης Λίστα)

### 6.1 Docker & Development

- **Το Docker compose χτίζει από GitHub URL**, όχι από local files. Αυτό σημαίνει ότι αλλαγές στον κώδικα του Pipeline ή στο Dockerfile ΔΕΝ εμφανίζονται στο Docker μέχρι να γίνει git push. Για local development: native Python.
- **Το Dockerfile κάνει `COPY . /repo`** — αντιγράφει ΟΛΟ το repo. Το `.dockerignore` υπάρχει αλλά είναι κενό ουσιαστικά (δεν φάνηκε στο listing).

### 6.2 Requirements & Dependencies

- **Δύο διαφορετικά requirements files:**
  - `App/requirements.txt` → Streamlit UI μόνο (20 γραμμές)
  - `App/requirements_colab.txt` → Colab GPU με όλα τα ML deps (51 γραμμές)
  - Το `requirements_colab.txt` έχει τα TTS installations σχολιασμένα (kokoro, dia, bark, xtts, f5) — εγκαθίστανται από git.
- **`pip install bark` εγκαθιστά άσχετο work-diary package**, όχι το Bark TTS model. Το Bark TTS εγκαθίσταται από: `pip install git+https://github.com/suno-ai/bark.git`
- **`pip install dia` εγκαθιστά άσχετο work-diary package**, όχι το Dia-1.6B. Το Dia εγκαθίσταται από: `pip install git+https://github.com/nari-labs/dia.git`
- **`scipy`/`numpy` version mismatch** μετά από chaotic pip install → `pip install --force-reinstall scipy numpy` + kernel restart.

### 6.3 Colab Notebook

- **Drive FUSE είναι broken στο Colab** — ο watcher ΠΡΕΠΕΙ να χρησιμοποιεί `db.find_new_input_files()` (Drive API polling), ποτέ `os.listdir()` στο `/content/drive/...`. Αυτό είναι το πιο συχνό bug ("jobs never picked up").
- **`logging.basicConfig(force=True)` είναι ΥΠΟΧΡΕΩΤΙΚΟ** πριν από οποιοδήποτε import watcher module. Χωρίς αυτό, όλα τα errors καταπίνονται σιωπηλά γιατί το `basicConfig` τρέχει μόνο υπό `__main__`.
- **ΜΗΝ καλείς `cjw.main_loop()`** — το logging δεν είναι configured όταν γίνεται import ως module. Χρησιμοποίησε τον inline watcher loop.
- **Ο notebook watcher είναι `while True` χωρίς auto-stop.** Δεν υπάρχει idle timeout. Το Colab runtime θα τρέχει επ' αόριστον μέχρι να γίνει interrupt ή να λήξει το session.
- **Diarization είναι disabled στο Colab** λόγω numpy/pyannote version mismatch στο τρέχον Colab image.
- **Word timestamps είναι disabled στο Colab** — το CTranslate2 CUDA alignment κρασάρει στο T4 GPU. Λειτουργούν μόνο σε local CPU runs.

### 6.4 Authentication

- **`App/credentials.json`** — Google OAuth 2.0 Desktop client credentials. Κατεβαίνει από το Google Cloud Console. Τοποθετείται στο `App/`. Είναι gitignored. Οδηγίες δημιουργίας στο `GOOGLE_CREDENTIALS_SETUP.md`.
- **`App/token.json`** — cached OAuth token. Δημιουργείται αυτόματα μετά το πρώτο browser OAuth login (κλικ στο "🔗 Connect Google Drive" στο sidebar).
- **Για αλλαγή Google account στο Streamlit:** αντικατάσταση `credentials.json` + διαγραφή `token.json` + restart + reconnect.
- **Για αλλαγή Google account στο Colab:** Runtime → Disconnect and delete runtime → rerun all cells → νέο auth prompt.
- **Streamlit και Colab πρέπει να χρησιμοποιούν το ΙΔΙΟ Google account** — αλλιώς τα uploads πάνε σε διαφορετικό Drive από ό,τι κάνει poll ο watcher.

### 6.5 Code Organization

- **`App/_pages_old/`** — περιέχει `upload.py`, `results.py`, `accuracy.py`, `__init__.py`. Είναι stale code από πριν το refactoring. Το τρέχον app είναι **monolithic** `streamlit_app.py` (~1216 γραμμές) που κάνει routing μέσω `st.session_state["_current_page"]`.
- **`comparison_metrics.py`** είναι στο `App/` (όχι στο `Pipeline/`) και import-άρεται απευθείας από το `streamlit_app.py`.
- **Όλα τα Drive paths προέρχονται από το `Pipeline/config.py`** — κανένα αρχείο δεν κάνει hardcode Drive folder strings.
- **`drive_bridge.py` κάνει auto-detect περιβάλλοντος** (Colab vs local) — ίδιο API surface, διαφορετικό auth path.
- **Job IDs προέρχονται από filenames** (`{job_id}.wav`, `{job_id}.json`), δεν generate-άρονται κατά το dispatch.
- **Failed jobs δεν archive-άρονται** — το input file μένει στο `input/` ώστε ένα Colab restart να κάνει αυτόματα retry.

### 6.6 WER & Evaluation

- **Το WER χρησιμοποιεί ΠΑΝΤΑ το raw `transcript.txt`** — το LLM normalization pass δεν το τροποποιεί ποτέ.
- **Evaluation commands:**
  ```bash
  # Real pipeline evaluation
  python3 -c "import sys; sys.path.insert(0, 'Benchmarks'); sys.path.insert(0, 'Pipeline'); from evaluate_real_pipeline import run_real_evaluation; run_real_evaluation()"
  
  # Benchmark all ASR models  
  python3 Benchmarks/benchmark_all.py Samples/sample_podcasts/bilingual_long.wav Samples/sample_podcasts/bilingual_long_gt.json --normalize
  ```

### 6.7 Streamlit Configuration

- `App/.streamlit/config.toml`:
  - `enableCORS = false`
  - `enableXsrfProtection = true`
  - `toolbarMode = "viewer"`
  - Dark theme με custom colors: `backgroundColor = "#070604"`, `textColor = "#d4c4a0"`
  - `font = "monospace"`
- **CLAUDE.md αναφέρει sidebar collapse bug:** το CSS rule `header { visibility: hidden; }` (line 49) κρύβει ολόκληρο το `<header>` element, συμπεριλαμβανομένου του expand arrow όταν το sidebar είναι collapsed — καθιστώντας αδύνατο το reopen χωρίς page refresh. Το fix είναι να αντικατασταθεί με `[data-testid="stToolbar"]`.

### 6.8 TTS Models

- **Μόνο το Kokoro-82M είναι tested end-to-end.**
- **Voices:** `af_heart` (Speaker A), `am_michael` (Speaker B)
- **Dia-1.6B χρειάζεται ~10 GB VRAM**
- **Bark, XTTS-v2, F5-TTS** είναι wired up αλλά untested

### 6.9 Non-fatal Errors

- **`RuntimeError: Event loop is closed`** spam από httpx async cleanup στο ASR thread — μη-θανατηφόρο, τα jobs ολοκληρώνονται κανονικά.
- **`ImportError: cannot import name 'box_iou'`** — προκαλείται από `sys.modules.setdefault` stubs για `transformers.loss.loss_for_object_detection`. Fix: αφαίρεσέ τα.

---

## 7. Αρχεία που Τροποποιήθηκαν

| Αρχείο | Ενέργεια | Περιγραφή |
|--------|---------|-----------|
| `AGENTS.md` | Updated | 12+ αλλαγές: διόρθωση cell count, προσθήκη missing env vars, gotchas, architecture notes (βλ. §2.3) |
| `App/.env` | Created | Αντιγραφή από `.env.example` για το OPENAI_API_KEY (βλ. §3.3) |

---

## 8. Τι ΔΕΝ Έγινε (Out of Scope)

- Δεν έγινε commit των αλλαγών (δεν ζητήθηκε)
- Δεν έγινε testing του pipeline ή του Streamlit
- Δεν έγιναν αλλαγές στον κώδικα του pipeline
- Δεν προστέθηκε actual API key στο `.env` (ο χρήστης θα το κάνει manually)

---

## 9. Χρήσιμα Commands (Quick Reference)

```bash
# Docker
cd App && docker compose up                    # Build + start
docker compose up -d --build                   # Rebuild μετά από git push

# Native Python  
cd App && streamlit run streamlit_app.py       # Τοπικό τρέξιμο

# Pipeline (local)
python3 -c "import sys; sys.path.insert(0, 'Pipeline'); from run_pipeline import run_pipeline; run_pipeline('Samples/sample_podcasts/bilingual_long.wav')"

# Evaluation
python3 -c "import sys; sys.path.insert(0, 'Benchmarks'); sys.path.insert(0, 'Pipeline'); from evaluate_real_pipeline import run_real_evaluation; run_real_evaluation()"

# Benchmark all ASR models (Colab GPU)
python3 Benchmarks/benchmark_all.py Samples/sample_podcasts/bilingual_long.wav Samples/sample_podcasts/bilingual_long_gt.json --normalize

# Source environment (IBM ACE — από AGENTS του OpenCode config, όχι του repo)
source "/Applications/IBM App Connect Enterprise/server/bin/mqsiprofile"
```

---

## 10. Πηγές που Χρησιμοποιήθηκαν

| Πηγή | Τι προσέφερε |
|------|-------------|
| `README.md` | Επισκόπηση, architecture, setup οδηγίες, environment variables |
| `AGENTS.md` (παλιό) | Βάση για ενημέρωση — επαληθεύτηκαν και βελτιώθηκαν όλοι οι ισχυρισμοί |
| `CLAUDE.md` | Deeper notes για CSS design system, sidebar bug, TTS, inline watcher pattern |
| `Pipeline/config.py` | Drive folder structure, polling intervals, job stages, TypedDict schemas |
| `Pipeline/notebook.ipynb` | Επαλήθευση cell layout, watcher loop, TTS install logic |
| `Pipeline/run_pipeline.py` | Επαλήθευση pipeline stages, imports, output files |
| `App/docker-compose.yml` | Docker build context (GitHub URL), env_file, volumes, healthcheck |
| `App/Dockerfile` | Docker image: Python 3.11-slim, system deps, WORKDIR, EXPOSE |
| `App/requirements.txt` | UI-only dependencies (20 lines) |
| `App/requirements_colab.txt` | Colab ML dependencies (51 lines) — TTS git URLs documented |
| `App/.env.example` | Template για OPENAI_API_KEY |
| `App/.streamlit/config.toml` | CORS, theme, toolbar settings |
| `App/streamlit_app.py` | Επαλήθευση OPENAI_API_KEY check, page routing |
| `App/_pages_old/` | Stale code — upload.py, results.py, accuracy.py |
| `.gitignore` | Επαλήθευση ότι .env, credentials.json, token.json είναι gitignored |
| `GOOGLE_CREDENTIALS_SETUP.md` | Οδηγίες δημιουργίας OAuth credentials |
| `PROJECT_8_AI_Audio_Assistant.md` | Παλιό assignment spec — ιστορική αναφορά μόνο |

---

*Τέλος handoff. Ημερομηνία: 17 Ιουνίου 2026. Session με OpenCode (model: deepseek-v4-pro).*
