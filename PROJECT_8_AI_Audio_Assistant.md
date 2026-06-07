# PROJECT 8: Multilingual Podcast Summarizer with Real-Time Processing


**Φοιτητής/τρία**: ΠΟΛΙΤΑΚΗΣ ΒΙΚΤΩΡ
**Email**: ece22073@go.uop.gr
**ΑΜ**: 9093202200073
**Επιβλέπων**: Παναγιώτης Ζέρβας
**Βαθμολόγηση**: 100 μόρια


## Επισκόπηση Έργου

Implement a real-time podcast processing system that transcribes audio, extracts key topics, and generates automatic summaries using ASR and LLM integration.

## Τεχνικές Απαιτήσεις

### Κύρια Υλοποίηση
1. **Audio Processing Pipeline** (20 points)
   - Stream-based audio processing
   - Real-time VAD (Voice Activity Detection)
   - Handle variable audio quality
   - Multi-language detection
   - Implement chunking strategy (5-10s chunks)

2. **Speech Recognition** (25 points)
   - Implement Whisper for ASR
   - Handle long audio (>1 hour)
   - Speaker diarization integration
   - Timestamp preservation
   - Confidence score filtering

3. **Topic and Content Extraction** (25 points)
   - LLM-based content understanding
   - Key topic extraction
   - Named entity recognition
   - Main ideas summarization
   - Segment-level analysis

4. **Summary Generation** (15 points)
   - Abstractive summarization (3-5 levels of detail)
   - Bullet-point extraction
   - Key takeaways identification
   - Question-answering integration
   - Output formatting (text, structured JSON)

5. **Evaluation** (15 points)
   - ASR quality (WER metric)
   - Summary ROUGE scores
   - User evaluation (comprehensiveness, accuracy)
   - Processing latency analysis
   - Computational resource usage

## Παραδοτέα

```
Politakis/
├── audio_processor.py
├── asr_pipeline.py
├── topic_extraction.py
├── summary_generator.py
├── llm_integration.py
├── real_time_processor.py
├── evaluate.py
├── exploration.ipynb                 # Jupyter: Data exploration and features
├── results.ipynb                      # Jupyter: Training and evaluation results
├── streamlit_app.py                  # Streamlit: Interactive dashboard
├── requirements.txt
├── sample_podcasts/               # Test audio files
├── results/
│   ├── transcription_samples.txt
│   ├── summary_outputs.json
│   ├── processing_time_analysis.json
│   ├── quality_metrics.json
│   └── evaluation_report.txt
└── REPORT.md                         # 5-6 σελίδων with examples
```

## Απαιτήσεις GUI

**Jupyter Notebooks** ( Υποχρεωτικό):
- `exploration.ipynb`: Interactive data exploration, feature/metric visualization
- `results.ipynb`: Training results, evaluation metrics, error analysis with visualizations

**Streamlit Application** (Προαιρετικό):
- `streamlit_app.py`: Web interface for inference and interactive result exploration


## Ελάχιστες Απαιτήσεις

- **ASR WER**: ≤ 8% on test podcasts
- **Summary ROUGE-1**: ≥ 0.40
- **Processing Speed**: ≤ 5x audio duration
- **Topic Extraction**: ≥ 80% recall of important topics
- **Multi-Language**: Support ≥ 3 languages

---

**Προθεσμία Υποβολής**: Τέλος εξαμήνου
**Παρουσίαση**: 15-minute demo with live podcast processing
