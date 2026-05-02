# Vietnamese Financial News RAG System

> **Course**: Text Data Mining & Applications  
> **Dataset**: Vietnamese Financial News 2015–2024 (10K working set)  
> **Pipeline Version**: v2

---

## Project Overview

A Retrieval-Augmented Generation (RAG) system built on 10,000 Vietnamese financial news articles spanning 2015–2024. The system compares three chunking strategies (fixed-size, sentence-aware, article-level) combined with three retrieval methods (dense, sparse BM25, hybrid RRF) across nine total configurations, evaluated with Groq + Llama 3.1 70B as cross-model judge.

---

## Repository Structure

```
implementation/
├── notebooks/           # One notebook per pipeline phase
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_chunking.ipynb
│   ├── 03_embedding.ipynb
│   ├── 04_bm25_indexing.ipynb
│   ├── 05_synthetic_qa.ipynb
│   ├── 06_retrieval_pipeline.ipynb
│   ├── 07_generation_pipeline.ipynb
│   ├── 08_evaluation.ipynb
│   └── 09_ui_demo.ipynb
├── src/                 # Importable Python modules
│   ├── preprocessing.py
│   ├── chunking.py
│   ├── embedding.py
│   ├── indexing.py
│   ├── retrieval.py
│   ├── generation.py
│   ├── evaluation.py
│   └── utils.py
├── app/
│   └── app.py           # Gradio demo (Phase 9)
├── configs/
│   └── config.yaml      # All hyperparameters and paths
├── data/                # gitignored — stored on Drive
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quick Start (Local)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up API keys
cp .env.example .env
# Edit .env and fill in GEMINI_API_KEY and GROQ_API_KEY

# 3. Place raw data
# Copy financial_news_10k.csv → data/raw/financial_news_10k.csv

# 4. Run Phase 1 notebook
jupyter notebook notebooks/01_data_preprocessing.ipynb
```

---

## Quick Start (Google Colab)

```python
# Mount Drive and clone repo
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/<your-repo>/rag-vn-finance /content/rag-vn-finance
%cd /content/rag-vn-finance/implementation
!pip install -r requirements.txt -q
```

All notebooks auto-detect Colab vs. local via `src/utils.py::resolve_path()` and switch paths accordingly.

---

## Pipeline Phases

| Phase | Notebook | Description | Hardware |
|-------|----------|-------------|----------|
| 0 | — | Project setup, model verification | CPU |
| 1 | `01_data_preprocessing.ipynb` | Cleaning + 8-section EDA | CPU |
| 2 | `02_chunking.ipynb` | 3 chunking strategies | CPU |
| 3 | `03_embedding.ipynb` | multilingual-e5-large embeddings | **GPU** |
| 4 | `04_bm25_indexing.ipynb` | BM25 sparse indexes | CPU |
| 5 | `05_synthetic_qa.ipynb` | Gemini batched QA generation | CPU |
| 6 | `06_retrieval_pipeline.ipynb` | 9-config retrieval benchmark | CPU |
| 7 | `07_generation_pipeline.ipynb` | Gemini answer generation | CPU |
| 8 | `08_evaluation.ipynb` | Groq+Llama LLM-as-judge eval | CPU |
| 9 | `09_ui_demo.ipynb` | Gradio demo → HF Spaces | CPU |

---

## Key Design Decisions

- **Cross-model evaluation**: Groq + Llama 3.1 70B judges Gemini outputs to eliminate same-model confirmation bias (Zheng et al., 2023).
- **Batched QA generation**: 10 articles per Gemini call → 50 total requests (10% of 500 RPD daily quota).
- **Checkpoint-first policy**: Every phase with >15 min runtime checkpoints every N steps to survive Colab disconnects.
- **Idempotent design**: Every write cell checks `if not os.path.exists(...)` — safe to re-run from any point.

---

## Google Drive Layout

```
MyDrive/rag-vn-finance/
├── data/raw/
├── data/processed/        ← cleaned.parquet, eda_report.json, eda_plots/
├── data/chunks/           ← fixed_size/, sentence_aware/, article_level/
├── embeddings/
├── indexes/
├── bm25/
├── synthetic_qa/
├── evaluation/
└── checkpoints/
```

---

## Citations

- Zheng et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS 2023.
- Touvron et al. (2023). *Llama: Open and Efficient Foundation Language Models.* arXiv:2302.13971.
