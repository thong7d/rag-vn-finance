<div align="center">

# 📈 RAG Finance VN

**Production-grade Retrieval-Augmented Generation system for Vietnamese financial news**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19-61dafb?logo=react)](https://react.dev)
[![CI](https://github.com/thong7d/rag-vn-finance/actions/workflows/ci.yml/badge.svg?branch=microservices-migration)](https://github.com/thong7d/rag-vn-finance/actions)

**[🚀 Live Demo](https://rag-vn-finance.vercel.app)** · **[🤗 Gradio Demo](https://huggingface.co/spaces/thong7d/financial-news-rag)** · **[📦 Dataset on HF Hub](https://huggingface.co/datasets/thong7d/rag-vn-finance-bm25)**

</div>

---

## Overview

RAG Finance VN is a full-stack Question & Answering system built on **10,000+ Vietnamese financial news articles** (2015–2024). Given a natural language question, the system retrieves the most relevant passages from a vector database and generates a concise, grounded answer with verifiable source citations.

The project demonstrates an end-to-end applied NLP pipeline — from raw data ingestion and embedding through hybrid retrieval, LLM generation with multi-provider fallback, and automated evaluation — deployed as an independent microservices architecture on cloud infrastructure.

---

## System Architecture

```
┌─────────────────────┐      SSE Stream (real-time)      ┌──────────────────────────────────────┐
│   React Frontend    │ ◄──────────────────────────────► │   FastAPI Backend                    │
│   (Vercel)          │                                   │   (Render)                           │
└─────────────────────┘                                   │                                      │
                                                          │  POST /api/ask → SSE pipeline:       │
                                                          │  ┌────────────────────────────────┐  │
                                                          │  │ 1. Query Decomposition (opt.)  │  │
                                                          │  │    └─ Gemma 4 31B via AI Studio│  │
                                                          │  │ 2. Dense Retrieval             │  │
                                                          │  │    └─ multilingual-e5-large    │  │
                                                          │  │       + Qdrant Cloud HNSW      │  │
                                                          │  │ 3. Sparse Retrieval            │  │
                                                          │  │    └─ SQLite FTS5 (BM25)       │  │
                                                          │  │ 4. RRF Fusion + Cohere Rerank  │  │
                                                          │  │ 5. LLM Generation              │  │
                                                          │  │    └─ Gemini → Mistral → Gemma │  │
                                                          │  └────────────────────────────────┘  │
                                                          └──────────────────────────────────────┘
```

---

## Key Features

| Feature | Detail |
|---|---|
| **Hybrid Retrieval** | Dense (multilingual-e5-large via HF Inference API) + Sparse (SQLite FTS5 BM25) fused with Reciprocal Rank Fusion |
| **Cohere Reranker** | Cross-encoder reranking of top-30 candidates → top-5 final passages |
| **3-Layer LLM Fallback** | Gemini 2.0 Flash Lite → Mistral Small 2506 → Gemma 3 27B — automatic failover with no user disruption |
| **Query Decomposition** | Optional deep-analysis mode: Gemma 4 31B decomposes complex multi-hop questions into parallel sub-queries |
| **SSE Streaming** | Token-by-token streaming with real-time pipeline progress events and ETA countdown |
| **Source Citations** | Every answer is accompanied by ranked source passages with Cohere relevance scores |
| **Automated Evaluation** | Nightly regression tests using LLM-as-a-Judge (Qwen 3.6 27B via Groq) measuring Faithfulness & Answer Relevancy |
| **CI/CD Pipeline** | GitHub Actions: lint (Ruff + ESLint), smoke tests on every push, nightly regression on 5 sampled ground-truth QA pairs |

---

## Evaluation Results

Evaluated on 150 synthetic ground-truth QA pairs using RAGAS framework:

| Metric | Score |
|---|---|
| Faithfulness | **0.93** |
| Answer Relevancy | **0.91** |
| Context Recall | **0.98** |

Retrieval benchmark (sentence-aware chunking, top-5 results):

| Method | Hit@5 | MRR@5 |
|---|---|---|
| Dense only (FAISS) | 0.71 | 0.58 |
| Sparse only (BM25) | 0.64 | 0.52 |
| **Hybrid + Rerank** | **0.89** | **0.81** |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Embedding** | `intfloat/multilingual-e5-large` via HuggingFace Inference API |
| **Vector Store** | Qdrant Cloud (HNSW index, cosine similarity) |
| **Sparse Search** | SQLite FTS5 (BM25, hosted on HF Hub as a dataset) |
| **Fusion** | Reciprocal Rank Fusion (RRF) |
| **Reranker** | Cohere `rerank-multilingual-v3.0` |
| **LLM** | Gemini 2.0 Flash Lite → Mistral Small 2506 → Gemma 3 27B |
| **Query Decomposition** | Gemma 4 31B via Google AI Studio |
| **LLM Judge** | Qwen 3.6 27B via Groq (nightly evaluation) |
| **Backend** | FastAPI + Uvicorn, SSE streaming, Prometheus metrics (`/metrics`) |
| **Frontend** | React 19 + Vite, Vanilla CSS (Bloomberg Terminal dark theme) |
| **Backend Deploy** | Render (Free Tier, Docker) |
| **Frontend Deploy** | Vercel |
| **CI/CD** | GitHub Actions (Ruff lint, ESLint, smoke test, nightly regression) |

---

## Repository Structure

```
rag-vn-finance/
├── backend/                        # FastAPI microservice (deployed on Render)
│   ├── core/
│   │   ├── config.py               # Pydantic Settings — env var management
│   │   ├── logging.py              # Structured logging setup
│   │   └── metrics.py              # Prometheus metrics definitions
│   ├── routers/
│   │   └── ask.py                  # POST /api/ask — async SSE pipeline orchestrator
│   ├── services/
│   │   ├── decomposer.py           # Query decomposition (Gemma 4 31B)
│   │   ├── embedding.py            # HuggingFace Inference API client
│   │   ├── retrieval.py            # Dense + Sparse + RRF + Cohere Rerank
│   │   ├── generation.py           # 3-layer LLM SSE streaming with auto-fallback
│   │   └── sqlite_loader.py        # SQLite FTS5 DB download from HF Hub
│   ├── main.py                     # FastAPI app + lifespan startup, Prometheus instrumentation
│   ├── requirements.txt
│   ├── .env.example
│   └── Dockerfile
│
├── frontend/                       # React 19 + Vite (deployed on Vercel)
│   ├── src/
│   │   ├── components/
│   │   │   ├── StatusBar.jsx       # Pipeline step progress + ETA countdown
│   │   │   ├── ChatMessage.jsx     # Streaming answer with blinking cursor
│   │   │   ├── ChatInput.jsx       # Textarea + Deep Analysis toggle
│   │   │   ├── DecompositionPanel.jsx  # Sub-query visualization
│   │   │   └── SourceCard.jsx      # Source citation with relevance score bar
│   │   ├── hooks/
│   │   │   └── useChat.js          # SSE lifecycle, ETA countdown, error classification
│   │   ├── services/
│   │   │   └── api.js              # Fetch-based SSE stream wrapper (async generator)
│   │   └── styles/
│   │       └── index.css           # Bloomberg Terminal dark theme, CSS custom properties
│   └── package.json
│
├── pipeline/                       # Offline RAG data pipeline (run locally or on Colab)
│   ├── src/
│   │   ├── preprocessing.py        # Text cleaning, deduplication, EDA
│   │   ├── chunking.py             # 3 strategies: fixed_size, sentence_aware, article_level
│   │   ├── embedding.py            # Batch embedding with multilingual-e5-large
│   │   ├── qdrant_indexing.py      # Upload vectors to Qdrant Cloud
│   │   ├── retrieval.py            # Hybrid retrieval + Cohere Rerank (local evaluation)
│   │   ├── generation.py           # RAG generation for pipeline evaluation
│   │   └── evaluation.py           # RAGAS metrics evaluation
│   ├── evaluation/
│   │   └── evaluate_aio.py         # Async end-to-end evaluation runner
│   ├── synthetic_qa/
│   │   ├── generate_qa_stratified.py  # Stratified QA pair generation
│   │   └── ground_truth_final.jsonl   # 150 ground-truth QA pairs for regression testing
│   └── configs/
│       └── config.yaml             # Centralized pipeline configuration
│
├── notebooks/                      # Research notebooks — full E2E experimental workflow
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_chunking.ipynb
│   ├── 03_embedding.ipynb
│   ├── 04_bm25_indexing.ipynb
│   ├── 06_retrieval_pipeline.ipynb
│   └── 07_generation_pipeline.ipynb
│
├── scripts/                        # Data preparation and upload utilities
│   ├── build_sqlite_fts.py         # Build SQLite FTS5 DB from chunks.parquet
│   ├── upload_bm25_hub.py          # Upload BM25 DB to HuggingFace Hub
│   └── upload_qdrant_cloud.py      # Upload vectors to Qdrant Cloud
│
├── tests/
│   ├── smoke_test.py               # 5-sample end-to-end connectivity test (CI on every push)
│   └── regression_test.py          # LLM-as-a-Judge regression (nightly, 5 rotating samples)
│
├── .github/workflows/
│   ├── ci.yml                      # Lint + smoke test (triggers on microservices-migration branch)
│   └── nightly-regression.yml      # Daily regression test at 9:00 AM ICT
│
├── render.yaml                     # Render deployment configuration
└── README.md
```

---

## SSE Event Protocol

The backend streams a strict sequence of Server-Sent Events:

```
event: progress      data: {"step": "embedding",     "label": "...", "eta_s": 2}
event: progress      data: {"step": "sparse_search",  "label": "...", "eta_s": 1}
event: progress      data: {"step": "rerank",          "label": "...", "eta_s": 3}
event: sources       data: {"sources": [{chunk_id, text, title, url, score}, ...]}
event: token         data: {"token": "...", "model": "Gemini"}   ← repeated N times
event: done          data: {"full_answer": "...", "model": "Gemini"}
event: error         data: {"message": "...", "detail": "..."}   ← on any failure
```

If `decompose=true` is set in the request body, an additional event fires before `progress`:
```
event: decomposition data: {"original": "...", "sub_queries": ["...", "..."]}
```

---

## Local Development

**Prerequisites:** Python 3.11+, Node.js 20+, API keys (see [Environment Variables](#environment-variables))

### 1. Clone & Setup

```bash
git clone https://github.com/thong7d/rag-vn-finance.git
cd rag-vn-finance
```

### 2. Run Backend

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
# Fill in your API keys in .env

uvicorn main:app --reload --port 8000
# Swagger UI: http://localhost:8000/docs
```

### 3. Run Frontend

```bash
cd frontend
npm install
cp .env.example .env.local
# Set VITE_API_URL=http://localhost:8000

npm run dev
# http://localhost:5173
```

---

## Production Deployment

### Backend → Render

1. Connect repo to [Render](https://render.com)
2. **Root Directory:** `backend`
3. **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Add all env vars from `backend/.env.example` in the Render Dashboard

### Frontend → Vercel

1. Import repo into [Vercel](https://vercel.com)
2. **Framework Preset:** Vite
3. **Root Directory:** `frontend`
4. Add environment variable: `VITE_API_URL=https://your-backend.onrender.com`

---

## Environment Variables

### Backend (`backend/.env`)

| Variable | Description | Source |
|---|---|---|
| `GEMINI_API_KEY` | Google Gemini API (primary LLM + query decomposition) | [aistudio.google.com](https://aistudio.google.com/apikey) |
| `GEMINI_API_KEY_2` | Secondary Gemini key for query decomposition (optional) | [aistudio.google.com](https://aistudio.google.com/apikey) |
| `MISTRAL_API_KEY` | Mistral API (LLM fallback layer 2) | [console.mistral.ai](https://console.mistral.ai/api-keys/) |
| `COHERE_API_KEY` | Cohere Reranker API | [dashboard.cohere.com](https://dashboard.cohere.com/api-keys) |
| `HF_TOKEN` | HuggingFace token (embedding inference + BM25 DB download) | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| `QDRANT_URL` | Qdrant Cloud cluster URL | [cloud.qdrant.io](https://cloud.qdrant.io) |
| `QDRANT_API_KEY` | Qdrant Cloud API key | [cloud.qdrant.io](https://cloud.qdrant.io) |
| `HF_BM25_REPO` | HF Dataset repo containing the SQLite FTS5 DB | `thong7d/rag-vn-finance-bm25` |
| `CHUNK_STRATEGY` | Chunking strategy used at index time | `sentence_aware` |

### CI/CD Secrets (`GitHub → Settings → Secrets`)

| Secret | Description |
|---|---|
| `RENDER_BACKEND_URL` | Public URL of the deployed backend (e.g. `https://rag-vn-finance-backend.onrender.com`) |
| `GROQ_API_KEY` | Groq API key for nightly regression judge (Qwen 3.6 27B) |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
Built with ❤️ by <a href="https://github.com/thong7d">thong7d</a>
</div>
