<div align="center">

# 📈 RAG Finance VN

**Hệ thống hỏi đáp thông minh về Tài chính Việt Nam**  
Retrieval-Augmented Generation (RAG) chuyên biệt cho báo chí tài chính tiếng Việt

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19-61dafb?logo=react)](https://react.dev)

**[🚀 Live Demo](https://rag-vn-finance.vercel.app)** · **[🤗 Gradio Demo](https://huggingface.co/spaces/thong7d/financial-news-rag)** · **[📦 BM25 Dataset](https://huggingface.co/datasets/thong7d/rag-vn-finance-bm25)**

</div>

---

## Tổng quan

RAG Finance VN là hệ thống Q&A tự động được huấn luyện trên **10.000+ bài báo tài chính Việt Nam** (2015–2024). Hệ thống truy xuất đoạn văn bản liên quan từ kho dữ liệu và sinh câu trả lời có trích dẫn nguồn, giúp người dùng tra cứu thông tin tài chính nhanh chóng và có thể kiểm chứng.

### Kiến trúc hệ thống

```
┌─────────────────┐     SSE Stream     ┌──────────────────────────────────────────┐
│  React Frontend │ ◄─────────────── ▶ │  FastAPI Backend                         │
│  (Vercel)       │                    │  (Render)                                 │
└─────────────────┘                    │                                           │
                                       │  POST /api/ask → _run_pipeline()          │
                                       │  ┌──────────────────────────────────┐     │
                                       │  │ 1. HF Inference API (Embedding)  │     │
                                       │  │ 2. Qdrant Cloud (Dense Search)   │     │
                                       │  │ 3. SQLite FTS5 (BM25 Sparse)     │     │
                                       │  │ 4. RRF Fusion + Cohere Rerank    │     │
                                       │  │ 5. LLM Generation (SSE Stream)   │     │
                                       │  └──────────────────────────────────┘     │
                                       └──────────────────────────────────────────┘
                                                        │
                              ┌─────────────────────────┴──────────────────────┐
                              │           External Cloud Services               │
                              │  Qdrant Cloud · HuggingFace Hub · Cohere API   │
                              │  Gemini API · Mistral API (LLM fallback chain)  │
                              └────────────────────────────────────────────────┘
```

### Tính năng nổi bật

| Tính năng | Chi tiết |
|---|---|
| **Hybrid Retrieval** | Dense (multilingual-e5-large) + Sparse (SQLite FTS5 BM25) + RRF Fusion |
| **Cohere Reranking** | Rerank top-30 candidates → top-5 passages |
| **SSE Progress Tracking** | Frontend hiển thị tiến độ realtime từng bước pipeline + ETA countdown |
| **3-Layer LLM Fallback** | Gemini 2.0 Flash Lite → Mistral Small → Gemma 3 27B |
| **Source Citations** | Mỗi câu trả lời đi kèm trích dẫn nguồn có thể kiểm chứng |
| **Error UX** | Phân loại lỗi cụ thể thay vì "Load failed" / "Fail to fetch" |
| **Cold Start Mitigation** | Cron-job keepalive + background metadata loading không block event loop |

---

## Cấu trúc Repository

```
rag-vn-finance/
├── backend/                        # FastAPI Microservice (Deploy: Render)
│   ├── core/
│   │   ├── config.py               # Pydantic Settings (env vars)
│   │   └── logging.py
│   ├── routers/
│   │   └── ask.py                  # POST /api/ask — async SSE pipeline orchestrator
│   ├── services/
│   │   ├── embedding.py            # HuggingFace Inference API (multilingual-e5-large)
│   │   ├── retrieval.py            # Dense + Sparse + RRF + Cohere Rerank
│   │   ├── generation.py           # 3-layer LLM SSE streaming
│   │   └── sqlite_loader.py        # SQLite FTS5 DB download từ HF Hub
│   ├── main.py                     # FastAPI app + lifespan startup
│   ├── requirements.txt
│   ├── .env.example
│   └── Dockerfile
│
├── frontend/                       # React 19 + Vite (Deploy: Vercel)
│   ├── src/
│   │   ├── components/
│   │   │   ├── StatusBar.jsx       # Pipeline progress visualization + ETA countdown
│   │   │   ├── ChatMessage.jsx     # Streaming answer với blinking cursor
│   │   │   ├── ChatInput.jsx
│   │   │   └── SourceCard.jsx      # Trích dẫn nguồn
│   │   ├── hooks/
│   │   │   └── useChat.js          # SSE lifecycle + ETA countdown + error classification
│   │   ├── services/
│   │   │   └── api.js              # Fetch SSE stream wrapper
│   │   └── styles/
│   │       └── index.css           # Bloomberg Terminal dark theme
│   └── package.json
│
├── pipeline/                       # Core RAG Engine (Offline — chạy local/Colab)
│   ├── src/                        # Module Python: chunking, embedding, indexing...
│   │   ├── preprocessing.py        # Text cleaning, EDA, dedup
│   │   ├── chunking.py             # 3 chiến lược: fixed_size, sentence_aware, article_level
│   │   ├── embedding.py            # Batch embedding với multilingual-e5-large
│   │   ├── indexing.py             # FAISS + BM25 indexing
│   │   ├── qdrant_indexing.py      # Upload vectors lên Qdrant Cloud
│   │   ├── retrieval.py            # Hybrid retrieval + Cohere Rerank (local)
│   │   ├── generation.py           # RAG generation (local/Gradio)
│   │   └── evaluation.py           # RAGAS metrics evaluation
│   ├── configs/
│   │   └── config.yaml             # Cấu hình trung tâm toàn bộ pipeline
│   ├── evaluation/                 # Script đánh giá RAGAS & benchmark
│   ├── synthetic_qa/               # Script tạo bộ dữ liệu kiểm thử tổng hợp
│   ├── requirements.txt            # Dependencies offline pipeline
│   └── .env.example
│
├── notebooks/                      # Jupyter Notebooks — quy trình nghiên cứu E2E
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_chunking.ipynb
│   ├── 03_embedding.ipynb
│   ├── 04_bm25_indexing.ipynb
│   ├── 06_retrieval_pipeline.ipynb
│   └── 07_generation_pipeline.ipynb
│
├── scripts/                        # Scripts chuẩn bị & đẩy dữ liệu lên Cloud
│   ├── build_sqlite_fts.py         # Tạo SQLite FTS5 DB từ chunks.parquet
│   ├── upload_bm25_hub.py          # Upload BM25 DB lên HuggingFace Hub
│   └── upload_qdrant_cloud.py      # Upload vectors lên Qdrant Cloud
│
├── .env.example                    # Template biến môi trường
├── .gitignore
├── render.yaml                     # Cấu hình deploy Render
└── README.md
```

---

## Hướng dẫn chạy lại toàn bộ từ đầu

> **Điều kiện tiên quyết:** Python 3.11+, Node.js 18+, các API key (xem phần [Biến môi trường](#biến-môi-trường))

### Bước 0: Clone repo & chuẩn bị dữ liệu

```bash
git clone https://github.com/thong7d/rag-vn-finance.git
cd rag-vn-finance
```

Đặt dataset `financial_news_10k.csv` vào `pipeline/data/raw/`.  
*(Dataset có thể tải từ HuggingFace Hub: `thong7d/rag-vn-finance-bm25`)*

### Bước 1: Cài đặt pipeline (local/Colab)

```bash
cd pipeline
pip install -r requirements.txt
cp .env.example .env
# → Điền API keys vào .env
```

Chạy lần lượt các notebook từ `notebooks/` hoặc dùng trực tiếp module trong `pipeline/src/`.

### Bước 2: Tạo SQLite FTS5 BM25 DB

```bash
python scripts/build_sqlite_fts.py \
    --strategy sentence_aware \
    --repo-id <hf_username>/rag-vn-finance-bm25 \
    --hf-token $HF_TOKEN
```

### Bước 3: Upload vectors lên Qdrant Cloud

```bash
python scripts/upload_qdrant_cloud.py \
    --strategy sentence_aware \
    --cloud-url $QDRANT_URL \
    --api-key $QDRANT_API_KEY
```

### Bước 4: Chạy Backend local

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
# → Điền đầy đủ keys vào .env

uvicorn main:app --reload --port 8000
# API docs: http://localhost:8000/docs
```

### Bước 5: Chạy Frontend local

```bash
cd frontend
npm install
cp .env.example .env.local
# → Đặt VITE_API_URL=http://localhost:8000

npm run dev
# http://localhost:5173
```

---

## Deploy Production

### Backend → Render

1. Fork repo, kết nối với [Render](https://render.com)
2. **Root Directory:** `backend`
3. **Build Command:** `pip install -r requirements.txt`
4. **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Thêm toàn bộ biến môi trường từ `backend/.env.example` vào Render Dashboard

### Frontend → Vercel

1. Import repo vào [Vercel](https://vercel.com)
2. **Framework Preset:** Vite
3. **Root Directory:** `frontend`
4. Thêm biến môi trường:
   ```
   VITE_API_URL=https://your-backend.onrender.com
   ```

---

## Biến môi trường

### Backend (`backend/.env`)

| Biến | Mô tả | Lấy từ |
|---|---|---|
| `GEMINI_API_KEY` | Google Gemini API (LLM chính) | [aistudio.google.com](https://aistudio.google.com/apikey) |
| `MISTRAL_API_KEY` | Mistral API (LLM fallback lớp 2) | [console.mistral.ai](https://console.mistral.ai/api-keys/) |
| `COHERE_API_KEY` | Cohere Reranker | [dashboard.cohere.com](https://dashboard.cohere.com/api-keys) |
| `HF_TOKEN` | HuggingFace token (embedding + BM25 download) | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| `QDRANT_URL` | Qdrant Cloud cluster URL | [cloud.qdrant.io](https://cloud.qdrant.io) |
| `QDRANT_API_KEY` | Qdrant Cloud API key | [cloud.qdrant.io](https://cloud.qdrant.io) |
| `HF_BM25_REPO` | HF Dataset repo chứa SQLite FTS5 DB | `thong7d/rag-vn-finance-bm25` |
| `CHUNK_STRATEGY` | Chiến lược chunking | `sentence_aware` |

### Frontend (`frontend/.env.local`)

| Biến | Mô tả |
|---|---|
| `VITE_API_URL` | URL backend FastAPI (local hoặc Render) |

---

## Ngăn xếp công nghệ

| Lớp | Công nghệ |
|---|---|
| **Embedding** | `intfloat/multilingual-e5-large` via HuggingFace Inference API |
| **Vector Store** | Qdrant Cloud (Free Tier) |
| **Sparse Search** | SQLite FTS5 (BM25 pre-tokenized, hosted on HF Hub) |
| **Reranker** | Cohere `rerank-multilingual-v3.0` |
| **LLM** | Gemini 2.0 Flash Lite → Mistral Small 2506 → Gemma 3 27B |
| **Backend** | FastAPI + Uvicorn (SSE streaming, Python 3.11) |
| **Frontend** | React 19 + Vite 8 (Vanilla CSS, Bloomberg dark theme) |
| **Backend Deploy** | Render (Free Tier) |
| **Frontend Deploy** | Vercel |

---

## Cấu trúc SSE Events

Backend stream các event theo thứ tự:

```
event: progress  data: {"step": "embedding",     "label": "...", "eta_s": 2}
event: progress  data: {"step": "sparse_search",  "label": "...", "eta_s": 1}
event: progress  data: {"step": "rerank",          "label": "...", "eta_s": 3}
event: sources   data: {"sources": [...]}
event: token     data: {"token": "...", "model": "Gemini"}
event: done      data: {"full_answer": "...", "model": "Gemini"}
event: error     data: {"message": "...", "detail": "..."}   ← nếu thất bại
```

---

## License

MIT License — Xem [LICENSE](LICENSE) để biết thêm chi tiết.

---

<div align="center">
Được xây dựng với ❤️ bởi <a href="https://github.com/thong7d">thong7d</a>
</div>
