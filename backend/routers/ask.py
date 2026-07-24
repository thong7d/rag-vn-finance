"""
ask.py — FastAPI router for POST /api/ask and GET /api/health.

Pipeline (async orchestration with SSE progress events):
  1. event: progress  {step: "embedding",     label: ..., eta_s: 2}
  2. dense_retrieve()  — HF Inference API + Qdrant
  3. event: progress  {step: "sparse_search", label: ..., eta_s: 1}
  4. sparse_retrieve() — SQLite FTS5 BM25
  5. event: progress  {step: "rerank",        label: ..., eta_s: 3}
  6. fuse_and_rerank() — RRF + Cohere Rerank
  7. stream_answer()  — sources → token × N → done

On any step failure:
  event: error {message: str (user-friendly VN), detail: str (raw)}
"""

import asyncio
import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core.config import get_settings
from core.logging import setup_logger
from services import retrieval as retrieval_service
from services.generation import stream_answer

logger = setup_logger("AskRouter")
router = APIRouter()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _classify_error(e: Exception) -> str:
    """Map low-level exceptions to user-friendly Vietnamese error messages."""
    msg = str(e).lower()
    if any(k in msg for k in ["nameresolutionerror", "no address", "failed to resolve", "gaierror", "dns"]):
        return "🔌 Dịch vụ embedding tạm thời gián đoạn (lỗi DNS). Vui lòng thử lại sau ít phút."
    if "timeout" in msg:
        return "⏱️ Quá thời gian chờ kết nối dịch vụ. Vui lòng thử lại."
    if any(k in msg for k in ["429", "rate limit", "toomanyrequests", "quota"]):
        return "⚠️ Đã đạt giới hạn request API. Vui lòng thử lại sau vài phút."
    if any(k in msg for k in ["401", "unauthorized", "invalid token"]):
        return "🔑 Lỗi xác thực API key. Vui lòng kiểm tra cấu hình máy chủ."
    if any(k in msg for k in ["503", "service unavailable", "connection refused"]):
        return "🔌 Dịch vụ bên ngoài tạm thời không khả dụng. Vui lòng thử lại."
    if any(k in msg for k in ["connectionerror", "connection error", "max retries"]):
        return "🔌 Không thể kết nối đến dịch vụ bên ngoài. Vui lòng thử lại."
    return "❌ Lỗi không xác định trong quá trình truy xuất. Vui lòng thử lại."


# ── Main Pipeline Generator ───────────────────────────────────────────────────

async def _run_pipeline(question: str):
    """
    Full async RAG pipeline as an SSE generator.

    Event sequence:
      progress (embedding) → progress (sparse_search) → progress (rerank)
      → sources → token × N → done
      → error (on failure)
    """
    settings = get_settings()

    # ── Step 1: Dense retrieval (includes HF embedding + Qdrant search) ──────
    yield _sse("progress", {
        "step": "embedding",
        "label": "Đang vector hóa câu hỏi...",
        "eta_s": 2,
    })
    try:
        dense_results = await asyncio.to_thread(
            retrieval_service.dense_retrieve,
            question,
            settings.top_k_hybrid,
        )
    except Exception as e:
        logger.error(f"Dense retrieval failed: {e}")
        yield _sse("error", {"message": _classify_error(e), "detail": str(e)})
        return

    # ── Step 2: Sparse retrieval (SQLite FTS5 BM25 — fast, non-fatal) ────────
    yield _sse("progress", {
        "step": "sparse_search",
        "label": "Tìm kiếm BM25 trên SQLite FTS5...",
        "eta_s": 1,
    })
    try:
        sparse_results = await asyncio.to_thread(
            retrieval_service.sparse_retrieve,
            question,
            settings.top_k_hybrid,
        )
    except Exception as e:
        logger.warning(f"Sparse retrieval failed (non-fatal, dense-only fallback): {e}")
        sparse_results = []

    # ── Step 3: RRF fusion + Cohere Rerank ───────────────────────────────────
    yield _sse("progress", {
        "step": "rerank",
        "label": "Cohere reranking top-30 kết quả...",
        "eta_s": 3,
    })
    try:
        sources = await asyncio.to_thread(
            retrieval_service.fuse_and_rerank,
            question,
            dense_results,
            sparse_results,
        )
    except Exception as e:
        logger.error(f"Rerank failed: {e}")
        yield _sse("error", {"message": _classify_error(e), "detail": str(e)})
        return

    # ── Step 4: LLM Generation (async SSE stream) ─────────────────────────────
    async for chunk in stream_answer(question, sources):
        yield chunk


# ── Routes ────────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str


@router.post("/api/ask")
async def ask(request: AskRequest):
    """
    Main RAG endpoint — streams SSE events.

    Full event sequence:
      progress: {step, label, eta_s}   × 3  (embedding, sparse_search, rerank)
      sources:  {sources: [...]}
      token:    {token: str, model: str}  × N
      done:     {full_answer: str, model: str}
      error:    {message: str, detail: str}   — on any failure
    """
    question = request.question.strip()
    if not question:
        async def _empty():
            yield _sse("error", {"message": "Câu hỏi không được để trống.", "detail": ""})
        return StreamingResponse(_empty(), media_type="text/event-stream")

    logger.info(f"Question: {question[:80]}...")

    return StreamingResponse(
        _run_pipeline(question),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable Nginx buffering on Render
        },
    )


@router.get("/api/health")
async def health():
    """Liveness check — used by Render health checks and frontend status bar."""
    return {"status": "ok", "service": "rag-vn-finance-backend"}
