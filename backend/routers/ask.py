"""
ask.py — FastAPI router for POST /api/ask and GET /api/health.

Pipeline (async orchestration with SSE progress events):
  1. [If decompose] event: decomposition {sub_queries: [...]}
  2. event: progress  {step: "embedding",     label: ..., eta_s: 2}
  3. dense_retrieve()  — HF Inference API + Qdrant
  4. event: progress  {step: "sparse_search", label: ..., eta_s: 1}
  5. sparse_retrieve() — SQLite FTS5 BM25
  6. event: progress  {step: "rerank",        label: ..., eta_s: 3}
  7. fuse_and_rerank() — RRF + Cohere Rerank
  8. stream_answer()  — sources → token × N → done

On any step failure:
  event: error {message: str (user-friendly VN), detail: str (raw)}
"""

import asyncio
import json
import time
import uuid

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core.config import get_settings
from core.logging import setup_logger
from services import retrieval as retrieval_service
from services.decomposer import decompose_query
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
        return "🔌 Embedding service temporarily unavailable (DNS error). Please try again in a few minutes."
    if "timeout" in msg:
        return "⏱️ Connection to service timed out. Please try again."
    if any(k in msg for k in ["429", "rate limit", "toomanyrequests", "quota"]):
        return "⚠️ API rate limit reached. Please try again in a few minutes."
    if any(k in msg for k in ["401", "unauthorized", "invalid token"]):
        return "🔑 API authentication error. Please check server configuration."
    if any(k in msg for k in ["503", "service unavailable", "connection refused"]):
        return "🔌 External service temporarily unavailable. Please try again."
    if any(k in msg for k in ["connectionerror", "connection error", "max retries"]):
        return "🔌 Unable to connect to external service. Please try again."
    return "❌ Unknown retrieval error. Please try again."


# ── Main Pipeline Generator ───────────────────────────────────────────────────

async def _run_pipeline(
    question: str,
    decompose: bool,
    session_id: str,
    user_id: str,
    message_id: str,
):
    """
    Full async RAG pipeline as an SSE generator.

    Event sequence:
      [decomposition] → progress (embedding) → progress (sparse_search) → progress (rerank)
      → sources → token × N → done (includes message_id)
      → error (on failure)
    """
    settings = get_settings()
    t0 = time.perf_counter()

    # Collected for audit logging
    final_answer = ""
    final_model = ""
    final_sources: list[dict] = []
    final_sub_queries: list[str] | None = None

    # ── Step 0 (optional): Query Decomposition ─────────────────────────────
    queries = [question]  # default: single query
    if decompose:
        yield _sse("progress", {
            "step": "decompose",
            "label": "Đang phân tích câu hỏi phức hợp...",
            "eta_s": 3,
        })
        try:
            sub_queries = await asyncio.to_thread(decompose_query, question)
            queries = sub_queries
            final_sub_queries = sub_queries
            yield _sse("decomposition", {
                "original": question,
                "sub_queries": sub_queries,
            })
        except Exception as e:
            logger.warning(f"Decomposition failed, using original question: {e}")
            queries = [question]

    # ── Step 1: Dense retrieval (includes HF embedding + Qdrant search) ──────
    yield _sse("progress", {
        "step": "embedding",
        "label": f"Đang vector hóa {'câu hỏi' if len(queries) == 1 else f'{len(queries)} sub-queries'}...",
        "eta_s": 2 * len(queries),
    })

    all_dense = []
    try:
        for q in queries:
            dense = await asyncio.to_thread(
                retrieval_service.dense_retrieve,
                q,
                settings.top_k_hybrid,
            )
            all_dense.extend(dense)
    except Exception as e:
        logger.error(f"Dense retrieval failed: {e}")
        yield _sse("error", {"message": _classify_error(e), "detail": str(e)})
        return

    # ── Step 2: Sparse retrieval (SQLite FTS5 BM25 — fast, non-fatal) ────────
    yield _sse("progress", {
        "step": "sparse_search",
        "label": "BM25 full-text search on SQLite FTS5...",
        "eta_s": 1,
    })

    all_sparse = []
    try:
        for q in queries:
            sparse = await asyncio.to_thread(
                retrieval_service.sparse_retrieve,
                q,
                settings.top_k_hybrid,
            )
            all_sparse.extend(sparse)
    except Exception as e:
        logger.warning(f"Sparse retrieval failed (non-fatal, dense-only fallback): {e}")

    # ── Deduplicate by chunk_id (keep highest score) ─────────────────────────
    if decompose and len(queries) > 1:
        dense_dedup = {}
        for cid, score in all_dense:
            if cid not in dense_dedup or score > dense_dedup[cid]:
                dense_dedup[cid] = score
        all_dense = list(dense_dedup.items())

        sparse_dedup = {}
        for cid, score in all_sparse:
            if cid not in sparse_dedup or score > sparse_dedup[cid]:
                sparse_dedup[cid] = score
        all_sparse = list(sparse_dedup.items())

    # ── Step 3: RRF fusion + Cohere Rerank ───────────────────────────────────
    yield _sse("progress", {
        "step": "rerank",
        "label": "Cohere reranking top-30 candidates...",
        "eta_s": 3,
    })
    try:
        sources = await asyncio.to_thread(
            retrieval_service.fuse_and_rerank,
            question,  # rerank against original question for relevance
            all_dense,
            all_sparse,
        )
        final_sources = sources
    except Exception as e:
        logger.error(f"Rerank failed: {e}")
        yield _sse("error", {"message": _classify_error(e), "detail": str(e)})
        return

    # ── Step 4: LLM Generation (async SSE stream) ─────────────────────────────
    async for chunk in stream_answer(question, sources, message_id=message_id):
        # Capture final answer and model from done event
        if chunk.startswith("event: done"):
            try:
                data_line = [ln for ln in chunk.split("\n") if ln.startswith("data: ")]
                if data_line:
                    done_data = json.loads(data_line[0][6:])
                    final_answer = done_data.get("full_answer", "")
                    final_model = done_data.get("model", "")
            except Exception:
                pass
        yield chunk

    # ── Step 5: Fire-and-forget async audit log ───────────────────────────────
    latency_ms = int((time.perf_counter() - t0) * 1000)
    if final_answer:
        try:
            from services.audit_logger import log_chat_message
            asyncio.create_task(log_chat_message(
                session_id=session_id,
                user_id=user_id,
                message_id=message_id,
                question=question,
                answer=final_answer,
                model_used=final_model or None,
                latency_ms=latency_ms,
                decompose_enabled=decompose,
                sub_queries=final_sub_queries,
                sources=final_sources,
            ))
        except Exception as e:
            logger.warning(f"Failed to schedule audit log task (non-fatal): {e}")


# ── Routes ────────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str
    decompose: bool = False
    session_id: str = ""   # browser session UUID (from localStorage)
    user_id: str = ""      # browser user UUID (from localStorage)


@router.post("/api/ask")
async def ask(request: AskRequest):
    """
    Main RAG endpoint — streams SSE events.

    Full event sequence:
      [decomposition: {sub_queries}]        — only if decompose=true
      progress: {step, label, eta_s}   × 3  (embedding, sparse_search, rerank)
      sources:  {sources: [...]}
      token:    {token: str, model: str}  × N
      done:     {full_answer: str, model: str, message_id: str}
      error:    {message: str, detail: str}   — on any failure
    """
    question = request.question.strip()
    if not question:
        async def _empty():
            yield _sse("error", {"message": "Question cannot be empty.", "detail": ""})
        return StreamingResponse(_empty(), media_type="text/event-stream")

    # Generate session and user IDs if not provided by frontend
    session_id = request.session_id.strip() or str(uuid.uuid4())
    user_id = request.user_id.strip() or f"anon_{str(uuid.uuid4())[:8]}"
    message_id = str(uuid.uuid4())  # unique ID for this Q&A pair

    logger.info(f"Question: {question[:80]}... | decompose={request.decompose} | session={session_id[:8]}...")

    return StreamingResponse(
        _run_pipeline(
            question,
            decompose=request.decompose,
            session_id=session_id,
            user_id=user_id,
            message_id=message_id,
        ),
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
