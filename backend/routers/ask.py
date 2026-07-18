"""
ask.py — FastAPI router for POST /api/ask and GET /api/health.
"""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core.logging import setup_logger
from services import retrieval as retrieval_service
from services.generation import stream_answer

logger = setup_logger("AskRouter")
router = APIRouter()


class AskRequest(BaseModel):
    question: str


@router.post("/api/ask")
async def ask(request: AskRequest):
    """
    Main RAG endpoint.

    Runs retrieval synchronously (fast, <5s), then streams LLM tokens as SSE.

    SSE event types:
      - sources: {sources: [...]}       — sent first, before any tokens
      - token:   {token: str, model: str}
      - done:    {full_answer: str, model: str}
      - error:   {message: str, detail: str}
    """
    question = request.question.strip()
    if not question:
        async def _empty():
            import json
            yield f"event: error\ndata: {json.dumps({'message': 'Empty question'})}\n\n"
        return StreamingResponse(_empty(), media_type="text/event-stream")

    logger.info(f"Question: {question[:80]}...")

    # Run retrieval (sync, fast)
    sources = retrieval_service.retrieve(question)

    # Stream generation (async SSE)
    return StreamingResponse(
        stream_answer(question, sources),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",       # disable Nginx buffering on Render
            "Access-Control-Allow-Origin": "*",
        },
    )


@router.get("/api/health")
async def health():
    """Liveness check — used by Render health checks and frontend status bar."""
    return {"status": "ok", "service": "rag-vn-finance-backend"}
