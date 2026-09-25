"""
audit_logger.py — Fire-and-forget async logger for Enterprise Observability.

Logs chat messages and session info to PostgreSQL (Neon) in the background.
All writes are non-fatal: any exception is caught and logged as a warning.

Usage pattern (fire-and-forget):
    asyncio.create_task(log_chat_message(...))
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from core.logging import setup_logger
from db.engine import AsyncSessionLocal, is_db_enabled
from db.models import ChatMessage, ChatSession

logger = setup_logger("AuditLogger")


async def _upsert_session(session, session_id: str, user_id: str, first_question: str) -> None:
    """Create chat session if not exists, else update updated_at."""
    from sqlalchemy import select, update

    existing = await session.execute(
        select(ChatSession).where(ChatSession.id == session_id)
    )
    row = existing.scalar_one_or_none()

    if row is None:
        new_session = ChatSession(
            id=session_id,
            user_id=user_id,
            first_question=first_question[:500] if first_question else None,
        )
        session.add(new_session)
    else:
        await session.execute(
            update(ChatSession)
            .where(ChatSession.id == session_id)
            .values(updated_at=datetime.now(timezone.utc))
        )


async def log_chat_message(
    session_id: str,
    user_id: str,
    message_id: str,
    question: str,
    answer: str,
    model_used: Optional[str],
    latency_ms: Optional[int],
    decompose_enabled: bool,
    sub_queries: Optional[list[str]],
    sources: list[dict],
) -> None:
    """
    Write one chat Q&A pair to PostgreSQL.
    Called via asyncio.create_task() — runs in background, never blocks pipeline.

    Args:
        session_id:       Browser session UUID (from localStorage)
        user_id:          Browser user UUID (from localStorage)
        message_id:       Pre-generated UUID for this message (returned in done event)
        question:         User's original question
        answer:           Full generated answer text
        model_used:       LLM layer that succeeded (e.g., "Gemini")
        latency_ms:       Total pipeline latency in milliseconds
        decompose_enabled: Whether query decomposition was requested
        sub_queries:      List of sub-queries if decomposed, else None
        sources:          List of source dicts from retrieval pipeline
    """
    if not is_db_enabled():
        return

    try:
        # Extract lightweight source references (IDs + scores only)
        source_ids = [s.get("chunk_id") or s.get("id", "") for s in sources if s]
        source_scores = {
            (s.get("chunk_id") or s.get("id", "")): s.get("score", 0.0)
            for s in sources if s
        }
        source_count = len(sources)

        async with AsyncSessionLocal() as db_session:
            # Upsert session first (FK constraint)
            await _upsert_session(db_session, session_id, user_id, question)

            # Insert chat message
            msg = ChatMessage(
                id=message_id,
                session_id=session_id,
                question=question,
                answer=answer,
                model_used=model_used,
                latency_ms=latency_ms,
                decompose_enabled=decompose_enabled,
                sub_queries=sub_queries,
                source_ids=source_ids,
                source_scores=source_scores,
                source_count=source_count,
            )
            db_session.add(msg)
            await db_session.commit()

        logger.info(f"AuditLogger: logged message {message_id[:8]}... | model={model_used} | latency={latency_ms}ms")

    except Exception as e:
        # Non-fatal: log warning and continue
        logger.warning(f"AuditLogger: failed to log chat message (non-fatal): {e}")


async def log_evaluation(
    run_type: str,
    question: str,
    answer: str,
    context_snippet: Optional[str],
    faithfulness: Optional[float],
    answer_relevancy: Optional[float],
    judge_model: Optional[str],
    status: str = "PASSED",
) -> None:
    """
    Write one evaluation result to PostgreSQL evaluation_logs table.
    Can be called from regression_test.py or manually.
    """
    if not is_db_enabled():
        return

    from db.models import EvaluationLog

    try:
        async with AsyncSessionLocal() as db_session:
            log = EvaluationLog(
                id=str(uuid.uuid4()),
                run_type=run_type,
                question=question,
                answer=answer,
                context_snippet=context_snippet[:500] if context_snippet else None,
                faithfulness=faithfulness,
                answer_relevancy=answer_relevancy,
                judge_model=judge_model,
                status=status,
            )
            db_session.add(log)
            await db_session.commit()
        logger.info(f"AuditLogger: logged evaluation [{status}] | {run_type}")
    except Exception as e:
        logger.warning(f"AuditLogger: failed to log evaluation (non-fatal): {e}")
