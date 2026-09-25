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


async def save_feedback(
    message_id: str,
    user_id: str,
    vote: str,
    comment: Optional[str] = None,
) -> dict:
    """Save or update user feedback (thumbs up/down) for a message."""
    if not is_db_enabled():
        return {"status": "disabled", "message": "Database not configured"}

    from db.models import UserFeedback
    from sqlalchemy import select

    try:
        async with AsyncSessionLocal() as db_session:
            existing = await db_session.execute(
                select(UserFeedback).where(
                    UserFeedback.message_id == message_id,
                    UserFeedback.user_id == user_id,
                )
            )
            row = existing.scalar_one_or_none()

            if row:
                row.vote = vote
                if comment is not None:
                    row.comment = comment
            else:
                feedback = UserFeedback(
                    id=str(uuid.uuid4()),
                    message_id=message_id,
                    user_id=user_id,
                    vote=vote,
                    comment=comment,
                )
                db_session.add(feedback)

            await db_session.commit()
            logger.info(f"AuditLogger: saved feedback [{vote}] for msg {message_id[:8]}...")
            return {"status": "ok", "message": "Feedback recorded"}
    except Exception as e:
        logger.warning(f"AuditLogger: failed to save feedback: {e}")
        return {"status": "error", "message": str(e)}


async def get_user_sessions(user_id: str, limit: int = 20) -> list[dict]:
    """Fetch recent chat sessions for a specific user ID."""
    if not is_db_enabled():
        return []

    from db.models import ChatSession
    from sqlalchemy import select

    try:
        async with AsyncSessionLocal() as db_session:
            stmt = (
                select(ChatSession)
                .where(ChatSession.user_id == user_id)
                .order_by(ChatSession.updated_at.desc())
                .limit(limit)
            )
            result = await db_session.execute(stmt)
            sessions = result.scalars().all()

            return [
                {
                    "session_id": s.id,
                    "first_question": s.first_question or "New Chat",
                    "created_at": s.created_at.isoformat() if s.created_at else None,
                    "updated_at": s.updated_at.isoformat() if s.updated_at else None,
                }
                for s in sessions
            ]
    except Exception as e:
        logger.warning(f"AuditLogger: failed to fetch sessions: {e}")
        return []


async def get_session_messages(session_id: str) -> list[dict]:
    """Fetch all messages for a given session ID."""
    if not is_db_enabled():
        return []

    from db.models import ChatMessage
    from sqlalchemy import select

    try:
        async with AsyncSessionLocal() as db_session:
            stmt = (
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at.asc())
            )
            result = await db_session.execute(stmt)
            messages = result.scalars().all()

            return [
                {
                    "message_id": m.id,
                    "session_id": m.session_id,
                    "question": m.question,
                    "answer": m.answer,
                    "model_used": m.model_used,
                    "latency_ms": m.latency_ms,
                    "decompose_enabled": m.decompose_enabled,
                    "sub_queries": m.sub_queries,
                    "source_count": m.source_count,
                    "created_at": m.created_at.isoformat() if m.created_at else None,
                }
                for m in messages
            ]
    except Exception as e:
        logger.warning(f"AuditLogger: failed to fetch session messages: {e}")
        return []


async def get_admin_metrics() -> dict:
    """Compute aggregate observability metrics for the Admin Dashboard."""
    if not is_db_enabled():
        return {"enabled": False, "message": "Database not configured"}

    from db.models import ChatMessage, ChatSession, UserFeedback, EvaluationLog
    from sqlalchemy import select, func

    try:
        async with AsyncSessionLocal() as db_session:
            total_chats = (await db_session.execute(select(func.count()).select_from(ChatMessage))).scalar() or 0
            total_sessions = (await db_session.execute(select(func.count()).select_from(ChatSession))).scalar() or 0

            # Feedbacks
            up_votes = (
                await db_session.execute(
                    select(func.count()).select_from(UserFeedback).where(UserFeedback.vote == "up")
                )
            ).scalar() or 0
            down_votes = (
                await db_session.execute(
                    select(func.count()).select_from(UserFeedback).where(UserFeedback.vote == "down")
                )
            ).scalar() or 0

            # Latency
            avg_latency = (
                await db_session.execute(select(func.avg(ChatMessage.latency_ms)))
            ).scalar() or 0.0

            # Model usage breakdown
            model_counts_raw = (
                await db_session.execute(
                    select(ChatMessage.model_used, func.count(ChatMessage.id)).group_by(ChatMessage.model_used)
                )
            ).all()
            model_breakdown = {m: c for m, c in model_counts_raw if m}

            # Recent messages
            recent_msgs_raw = (
                await db_session.execute(
                    select(ChatMessage).order_by(ChatMessage.created_at.desc()).limit(10)
                )
            ).scalars().all()

            recent_messages = [
                {
                    "message_id": m.id,
                    "question": m.question,
                    "model_used": m.model_used,
                    "latency_ms": m.latency_ms,
                    "created_at": m.created_at.isoformat() if m.created_at else None,
                }
                for m in recent_msgs_raw
            ]

            # Recent evaluation logs
            eval_logs_raw = (
                await db_session.execute(
                    select(EvaluationLog).order_by(EvaluationLog.created_at.desc()).limit(10)
                )
            ).scalars().all()

            recent_evaluations = [
                {
                    "id": ev.id,
                    "run_type": ev.run_type,
                    "question": ev.question,
                    "faithfulness": ev.faithfulness,
                    "answer_relevancy": ev.answer_relevancy,
                    "status": ev.status,
                    "run_date": ev.run_date.isoformat() if ev.run_date else None,
                }
                for ev in eval_logs_raw
            ]

            return {
                "enabled": True,
                "total_chats": total_chats,
                "total_sessions": total_sessions,
                "up_votes": up_votes,
                "down_votes": down_votes,
                "positive_rate": round(up_votes / (up_votes + down_votes) * 100, 1) if (up_votes + down_votes) > 0 else 0,
                "avg_latency_ms": round(float(avg_latency), 1),
                "model_breakdown": model_breakdown,
                "recent_messages": recent_messages,
                "recent_evaluations": recent_evaluations,
            }
    except Exception as e:
        logger.warning(f"AuditLogger: failed to compute admin metrics: {e}")
        return {"enabled": True, "error": str(e)}

