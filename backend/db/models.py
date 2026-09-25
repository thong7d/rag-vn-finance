"""
models.py — SQLAlchemy 2.0 ORM models for Enterprise Observability Engine.
Includes: ChatSession, ChatMessage, UserFeedback, EvaluationLog.
"""

import uuid
from datetime import datetime
from typing import Any, List, Optional

from sqlalchemy import String, Text, Integer, Float, Boolean, DateTime, ForeignKey, JSON, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""
    pass


class ChatSession(Base):
    """Stores user chat sessions identified by anonymous browser UUID."""
    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    first_question: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    messages: Mapped[List["ChatMessage"]] = relationship(
        "ChatMessage", back_populates="session", cascade="all, delete-orphan", order_by="ChatMessage.created_at"
    )


class ChatMessage(Base):
    """Stores Q&A pairs, retrieval metadata, and performance metrics."""
    __tablename__ = "chat_messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True, nullable=False
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    model_used: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    decompose_enabled: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    sub_queries: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    source_ids: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    source_scores: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    source_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True, nullable=False)

    session: Mapped["ChatSession"] = relationship("ChatSession", back_populates="messages")
    feedbacks: Mapped[List["UserFeedback"]] = relationship(
        "UserFeedback", back_populates="message", cascade="all, delete-orphan"
    )


class UserFeedback(Base):
    """Stores user votes (thumbs up/down) and comments on RAG answers."""
    __tablename__ = "user_feedbacks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    message_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("chat_messages.id", ondelete="CASCADE"), index=True, nullable=False
    )
    user_id: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    vote: Mapped[str] = mapped_column(String(10), nullable=False)  # 'up' or 'down'
    comment: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    message: Mapped["ChatMessage"] = relationship("ChatMessage", back_populates="feedbacks")


class EvaluationLog(Base):
    """Audit log for automated regression tests (RAGAS / LLM-as-a-Judge)."""
    __tablename__ = "evaluation_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    run_type: Mapped[str] = mapped_column(String(50), index=True, nullable=False)  # e.g., 'nightly_ci', 'manual_test'
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    context_snippet: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    faithfulness: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    answer_relevancy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    judge_model: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="PASSED", nullable=False)  # 'PASSED' or 'FAILED'
    run_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
