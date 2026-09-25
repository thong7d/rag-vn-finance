"""
sessions.py — Endpoints for listing user chat sessions and session message history.
"""

from fastapi import APIRouter, Query

from services.audit_logger import get_user_sessions, get_session_messages

router = APIRouter()


@router.get("/sessions")
async def list_user_sessions(user_id: str = Query(...), limit: int = Query(20, ge=1, le=100)):
    """
    GET /api/sessions?user_id=...&limit=20
    Fetch recent sessions for a user ID.
    """
    sessions = await get_user_sessions(user_id=user_id, limit=limit)
    return {"sessions": sessions}


@router.get("/sessions/{session_id}/messages")
async def list_session_messages(session_id: str):
    """
    GET /api/sessions/{session_id}/messages
    Fetch all Q&A messages for a given session.
    """
    messages = await get_session_messages(session_id=session_id)
    return {"messages": messages}
