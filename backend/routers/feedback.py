"""
feedback.py — Endpoint for submitting user feedback (thumbs up/down).
"""

from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.audit_logger import save_feedback

router = APIRouter()


class FeedbackRequest(BaseModel):
    message_id: str
    user_id: str
    vote: str  # 'up' or 'down'
    comment: Optional[str] = None


@router.post("/feedback")
async def submit_user_feedback(req: FeedbackRequest):
    """
    POST /api/feedback
    Record user thumbs up/down and optional comment for a chat response.
    """
    if req.vote not in ("up", "down"):
        raise HTTPException(status_code=400, detail="vote must be 'up' or 'down'")

    res = await save_feedback(
        message_id=req.message_id,
        user_id=req.user_id,
        vote=req.vote,
        comment=req.comment,
    )
    return res
