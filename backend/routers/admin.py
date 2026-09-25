"""
admin.py — Observability & Admin metrics endpoint.
"""

from fastapi import APIRouter

from services.audit_logger import get_admin_metrics

router = APIRouter()


@router.get("/admin/metrics")
async def fetch_admin_metrics():
    """
    GET /api/admin/metrics
    Fetch system-wide observability metrics, model usage breakdown, recent feedback & eval logs.
    """
    metrics = await get_admin_metrics()
    return metrics
