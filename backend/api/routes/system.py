"""System status HTTP endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from backend.api.models import VectorStoreWarmupResponse
from backend.api.services.vector_store_warmup import VectorStoreWarmup

router = APIRouter()


def _get_vector_store_warmup(request: Request) -> VectorStoreWarmup:
    """Return vector-store warm-up service from FastAPI app state."""
    warmup = getattr(request.app.state, "vector_store_warmup", None)
    if not isinstance(warmup, VectorStoreWarmup):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Vector store warm-up service is not initialized.",
        )
    return warmup


@router.get("/system/vector-store", response_model=VectorStoreWarmupResponse)
def get_vector_store_status(request: Request) -> VectorStoreWarmupResponse:
    """Return startup vector-store warm-up status."""
    return VectorStoreWarmupResponse.model_validate(
        _get_vector_store_warmup(request).snapshot()
    )


__all__ = ["router"]
