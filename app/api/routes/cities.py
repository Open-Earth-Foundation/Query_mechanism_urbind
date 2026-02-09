"""City list HTTP endpoint."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request

from app.api.models import CityGroupListResponse, CityListResponse
from app.api.services import list_city_names, load_city_groups

router = APIRouter()


@router.get("/cities", response_model=CityListResponse)
def get_cities(request: Request) -> CityListResponse:
    """Return available city names based on markdown filenames."""
    markdown_dir = getattr(request.app.state, "markdown_dir", Path("documents"))
    if not isinstance(markdown_dir, Path):
        markdown_dir = Path(str(markdown_dir))

    cities = list_city_names(markdown_dir)
    return CityListResponse(
        cities=cities,
        total=len(cities),
        markdown_dir=str(markdown_dir),
    )


@router.get("/city-groups", response_model=CityGroupListResponse)
def get_city_groups(request: Request) -> CityGroupListResponse:
    """Return predefined city groups loaded from JSON catalog."""
    markdown_dir = getattr(request.app.state, "markdown_dir", Path("documents"))
    groups_path = getattr(
        request.app.state,
        "city_groups_path",
        Path(__file__).resolve().parents[1] / "assets" / "city_groups.json",
    )
    if not isinstance(markdown_dir, Path):
        markdown_dir = Path(str(markdown_dir))
    if not isinstance(groups_path, Path):
        groups_path = Path(str(groups_path))

    available_cities = list_city_names(markdown_dir)
    groups = load_city_groups(groups_path, available_cities)
    return CityGroupListResponse(
        groups=groups,
        total=len(groups),
        groups_path=str(groups_path),
    )


__all__ = ["router"]
