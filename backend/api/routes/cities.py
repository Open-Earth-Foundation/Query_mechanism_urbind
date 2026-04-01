"""City list HTTP endpoint."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, status

from backend.api.models import (
    CityGroupListResponse,
    CityListResponse,
    CityMarkdownResponse,
)
from backend.api.services.city_catalog import (
    list_city_names,
    load_city_groups,
    load_city_markdown,
)

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


@router.get("/cities/{city_name}/markdown", response_model=CityMarkdownResponse)
def get_city_markdown(city_name: str, request: Request) -> CityMarkdownResponse:
    """Return concatenated markdown source content for one city."""
    markdown_dir = getattr(request.app.state, "markdown_dir", Path("documents"))
    if not isinstance(markdown_dir, Path):
        markdown_dir = Path(str(markdown_dir))

    city_markdown = load_city_markdown(markdown_dir, city_name)
    if city_markdown is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"City markdown for `{city_name}` was not found.",
        )

    resolved_city_name, content, source_paths = city_markdown
    return CityMarkdownResponse(
        city_name=resolved_city_name,
        content=content,
        source_paths=[str(path) for path in source_paths],
    )


__all__ = ["router"]
