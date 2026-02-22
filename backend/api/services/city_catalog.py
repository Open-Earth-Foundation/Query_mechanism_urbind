"""City catalog utilities derived from markdown files."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from backend.utils.city_normalization import normalize_city_key


def list_city_names(markdown_dir: Path) -> list[str]:
    """Return unique city names based on markdown file stems."""
    if not markdown_dir.exists():
        return []

    names = {
        markdown_file.stem
        for markdown_file in markdown_dir.rglob("*.md")
        if markdown_file.is_file()
    }
    return sorted(names, key=str.casefold)


def index_city_markdown_files(markdown_dir: Path) -> dict[str, list[Path]]:
    """Index markdown files by city stem name (case-insensitive key)."""
    if not markdown_dir.exists():
        return {}

    index: dict[str, list[Path]] = {}
    for markdown_file in sorted(markdown_dir.rglob("*.md")):
        if not markdown_file.is_file():
            continue
        key = normalize_city_key(markdown_file.stem)
        index.setdefault(key, []).append(markdown_file)
    return index


def build_city_subset(
    source_markdown_dir: Path,
    target_markdown_dir: Path,
    selected_cities: list[str],
) -> list[Path]:
    """Copy selected city markdown files into a run-local subset directory."""
    index = index_city_markdown_files(source_markdown_dir)
    requested = {
        normalize_city_key(city)
        for city in selected_cities
        if isinstance(city, str) and city.strip()
    }
    if not requested:
        return []

    copied_paths: list[Path] = []
    for city_key in sorted(requested):
        for source_path in index.get(city_key, []):
            relative_path = source_path.relative_to(source_markdown_dir)
            target_path = target_markdown_dir / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)
            copied_paths.append(target_path)
    return copied_paths


def load_city_groups(groups_path: Path, available_cities: list[str]) -> list[dict[str, object]]:
    """Load and sanitize predefined city groups from JSON."""
    if not groups_path.exists():
        return []

    try:
        raw = json.loads(groups_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(raw, dict):
        raw_groups = raw.get("groups", [])
    elif isinstance(raw, list):
        raw_groups = raw
    else:
        raw_groups = []
    if not isinstance(raw_groups, list):
        return []

    available_by_key = {normalize_city_key(city): city for city in available_cities}
    groups: list[dict[str, object]] = []
    seen_ids: set[str] = set()

    for item in raw_groups:
        normalized = _normalize_group_item(item, available_by_key)
        if normalized is None:
            continue
        group_id = str(normalized["id"])
        if group_id in seen_ids:
            continue
        seen_ids.add(group_id)
        groups.append(normalized)
    return groups


def _normalize_group_item(
    item: object, available_by_key: dict[str, str]
) -> dict[str, object] | None:
    """Validate one group payload and intersect with available city names."""
    if not isinstance(item, dict):
        return None
    group_id = item.get("id")
    name = item.get("name")
    cities_raw = item.get("cities")
    if not isinstance(group_id, str) or not group_id.strip():
        return None
    if not isinstance(name, str) or not name.strip():
        return None
    if not isinstance(cities_raw, list):
        return None

    matched_cities: list[str] = []
    seen_city_keys: set[str] = set()
    for city in cities_raw:
        if not isinstance(city, str):
            continue
        city_key = normalize_city_key(city)
        if not city_key or city_key in seen_city_keys:
            continue
        if city_key in available_by_key:
            matched_cities.append(available_by_key[city_key])
            seen_city_keys.add(city_key)
    if not matched_cities:
        return None

    description = item.get("description")
    return {
        "id": group_id.strip(),
        "name": name.strip(),
        "description": description.strip() if isinstance(description, str) else None,
        "cities": matched_cities,
    }


__all__ = [
    "list_city_names",
    "index_city_markdown_files",
    "build_city_subset",
    "load_city_groups",
]
