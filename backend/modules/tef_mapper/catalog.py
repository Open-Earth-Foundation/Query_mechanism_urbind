from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from backend.modules.tef_mapper.models import (
    TefSectorCard,
    TefSectorKey,
    TefSubsectorCard,
    TefTransitionElement,
)

SOURCE_COMMIT_PATTERN = re.compile(r"Source commit:\s*`([^`]+)`")


def _read_json_list(path: Path) -> list[Any]:
    """Read a JSON list from disk or raise a clear validation error."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list at {path}")
    return payload


def _group_by_parent(
    subcategories: list[TefSubsectorCard],
) -> dict[str, list[TefSubsectorCard]]:
    """Group category records by their direct parent path."""
    grouped: dict[str, list[TefSubsectorCard]] = {}
    for subcategory in subcategories:
        grouped.setdefault(subcategory.parent_path, []).append(subcategory)
    return grouped


def _group_transitions_by_path(
    transitions: list[TefTransitionElement],
) -> dict[str, list[TefTransitionElement]]:
    """Group Transition Elements by their direct category path."""
    grouped: dict[str, list[TefTransitionElement]] = {}
    for transition in transitions:
        grouped.setdefault(transition.path, []).append(transition)
    return grouped


class TefCatalog:
    """Local JSON-backed TEF catalog used by the staged mapper."""

    def __init__(self, root: Path) -> None:
        self.root = root
        catalog_root = root / "catalog"
        self.sectors = [
            TefSectorCard.model_validate(item)
            for item in _read_json_list(catalog_root / "sectors.json")
        ]
        self.subcategories = [
            TefSubsectorCard.model_validate(item)
            for item in _read_json_list(catalog_root / "subcategories.json")
        ]
        self.subsubcategories = [
            TefSubsectorCard.model_validate(item)
            for item in _read_json_list(catalog_root / "subsubcategories.json")
        ]
        self.transition_records = [
            TefTransitionElement.model_validate(item)
            for item in _read_json_list(catalog_root / "transition_elements.json")
        ]
        self.sectors_by_key = {sector.sector: sector for sector in self.sectors}
        self.sectors_by_path = {sector.path: sector for sector in self.sectors}
        self.subcategories_by_parent = _group_by_parent(self.subcategories)
        self.subsubcategories_by_parent = _group_by_parent(self.subsubcategories)
        self.subcategories_by_path = {
            subcategory.path: subcategory
            for subcategory in [*self.subcategories, *self.subsubcategories]
        }
        self.transitions_by_path = _group_transitions_by_path(self.transition_records)

    @property
    def source_version(self) -> str:
        """Return the TEF source commit recorded in SOURCE.md when available."""
        source_path = self.root / "SOURCE.md"
        if not source_path.exists():
            return "unknown"
        match = SOURCE_COMMIT_PATTERN.search(source_path.read_text(encoding="utf-8"))
        return match.group(1) if match else "unknown"

    def sector_path(self, sector: TefSectorKey) -> str:
        """Return the catalog path for one root sector."""
        return self.sectors_by_key[sector].path

    def child_subsectors(self, parent_path: str) -> list[TefSubsectorCard]:
        """Return direct child categories for a catalog path."""
        if parent_path in self.sectors_by_path:
            return self.subcategories_by_parent.get(parent_path, [])
        return self.subsubcategories_by_parent.get(parent_path, [])

    def child_subcategories(self, parent_path: str) -> list[TefSubsectorCard]:
        """Return direct child subcategories for compatibility with catalog naming."""
        return self.child_subsectors(parent_path)

    def category_payload(self, path: str) -> dict[str, Any]:
        """Return JSON-serializable metadata for a sector or subcategory path."""
        if path in self.sectors_by_path:
            return self.sectors_by_path[path].model_dump(mode="json")
        return self.subcategories_by_path[path].model_dump(mode="json")

    def total_transition_count(self, path: str) -> int:
        """Return total Transition Element count for a sector or category branch."""
        if path in self.sectors_by_path:
            return self.sectors_by_path[path].total_transition_count
        return self.subcategories_by_path[path].total_transition_count

    def direct_transition_count(self, path: str) -> int:
        """Return direct Transition Element count for a sector or category path."""
        if path in self.sectors_by_path:
            return 0
        return self.subcategories_by_path[path].direct_transition_count

    def sector_cards(self) -> list[dict[str, Any]]:
        """Return root sector cards enriched with direct child labels."""
        cards: list[dict[str, Any]] = []
        for sector in self.sectors:
            children = self.child_subsectors(sector.path)
            payload = sector.model_dump(mode="json")
            payload["child_subcategories"] = [
                {
                    "path": child.path,
                    "label": child.label,
                    "direct_transition_count": child.direct_transition_count,
                    "total_transition_count": child.total_transition_count,
                    "has_transition_elements": child.has_transition_elements,
                }
                for child in children
            ]
            cards.append(payload)
        return cards

    def subsector_cards(self, parent_path: str) -> list[dict[str, Any]]:
        """Return direct child category cards for one parent path."""
        return [child.model_dump(mode="json") for child in self.child_subsectors(parent_path)]

    def subcategory_cards(self, parent_path: str) -> list[dict[str, Any]]:
        """Return direct child subcategory cards for compatibility with catalog naming."""
        return self.subsector_cards(parent_path)

    def transition_elements(self, path: str) -> list[TefTransitionElement]:
        """Return direct Transition Elements for one catalog category path."""
        return self.transitions_by_path.get(path, [])


__all__ = ["TefCatalog"]
