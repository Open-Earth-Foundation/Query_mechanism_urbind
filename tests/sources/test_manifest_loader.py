"""Tests for the data sources manifest loader and state helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from backend.modules.sources.manifest import (
    Manifest,
    load_manifest,
)
from backend.modules.sources.state import (
    IngestionState,
    load_state,
    save_state,
    state_path,
)


def _write_yaml(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------


def test_load_empty_manifest(tmp_path: Path) -> None:
    manifest_file = _write_yaml(tmp_path / "manifest.yaml", {"version": 1, "sources": []})
    manifest = load_manifest(manifest_file)
    assert manifest.version == 1
    assert manifest.sources == []
    assert manifest.iter_ingestions() == []


def test_load_manifest_with_full_source(tmp_path: Path) -> None:
    payload = {
        "version": 1,
        "sources": [
            {
                "id": "urbind_additional",
                "name": "URBIND Additional Documentation",
                "provider": "github",
                "repo": "Open-Earth-Foundation/urbind-additional-documentation",
                "pinned_commit": "9a3c1f8e2b",
                "ingestions": [
                    {
                        "id": "urbind_tier1_cities",
                        "kind": "markdown_corpus",
                        "inputs": {
                            "paths": ["tier-1-city-plans/"],
                            "formats": ["pdf"],
                        },
                        "output": {
                            "path": "documents/additional/",
                            "naming": "{city}_{slug}.md",
                        },
                        "handler": "ingest.pdf_to_markdown",
                        "consumed_by": "markdown_researcher",
                        "coverage": {
                            "cities": ["aachen", "dresden"],
                        },
                    },
                ],
            }
        ],
    }
    manifest = load_manifest(_write_yaml(tmp_path / "manifest.yaml", payload))

    source = manifest.get_source("urbind_additional")
    assert source is not None
    assert source.repo == "Open-Earth-Foundation/urbind-additional-documentation"
    assert source.pinned_commit == "9a3c1f8e2b"

    found = manifest.get_ingestion("urbind_tier1_cities")
    assert found is not None
    parent_source, ingestion = found
    assert parent_source.id == "urbind_additional"
    assert ingestion.kind == "markdown_corpus"
    assert ingestion.inputs.paths == ["tier-1-city-plans/"]
    assert ingestion.coverage is not None
    assert ingestion.coverage.cities == ["aachen", "dresden"]


def test_missing_manifest_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path / "nope.yaml")


def test_invalid_yaml_raises(tmp_path: Path) -> None:
    bad = tmp_path / "manifest.yaml"
    bad.write_text("version: 1\nsources: [\n  this is broken", encoding="utf-8")
    with pytest.raises(yaml.YAMLError):
        load_manifest(bad)


def test_missing_required_field_raises(tmp_path: Path) -> None:
    payload = {
        "version": 1,
        "sources": [
            {
                "id": "broken",
                # missing `name`, `provider`
                "ingestions": [],
            }
        ],
    }
    with pytest.raises(Exception):
        load_manifest(_write_yaml(tmp_path / "manifest.yaml", payload))


def test_github_provider_requires_repo(tmp_path: Path) -> None:
    payload = {
        "version": 1,
        "sources": [
            {
                "id": "no_repo",
                "name": "No Repo",
                "provider": "github",
                # repo intentionally omitted
                "ingestions": [],
            }
        ],
    }
    with pytest.raises(Exception):
        load_manifest(_write_yaml(tmp_path / "manifest.yaml", payload))


def test_duplicate_source_id_raises(tmp_path: Path) -> None:
    payload = {
        "version": 1,
        "sources": [
            {"id": "dupe", "name": "A", "provider": "local", "ingestions": []},
            {"id": "dupe", "name": "B", "provider": "local", "ingestions": []},
        ],
    }
    with pytest.raises(Exception):
        load_manifest(_write_yaml(tmp_path / "manifest.yaml", payload))


def test_duplicate_ingestion_id_raises(tmp_path: Path) -> None:
    payload = {
        "version": 1,
        "sources": [
            {
                "id": "src_a",
                "name": "A",
                "provider": "local",
                "ingestions": [
                    {
                        "id": "shared_id",
                        "kind": "markdown_corpus",
                        "inputs": {"paths": ["a/"], "formats": ["markdown"]},
                        "output": {"path": "documents/a/"},
                    }
                ],
            },
            {
                "id": "src_b",
                "name": "B",
                "provider": "local",
                "ingestions": [
                    {
                        "id": "shared_id",
                        "kind": "markdown_corpus",
                        "inputs": {"paths": ["b/"], "formats": ["markdown"]},
                        "output": {"path": "documents/b/"},
                    }
                ],
            },
        ],
    }
    with pytest.raises(Exception):
        load_manifest(_write_yaml(tmp_path / "manifest.yaml", payload))


def test_repo_manifest_loads() -> None:
    """The committed manifest at backend/data/sources_manifest.yaml must validate."""
    repo_root = Path(__file__).resolve().parents[2]
    manifest = load_manifest(repo_root / "backend" / "data" / "sources_manifest.yaml")
    assert isinstance(manifest, Manifest)
    assert manifest.version >= 1


# ---------------------------------------------------------------------------
# State files
# ---------------------------------------------------------------------------


def test_state_path_under_root(tmp_path: Path) -> None:
    assert state_path("ing_foo", tmp_path) == tmp_path / "ing_foo.json"


def test_state_save_and_load_roundtrip(tmp_path: Path) -> None:
    state = IngestionState(
        ingestion_id="ing_foo",
        source_id="src_foo",
        last_ingested_at=datetime(2026, 4, 25, 8, 30, tzinfo=timezone.utc).isoformat(),
        source_commit="abc1234567890",
        file_count=8,
    )
    written = save_state(state, tmp_path)
    assert written == tmp_path / "ing_foo.json"

    loaded = load_state("ing_foo", tmp_path)
    assert loaded is not None
    assert loaded.ingestion_id == "ing_foo"
    assert loaded.source_id == "src_foo"
    assert loaded.source_commit == "abc1234567890"
    # Extras survive the roundtrip thanks to ConfigDict(extra="allow").
    assert loaded.model_dump().get("file_count") == 8


def test_load_state_missing_returns_none(tmp_path: Path) -> None:
    assert load_state("never_run", tmp_path) is None
