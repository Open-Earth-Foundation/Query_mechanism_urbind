"""Tests for the pdf_to_markdown ingestion handler.

Uses a stub ``convert_fn`` so we don't depend on pymupdf4llm or any real
PDF binary in unit tests.  An end-to-end run against the upstream repo
is exercised by the script CLI in integration / manual runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.modules.sources.handlers import IngestionContext
from backend.modules.sources.handlers.pdf_to_markdown import run_pdf_to_markdown
from backend.modules.sources.manifest import (
    IngestionConfig,
    IngestionInputs,
    IngestionOutput,
    SourceConfig,
)


def _make_pdf_fixture(tmp_path: Path) -> Path:
    """Build a fake "upstream" tier-1 layout under tmp_path.

    Returns the upstream root.
    """
    root = tmp_path / "upstream"
    (root / "tier-1-city-plans" / "dresden").mkdir(parents=True)
    (root / "tier-1-city-plans" / "munich").mkdir(parents=True)
    (root / "tier-1-city-plans" / "dresden" / "plan-a.pdf").write_bytes(b"%PDF-fake-a")
    (root / "tier-1-city-plans" / "dresden" / "plan-b.pdf").write_bytes(b"%PDF-fake-b")
    (root / "tier-1-city-plans" / "munich" / "strategy.pdf").write_bytes(b"%PDF-fake-c")
    (root / "tier-1-city-plans" / "munich" / "notes.txt").write_text("not a pdf")
    return root


def _make_context(
    *,
    upstream_root: Path,
    project_root: Path,
    excludes: list[str] | None = None,
) -> IngestionContext:
    source = SourceConfig(
        id="urbind_additional",
        name="URBIND Additional Documentation",
        provider="github",
        repo="Open-Earth-Foundation/urbind-additional-documentation",
        pinned_commit="c0a3993583068b73f57f1ad99023eb7727a4de14",
    )
    ingestion = IngestionConfig(
        id="urbind_tier1_cities",
        kind="markdown_corpus",
        inputs=IngestionInputs(
            paths=["tier-1-city-plans/"],
            formats=["pdf"],
            excludes=excludes or [],
        ),
        output=IngestionOutput(
            path="documents/additional/",
            naming="{city}_{slug}/{city}.md",
        ),
        handler="ingest.pdf_to_markdown",
        converter="pymupdf4llm",
        consumed_by="markdown_researcher",
    )
    return IngestionContext(
        source=source,
        ingestion=ingestion,
        upstream_root=upstream_root,
        project_root=project_root,
        resolved_commit="c0a3993583068b73f57f1ad99023eb7727a4de14",
    )


def test_converts_pdfs_with_city_and_slug_naming(tmp_path: Path) -> None:
    upstream = _make_pdf_fixture(tmp_path)
    project_root = tmp_path / "project"
    ctx = _make_context(upstream_root=upstream, project_root=project_root)

    state = run_pdf_to_markdown(
        ctx,
        convert_fn=lambda b: f"# Stubbed\n\nbytes={len(b)}",
    )

    out_dir = project_root / "documents" / "additional"
    # Subdir-per-doc so each output's filename stem equals the city stem
    # — required by the markdown researcher's stem-based city filter.
    expected = sorted([
        out_dir / "dresden_plan-a" / "dresden.md",
        out_dir / "dresden_plan-b" / "dresden.md",
        out_dir / "munich_strategy" / "munich.md",
    ])
    written = sorted(out_dir.rglob("*.md"))
    assert written == expected
    for path in written:
        assert path.stem in {"dresden", "munich"}

    sample = (out_dir / "dresden_plan-a" / "dresden.md").read_text(encoding="utf-8")
    assert sample.startswith("# Stubbed")

    assert state.ingestion_id == "urbind_tier1_cities"
    assert state.source_id == "urbind_additional"
    assert state.last_ingested_at is not None
    assert state.source_commit == "c0a3993583068b73f57f1ad99023eb7727a4de14"

    files = state.model_dump()["files"]
    assert {f["city"] for f in files} == {"dresden", "munich"}
    assert all(f["input_sha"] for f in files)
    assert all(f["output_sha"] for f in files)
    assert state.model_dump()["file_count"] == 3


def test_skips_non_pdf_files(tmp_path: Path) -> None:
    upstream = _make_pdf_fixture(tmp_path)
    ctx = _make_context(upstream_root=upstream, project_root=tmp_path / "project")
    state = run_pdf_to_markdown(ctx, convert_fn=lambda b: "ok")
    files = state.model_dump()["files"]
    assert all(f["input"].endswith(".pdf") for f in files)


def test_excludes_are_honoured(tmp_path: Path) -> None:
    upstream = _make_pdf_fixture(tmp_path)
    ctx = _make_context(
        upstream_root=upstream,
        project_root=tmp_path / "project",
        excludes=["tier-1-city-plans/munich/strategy.pdf"],
    )
    state = run_pdf_to_markdown(ctx, convert_fn=lambda b: "ok")
    files = state.model_dump()["files"]
    cities = {f["city"] for f in files}
    assert cities == {"dresden"}
    assert state.model_dump()["file_count"] == 2


def test_missing_input_path_logs_warning_and_returns_empty(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    ctx = _make_context(upstream_root=tmp_path / "no-such-upstream", project_root=project_root)
    state = run_pdf_to_markdown(ctx, convert_fn=lambda b: "ok")
    assert state.model_dump()["file_count"] == 0
    assert (project_root / "documents" / "additional").exists()


def test_handler_is_registered() -> None:
    from backend.modules.sources.handlers import REGISTRY

    assert "ingest.pdf_to_markdown" in REGISTRY


def test_output_requires_path(tmp_path: Path) -> None:
    upstream = _make_pdf_fixture(tmp_path)
    ctx = _make_context(upstream_root=upstream, project_root=tmp_path / "project")
    ctx.ingestion.output.path = None
    with pytest.raises(ValueError, match="output.path"):
        run_pdf_to_markdown(ctx, convert_fn=lambda b: "ok")
