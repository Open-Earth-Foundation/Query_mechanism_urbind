from __future__ import annotations

import json
from pathlib import Path

from backend.utils.artifact_manifest import resolve_manifest_alias
from backend.utils.artifact_writer import (
    ARTIFACT_SCHEMA_VERSION,
    ArtifactWriter,
    resolve_stage_number,
)


def test_artifact_writer_pairs_summary_event_stage_detail_and_manifest(
    tmp_path: Path,
) -> None:
    writer = ArtifactWriter(tmp_path, "run-1")

    writer.write_event("decision_recorded", {"step": "enrichment", "status": "completed"})
    event_index = writer.write_event(
        "stage_completed",
        {"step": "retrieval", "metrics": {"retrieval_total_chunks": 2}},
        stage_name="retrieval",
    )
    detail_path = writer.write_step_detail(
        "retrieval",
        {
            "inputs": {"queries": ["solar"]},
            "outputs": {"retrieved_count": 2},
            "metrics": {"retrieval_total_chunks": 2},
        },
        event_index=event_index,
        stage_number=resolve_stage_number("retrieval"),
    )
    payload_path = writer.write_stage_file(
        "retrieval",
        "chunks_full.json",
        {"chunks": [{"chunk_id": "chunk-1"}]},
        alias="retrieval_chunks_full",
    )
    manifest_path = writer.write_manifest({"status": "completed"})

    summary_lines = (tmp_path / "summary.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(summary_lines) == 2
    summary_event = json.loads(summary_lines[1])
    assert summary_event["schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert summary_event["event_index"] == event_index
    assert summary_event["stage_number"] == 3
    assert summary_event["payload"]["metrics"]["retrieval_total_chunks"] == 2

    detail_payload = json.loads(detail_path.read_text(encoding="utf-8"))
    assert detail_path.name == "003_retrieval.json"
    assert detail_payload["event_index"] == event_index
    assert detail_payload["stage_number"] == 3
    assert detail_payload["outputs"]["retrieved_count"] == 2

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["aliases"]["stage_retrieval"]["path"] == "stages/003_retrieval.json"
    assert (
        manifest["aliases"]["retrieval_chunks_full"]["path"]
        == "stage_files/003_retrieval/chunks_full.json"
    )
    assert "summary.jsonl" in manifest["generated_files"]
    assert payload_path.exists()


def test_resolve_manifest_alias_returns_run_local_path(tmp_path: Path) -> None:
    writer = ArtifactWriter(tmp_path, "run-1")
    expected = writer.write_stage_file(
        "markdown_batching",
        "source_chunk_index.json",
        {"source_chunks": []},
        alias="source_chunk_index",
    )
    writer.write_manifest()

    assert resolve_manifest_alias(tmp_path, "source_chunk_index") == expected
    assert resolve_manifest_alias(tmp_path, "missing") is None


def test_resolve_manifest_alias_rejects_parent_traversal(tmp_path: Path) -> None:
    outside_path = tmp_path.parent / "outside.json"
    outside_path.write_text("{}", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"aliases": {"outside": {"path": "../outside.json"}}}),
        encoding="utf-8",
    )

    assert resolve_manifest_alias(tmp_path, "outside") is None


def test_resolve_manifest_alias_rejects_absolute_paths_outside_run_dir(tmp_path: Path) -> None:
    outside_path = tmp_path.parent / "outside.json"
    outside_path.write_text("{}", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"aliases": {"outside": {"path": str(outside_path)}}}),
        encoding="utf-8",
    )

    assert resolve_manifest_alias(tmp_path, "outside") is None
