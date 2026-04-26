from __future__ import annotations

import sys
from pathlib import Path

import pytest

from backend.scripts import map_initiatives_to_tef as script
from tests.support import build_test_app_config


class _FakeRunResult:
    """Minimal run result for script orchestration tests."""

    def __init__(self, run_id: str, output_dir: Path) -> None:
        self.run_id = run_id
        self.output_dir = str(output_dir)

    def model_dump(self, mode: str = "json") -> dict[str, str]:
        """Return the logging payload shape used by the CLI."""
        return {"run_id": self.run_id, "output_dir": self.output_dir}


def _patch_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch runtime setup that is not relevant to CLI orchestration."""
    config = build_test_app_config()
    monkeypatch.setattr(script, "setup_logger", lambda: None)
    monkeypatch.setattr(script, "load_dotenv", lambda: None)
    monkeypatch.setattr(script, "load_config", lambda _path: config)
    monkeypatch.setattr(script, "resolve_openrouter_api_key", lambda: "test-api-key")


def test_default_run_extracts_selected_cities_before_mapping(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Default CLI mode should extract from Markdown and then map the extraction output."""
    _patch_runtime(monkeypatch)
    extraction_root = tmp_path / "initiative_extraction"
    mapping_root = tmp_path / "tef_mapping"
    markdown_path = tmp_path / "documents"
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_extract_initiatives(**kwargs: object) -> _FakeRunResult:
        calls.append(("extract", kwargs))
        return _FakeRunResult("full_run", extraction_root / "full_run")

    def fake_map_initiatives_to_tef(**kwargs: object) -> _FakeRunResult:
        calls.append(("map", kwargs))
        return _FakeRunResult("full_run", mapping_root / "full_run")

    monkeypatch.setattr(script, "extract_initiatives", fake_extract_initiatives)
    monkeypatch.setattr(script, "map_initiatives_to_tef", fake_map_initiatives_to_tef)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "map_initiatives_to_tef",
            "--markdown-path",
            str(markdown_path),
            "--city",
            "Krakow",
            "--city",
            "Munich",
            "--run-id",
            "full_run",
            "--extraction-output-dir",
            str(extraction_root),
            "--output-dir",
            str(mapping_root),
            "--extraction-max-workers",
            "2",
            "--max-workers",
            "3",
        ],
    )

    script.main()

    assert [call[0] for call in calls] == ["extract", "map"]
    extract_kwargs = calls[0][1]
    assert extract_kwargs["markdown_path"] == markdown_path
    assert extract_kwargs["output_root"] == extraction_root
    assert extract_kwargs["run_id"] == "full_run"
    assert extract_kwargs["selected_cities"] == ["Krakow", "Munich"]
    assert extract_kwargs["max_workers"] == 2

    map_kwargs = calls[1][1]
    assert map_kwargs["extraction_run_dir"] == extraction_root / "full_run"
    assert map_kwargs["initiatives_jsonl"] is None
    assert map_kwargs["output_root"] == mapping_root
    assert map_kwargs["run_id"] == "full_run"
    assert map_kwargs["selected_cities"] == ["Krakow", "Munich"]
    assert map_kwargs["max_workers"] == 3


def test_mapping_only_skips_extraction_and_processes_all_cities(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Mapping-only mode should use existing extracted initiatives and no city filter by default."""
    _patch_runtime(monkeypatch)
    extraction_run_dir = tmp_path / "initiative_extraction" / "existing_run"
    mapping_root = tmp_path / "tef_mapping"
    calls: list[dict[str, object]] = []

    def fail_extract_initiatives(**_kwargs: object) -> _FakeRunResult:
        raise AssertionError("mapping-only mode should not run extraction")

    def fake_map_initiatives_to_tef(**kwargs: object) -> _FakeRunResult:
        calls.append(kwargs)
        return _FakeRunResult("mapping_only", mapping_root / "mapping_only")

    monkeypatch.setattr(script, "extract_initiatives", fail_extract_initiatives)
    monkeypatch.setattr(script, "map_initiatives_to_tef", fake_map_initiatives_to_tef)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "map_initiatives_to_tef",
            "--mapping-only",
            "--extraction-run-dir",
            str(extraction_run_dir),
            "--output-dir",
            str(mapping_root),
            "--run-id",
            "mapping_only",
        ],
    )

    script.main()

    assert len(calls) == 1
    assert calls[0]["extraction_run_dir"] == extraction_run_dir
    assert calls[0]["initiatives_jsonl"] is None
    assert calls[0]["selected_cities"] is None
    assert calls[0]["run_id"] == "mapping_only"


def test_mapping_only_requires_existing_mapping_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mapping-only mode should reject calls without an existing extraction source."""
    monkeypatch.setattr(sys, "argv", ["map_initiatives_to_tef", "--mapping-only"])

    with pytest.raises(SystemExit):
        script.parse_args()
