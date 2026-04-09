"""Tests for the PDF export service and endpoint."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from backend.api.services.pdf_export import markdown_to_pdf, strip_citation_markers


class TestStripCitationMarkers:
    def test_removes_single_marker(self):
        assert strip_citation_markers("hello [ref_1] world") == "hello  world"

    def test_removes_multiple_markers(self):
        text = "data [ref_12] is [ref_3] here"
        assert strip_citation_markers(text) == "data  is  here"

    def test_leaves_non_ref_brackets(self):
        text = "array[0] and [link](url)"
        assert strip_citation_markers(text) == "array[0] and [link](url)"

    def test_empty_string(self):
        assert strip_citation_markers("") == ""

    def test_no_markers(self):
        text = "no markers here"
        assert strip_citation_markers(text) == text


class TestMarkdownToPdf:
    def test_produces_pdf_bytes(self):
        md = "# Hello World\n\nThis is a test document.\n"
        result = markdown_to_pdf(md)
        assert isinstance(result, bytes)
        assert result[:5] == b"%PDF-"

    def test_strips_citations_before_render(self):
        md = "Value is 42 [ref_1] and growing [ref_2].\n"
        result = markdown_to_pdf(md)
        assert isinstance(result, bytes)
        assert result[:5] == b"%PDF-"

    def test_handles_table(self):
        md = "| City | Value |\n|------|-------|\n| Oslo | 100 |\n"
        result = markdown_to_pdf(md)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_handles_code_block(self):
        md = "```python\nprint('hello')\n```\n"
        result = markdown_to_pdf(md)
        assert isinstance(result, bytes)

    def test_handles_unicode_smart_quotes(self):
        md = "The city\u2019s plan includes \u201csmart\u201d infrastructure \u2013 and more.\n"
        result = markdown_to_pdf(md)
        assert isinstance(result, bytes)
        assert result[:5] == b"%PDF-"

    def test_handles_mixed_unicode_characters(self):
        md = "Arrows \u2192 bullets \u2022 ellipsis\u2026 and em\u2014dash.\n"
        result = markdown_to_pdf(md)
        assert isinstance(result, bytes)
        assert result[:5] == b"%PDF-"

    def test_handles_bold_in_table_cells(self):
        md = "| City | **Value** |\n|------|-------|\n| **Oslo** | 100 |\n"
        result = markdown_to_pdf(md)
        assert isinstance(result, bytes)
        assert result[:5] == b"%PDF-"

    def test_handles_links_and_emphasis_in_table_cells(self):
        md = (
            "| Source | Notes |\n"
            "|--------|-------|\n"
            "| [link](https://example.com) | *important* and `code` |\n"
        )
        result = markdown_to_pdf(md)
        assert isinstance(result, bytes)
        assert result[:5] == b"%PDF-"


class TestPdfExportEndpoint:
    """Integration tests for GET /runs/{run_id}/export/pdf."""

    @pytest.fixture()
    def app_client(self, tmp_path: Path):
        from fastapi.testclient import TestClient

        from backend.api.main import create_app
        from backend.utils.paths import create_run_paths
        from tests.support import build_test_app_config

        runs_dir = tmp_path / "runs"
        markdown_dir = tmp_path / "markdown"
        runs_dir.mkdir()
        markdown_dir.mkdir()

        config = build_test_app_config(
            runs_dir=runs_dir,
            markdown_dir=markdown_dir,
            enable_sql=False,
        )

        run_id = "pdf-test-run"
        paths = create_run_paths(runs_dir, run_id, config.orchestrator.context_bundle_name)
        paths.base_dir.mkdir(parents=True, exist_ok=True)

        context_bundle = {"sql": None, "markdown": {"status": "success", "excerpts": []}}
        paths.context_bundle.write_text(
            json.dumps(context_bundle), encoding="utf-8"
        )
        paths.final_output.write_text(
            "# Test Report\n\nCity data [ref_1] is shown.\n", encoding="utf-8"
        )
        run_log = {
            "run_id": run_id,
            "question": "test question",
            "status": "completed",
            "started_at": datetime.now(tz=timezone.utc).isoformat(),
            "completed_at": datetime.now(tz=timezone.utc).isoformat(),
            "finish_reason": "completed (write)",
            "artifacts": {
                "context_bundle": str(paths.context_bundle),
                "final_output": str(paths.final_output),
            },
        }
        paths.run_log.write_text(
            json.dumps(run_log), encoding="utf-8"
        )

        app = create_app(runs_dir=runs_dir, markdown_dir=markdown_dir, max_workers=1)
        with TestClient(app) as client:
            yield client, run_id

    def test_returns_pdf(self, app_client):
        client, run_id = app_client
        response = client.get(f"/api/v1/runs/{run_id}/export/pdf")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"
        assert response.content[:5] == b"%PDF-"
        assert "content-disposition" in response.headers

    def test_404_for_missing_run(self, app_client):
        client, _ = app_client
        response = client.get("/api/v1/runs/nonexistent/export/pdf")
        assert response.status_code == 404
