"""Tests for the progress tracker structured item support."""

import json
from pathlib import Path

from backend.services.progress_tracker import ProgressTracker


class TestProgressTrackerAddItem:
    def test_text_only_item(self, tmp_path: Path):
        tracker = ProgressTracker(tmp_path)
        tracker.start_step("s1", "Step 1")
        tracker.add_item("s1", "hello")

        data = json.loads((tmp_path / "progress.json").read_text())
        items = data["steps"][0]["items"]
        assert len(items) == 1
        assert items[0] == {"text": "hello"}

    def test_structured_item_includes_non_none_fields(self, tmp_path: Path):
        tracker = ProgressTracker(tmp_path)
        tracker.start_step("s1", "Step 1")
        tracker.add_item(
            "s1",
            "Found result",
            item_type="search_result",
            title="Oslo / emissions = 42",
            domain="example.com",
            url="https://example.com/page",
        )

        data = json.loads((tmp_path / "progress.json").read_text())
        item = data["steps"][0]["items"][0]
        assert item["text"] == "Found result"
        assert item["item_type"] == "search_result"
        assert item["title"] == "Oslo / emissions = 42"
        assert item["domain"] == "example.com"
        assert item["url"] == "https://example.com/page"

    def test_none_fields_omitted(self, tmp_path: Path):
        tracker = ProgressTracker(tmp_path)
        tracker.start_step("s1", "Step 1")
        tracker.add_item("s1", "gap item", item_type="gap", title="Oslo")

        data = json.loads((tmp_path / "progress.json").read_text())
        item = data["steps"][0]["items"][0]
        assert "domain" not in item
        assert "url" not in item
        assert "count" not in item
        assert "metadata" not in item
        assert item["item_type"] == "gap"
        assert item["title"] == "Oslo"

    def test_count_and_metadata(self, tmp_path: Path):
        tracker = ProgressTracker(tmp_path)
        tracker.start_step("s1", "Step 1")
        tracker.add_item(
            "s1",
            "Batch summary",
            item_type="batch_summary",
            count=5,
            metadata={"cities": "Oslo, Berlin", "batch": "1/3"},
        )

        data = json.loads((tmp_path / "progress.json").read_text())
        item = data["steps"][0]["items"][0]
        assert item["count"] == 5
        assert item["metadata"]["cities"] == "Oslo, Berlin"

    def test_backward_compat_with_existing_callers(self, tmp_path: Path):
        """Existing callers passing only (step_id, text) still work."""
        tracker = ProgressTracker(tmp_path)
        tracker.start_step("s1", "Step 1")
        tracker.add_item("s1", "plain text")
        tracker.add_item("s1", "another one")

        data = json.loads((tmp_path / "progress.json").read_text())
        items = data["steps"][0]["items"]
        assert len(items) == 2
        assert items[0] == {"text": "plain text"}
        assert items[1] == {"text": "another one"}

    def test_ignores_unknown_step_id(self, tmp_path: Path):
        tracker = ProgressTracker(tmp_path)
        tracker.start_step("s1", "Step 1")
        tracker.add_item("nonexistent", "should be ignored")

        data = json.loads((tmp_path / "progress.json").read_text())
        items = data["steps"][0]["items"]
        assert len(items) == 0
