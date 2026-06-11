"""Thread-safe, best-effort pipeline progress tracker.

Writes ``progress.json`` atomically into the run directory so the status
polling endpoint can return live step information to the frontend.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from backend.utils.artifact_writer import resolve_stage_number

logger = logging.getLogger(__name__)

_VERSION = 1
_PROGRESS_STAGE_ALIASES = {
    "external_sources": "enrichment",
    "gap_analysis": "enrichment",
    "markdown_research": "markdown_extraction",
    "assumptions": "assumptions",
    "web_research": "enrichment",
    "writer": "writer",
}


def _canonical_stage_name(step_id: str) -> str:
    """Return the artifact-stage name associated with one progress step."""
    return _PROGRESS_STAGE_ALIASES.get(step_id, step_id)


class ProgressTracker:
    """Tracks pipeline step progress and flushes to disk.

    All public methods silently swallow exceptions so the tracker never
    breaks the pipeline. A threading lock protects concurrent access from
    concurrent pipeline updates.
    """

    def __init__(self, run_dir: Path) -> None:
        self._run_dir = run_dir
        self._lock = threading.Lock()
        self._steps: list[dict[str, Any]] = []
        self._step_index: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_step(self, step_id: str, label: str) -> None:
        """Mark *step_id* as ``running`` and flush to disk."""
        try:
            stage_name = _canonical_stage_name(step_id)
            with self._lock:
                entry: dict[str, Any] = {
                    "id": step_id,
                    "stage_name": stage_name,
                    "stage_number": resolve_stage_number(stage_name),
                    "label": label,
                    "status": "running",
                    "started_at": self._now(),
                    "completed_at": None,
                    "items": [],
                }
                self._step_index[step_id] = len(self._steps)
                self._steps.append(entry)
            self._flush()
        except Exception:
            logger.debug("ProgressTracker.start_step failed", exc_info=True)

    def add_item(
        self,
        step_id: str,
        text: str,
        *,
        item_type: str | None = None,
        title: str | None = None,
        domain: str | None = None,
        url: str | None = None,
        count: int | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        """Append a sub-item to an existing step and flush.

        Only non-None optional fields are written to keep progress.json compact.
        """
        try:
            with self._lock:
                idx = self._step_index.get(step_id)
                if idx is None:
                    return
                entry: dict[str, Any] = {"text": text}
                if item_type is not None:
                    entry["item_type"] = item_type
                if title is not None:
                    entry["title"] = title
                if domain is not None:
                    entry["domain"] = domain
                if url is not None:
                    entry["url"] = url
                if count is not None:
                    entry["count"] = count
                if metadata is not None:
                    entry["metadata"] = metadata
                self._steps[idx]["items"].append(entry)
            self._flush()
        except Exception:
            logger.debug("ProgressTracker.add_item failed", exc_info=True)

    def complete_step(self, step_id: str, status: str = "completed") -> None:
        """Mark *step_id* as finished (``completed`` or ``skipped``) and flush."""
        try:
            with self._lock:
                idx = self._step_index.get(step_id)
                if idx is None:
                    return
                self._steps[idx]["status"] = status
                self._steps[idx]["completed_at"] = self._now()
            self._flush()
        except Exception:
            logger.debug("ProgressTracker.complete_step failed", exc_info=True)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _now() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def _flush(self) -> None:
        """Atomic write via tempfile + replace."""
        try:
            with self._lock:
                payload = {"version": _VERSION, "steps": list(self._steps)}
            target = self._run_dir / "progress.json"
            fd, tmp_path = tempfile.mkstemp(
                dir=str(self._run_dir), suffix=".tmp", prefix=".progress_"
            )
            try:
                with open(fd, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh)
                Path(tmp_path).replace(target)
            except Exception:
                # Clean up temp file on failure
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except Exception:
                    pass
                raise
        except Exception:
            logger.debug("ProgressTracker._flush failed", exc_info=True)


__all__ = ["ProgressTracker"]
