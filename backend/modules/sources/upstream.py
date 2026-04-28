"""Upstream source resolution: fetch the bytes referenced by a SourceConfig.

The default strategy is a shallow git clone into a temp directory, pinned
to ``source.pinned_commit`` if set.  Handlers receive a resolved root
``Path`` and treat it as opaque — they don't care whether it's a temp
clone, a local checkout, or a vendored directory.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from backend.modules.sources.manifest import SourceConfig

logger = logging.getLogger(__name__)


class UpstreamResolutionError(RuntimeError):
    """Raised when an upstream source cannot be resolved to a directory."""


def _github_clone_url(repo: str) -> str:
    """Build a clone URL, allowing GH_TOKEN for private repos."""
    token = os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")
    if token:
        return f"https://x-access-token:{token}@github.com/{repo}.git"
    return f"https://github.com/{repo}.git"


def _shallow_clone(repo: str, commit: str | None, dest: Path) -> None:
    """Shallow-clone a repo into ``dest`` and (optionally) check out a commit."""
    url = _github_clone_url(repo)
    if commit:
        # Fetch only the pinned commit to keep the clone small.
        subprocess.run(["git", "init", "--quiet"], cwd=dest, check=True)
        subprocess.run(
            ["git", "remote", "add", "origin", url],
            cwd=dest,
            check=True,
        )
        subprocess.run(
            ["git", "fetch", "--depth=1", "--quiet", "origin", commit],
            cwd=dest,
            check=True,
        )
        subprocess.run(
            ["git", "checkout", "--quiet", "FETCH_HEAD"],
            cwd=dest,
            check=True,
        )
    else:
        subprocess.run(
            ["git", "clone", "--depth=1", "--quiet", url, str(dest)],
            check=True,
        )


@contextmanager
def resolve_upstream(source: SourceConfig) -> Iterator[Path]:
    """Yield a directory containing the upstream source's files.

    For ``provider=github``: clone to a temp dir; cleanup on context exit.
    For ``provider=local``: yield the project root (handlers resolve paths
    relative to the project root).
    """
    if source.provider == "local":
        yield Path.cwd()
        return

    if source.provider == "github":
        if not source.repo:
            raise UpstreamResolutionError(
                f"source {source.id!r}: provider=github requires a repo"
            )
        tmp_root = Path(tempfile.mkdtemp(prefix=f"urbind-src-{source.id}-"))
        try:
            logger.info(
                "Cloning %s%s into %s",
                source.repo,
                f" @ {source.pinned_commit[:8]}" if source.pinned_commit else "",
                tmp_root,
            )
            _shallow_clone(source.repo, source.pinned_commit, tmp_root)
            yield tmp_root
        except subprocess.CalledProcessError as exc:
            raise UpstreamResolutionError(
                f"git clone failed for {source.repo}: {exc}"
            ) from exc
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)
        return

    raise UpstreamResolutionError(f"Unknown provider: {source.provider!r}")


def resolved_commit(upstream_root: Path, source: SourceConfig) -> str | None:
    """Return the actual commit checked out (best-effort)."""
    if source.provider != "github":
        return None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=upstream_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or None
    except subprocess.CalledProcessError:
        return source.pinned_commit


__all__ = [
    "UpstreamResolutionError",
    "resolve_upstream",
    "resolved_commit",
]
