"""Handler: convert PDFs from upstream into markdown files in our repo.

The handler walks ``ingestion.inputs.paths`` (resolved against the upstream
root), collects ``*.pdf`` files (skipping ``inputs.excludes``), converts
each via ``pymupdf4llm.to_markdown``, and writes the result under
``ingestion.output.path`` named according to ``ingestion.output.naming``.

Naming template variables:
- ``{city}``  — parent directory stem (lowercased), or the file's parent
  directory name when the parent directory is the inputs root.
- ``{slug}``  — file stem (lowercased, slug-safe).

The default naming is ``{city}_{slug}/{city}.md`` because the markdown
researcher uses ``Path.stem`` as the city identifier
(``backend/modules/markdown_researcher/services.py``).  Putting each
document in its own subdirectory named ``{city}_{slug}/`` keeps the
output filename stem equal to the city while still allowing multiple
docs per city to coexist.

State emits per-file records with input sha + output sha so reruns are
cheap to diff against.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from backend.modules.sources.handlers import IngestionContext, register
from backend.modules.sources.state import IngestionState

logger = logging.getLogger(__name__)


_HANDLER_NAME = "ingest.pdf_to_markdown"
_DEFAULT_NAMING = "{city}_{slug}/{city}.md"


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _slugify(value: str) -> str:
    """Lowercase, strip extension, keep [a-z0-9-]."""
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def _iter_pdfs(roots: list[Path], excludes: list[Path]) -> list[Path]:
    """Recursively collect ``*.pdf`` files under each root, excluding paths in excludes."""
    excluded = {p.resolve() for p in excludes}
    seen: set[Path] = set()
    out: list[Path] = []
    for root in roots:
        if not root.exists():
            logger.warning("upstream input path missing: %s", root)
            continue
        if root.is_file():
            candidates = [root]
        else:
            candidates = sorted(root.rglob("*.pdf"))
        for path in candidates:
            resolved = path.resolve()
            if resolved in excluded or resolved in seen:
                continue
            if path.suffix.lower() != ".pdf":
                continue
            seen.add(resolved)
            out.append(path)
    return out


def _derive_city(pdf_path: Path, input_roots: list[Path]) -> str:
    """Return the city stem for a PDF.

    Convention for tier-1 city plans: the PDF's *parent* directory name
    is the city.  We fall back to the PDF's grandparent if the parent
    *is* an inputs root (i.e. PDFs sit directly in an inputs root with
    no city subdir).
    """
    input_roots_resolved = {r.resolve() for r in input_roots}
    parent = pdf_path.parent.resolve()
    if parent in input_roots_resolved:
        return _slugify(pdf_path.stem)
    return _slugify(pdf_path.parent.name)


def _convert_pdf(pdf_bytes: bytes) -> str:
    """Convert PDF bytes to markdown via pymupdf4llm. Imported lazily."""
    import io

    import pymupdf
    import pymupdf4llm

    doc = pymupdf.Document(stream=io.BytesIO(pdf_bytes), filetype="pdf")
    try:
        return pymupdf4llm.to_markdown(doc)
    finally:
        doc.close()


def run_pdf_to_markdown(
    context: IngestionContext,
    *,
    convert_fn: Callable[[bytes], str] | None = None,
) -> IngestionState:
    """Convert upstream PDFs to markdown files under the configured output dir."""
    convert = convert_fn or _convert_pdf

    inputs = context.ingestion.inputs
    output = context.ingestion.output
    if not output.path:
        raise ValueError(
            f"ingestion {context.ingestion.id!r}: pdf_to_markdown requires output.path"
        )

    naming = output.naming or _DEFAULT_NAMING
    output_dir = (context.project_root / output.path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    upstream_root = context.upstream_root
    input_roots = [upstream_root / p for p in inputs.paths]
    exclude_paths = [upstream_root / p for p in inputs.excludes]

    pdf_files = _iter_pdfs(input_roots, exclude_paths)
    logger.info(
        "pdf_to_markdown: found %d PDFs under %s",
        len(pdf_files),
        ", ".join(str(p) for p in input_roots),
    )

    file_records: list[dict] = []
    for pdf_path in pdf_files:
        pdf_bytes = pdf_path.read_bytes()
        input_sha = _hash_bytes(pdf_bytes)
        city = _derive_city(pdf_path, input_roots)
        slug = _slugify(pdf_path.stem)
        out_name = naming.format(city=city, slug=slug)
        out_path = output_dir / out_name

        markdown = convert(pdf_bytes)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown, encoding="utf-8")
        output_sha = _hash_bytes(markdown.encode("utf-8"))

        rel_input = pdf_path.relative_to(upstream_root).as_posix()
        rel_output = out_path.relative_to(context.project_root).as_posix()
        file_records.append(
            {
                "input": rel_input,
                "input_sha": input_sha,
                "output": rel_output,
                "output_sha": output_sha,
                "city": city,
                "slug": slug,
            }
        )
        logger.info("converted %s -> %s", rel_input, rel_output)

    return IngestionState(
        ingestion_id=context.ingestion.id,
        source_id=context.source.id,
        last_ingested_at=datetime.now(timezone.utc).isoformat(),
        source_commit=context.resolved_commit,
        converter=context.ingestion.converter or "pymupdf4llm",
        file_count=len(file_records),
        files=file_records,
    )


@register(_HANDLER_NAME)
def _entrypoint(context: IngestionContext) -> IngestionState:
    return run_pdf_to_markdown(context)


__all__ = ["run_pdf_to_markdown"]
