"""Writer markdown helper utilities."""

from __future__ import annotations

import logging

from backend.modules.orchestrator.utils.references import is_valid_ref_id
from backend.utils.city_normalization import (
    format_city_display_name,
    format_city_stem,
    normalize_city_key,
)

logger = logging.getLogger(__name__)
CITIES_CONSIDERED_HEADER = "## Cities considered"
NO_EVIDENCE_HEADER = "## Cities with no important evidence found"


def is_generated_footer_header(line: str) -> bool:
    """Return True when a line starts a generated footer block."""
    normalized = line.strip().lower()
    if not normalized:
        return False
    normalized = normalized[:-1] if normalized.endswith(":") else normalized
    return normalized in {
        "cities considered",
        "## cities considered",
        "## cities with no important evidence found",
    }


def extract_markdown_bundle(context_bundle: dict[str, object]) -> dict[str, object]:
    """Return markdown bundle from context or an empty dict."""
    markdown_bundle = context_bundle.get("markdown")
    if isinstance(markdown_bundle, dict):
        return markdown_bundle
    return {}


def extract_markdown_excerpts(markdown_bundle: dict[str, object]) -> list[dict[str, object]]:
    """Return markdown excerpts as a normalized list of mapping objects."""
    excerpts = markdown_bundle.get("excerpts")
    if not isinstance(excerpts, list):
        return []
    return [excerpt for excerpt in excerpts if isinstance(excerpt, dict)]


def extract_excerpt_count(markdown_bundle: dict[str, object]) -> int:
    """Resolve excerpt count with list-length fallback."""
    value = markdown_bundle.get("excerpt_count")
    if isinstance(value, int):
        return max(value, 0)
    return len(extract_markdown_excerpts(markdown_bundle))


def extract_expected_ref_ids(context_bundle: dict[str, object]) -> set[str]:
    """Collect valid reference ids from markdown excerpts."""
    markdown_bundle = extract_markdown_bundle(context_bundle)
    excerpts = extract_markdown_excerpts(markdown_bundle)

    expected_ids: set[str] = set()
    for excerpt in excerpts:
        ref_id = excerpt.get("ref_id")
        if not isinstance(ref_id, str):
            continue
        candidate = ref_id.strip()
        if is_valid_ref_id(candidate):
            expected_ids.add(candidate)
    return expected_ids


def city_display_name(value: str) -> str:
    """Normalize city to a readable display form."""
    display = format_city_display_name(value)
    if display:
        return display
    return format_city_stem(value)


def city_key(value: str) -> str:
    """Build normalized city key from display-friendly value."""
    return normalize_city_key(city_display_name(value))


def dedupe_city_names(values: list[str]) -> list[str]:
    """De-duplicate city names while preserving order."""
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        display = city_display_name(value)
        key = city_key(display)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(display)
    return deduped


def extract_selected_city_names(
    context_bundle: dict[str, object],
    markdown_bundle: dict[str, object],
) -> list[str]:
    """Resolve selected city names from bundle metadata or excerpts."""
    for key in ("selected_city_names", "inspected_city_names"):
        raw = markdown_bundle.get(key)
        if isinstance(raw, list):
            values = [str(item) for item in raw if isinstance(item, str)]
            resolved = dedupe_city_names(values)
            if resolved:
                return resolved

    for key in ("selected_cities", "inspected_cities"):
        raw = markdown_bundle.get(key)
        if isinstance(raw, list):
            values = [str(item) for item in raw if isinstance(item, str)]
            resolved = dedupe_city_names(values)
            if resolved:
                return resolved

    excerpts = extract_markdown_excerpts(markdown_bundle)
    excerpt_city_names = [
        str(excerpt.get("city_name", ""))
        for excerpt in excerpts
        if isinstance(excerpt.get("city_name"), str)
    ]
    resolved_from_excerpts = dedupe_city_names(excerpt_city_names)
    if resolved_from_excerpts:
        return resolved_from_excerpts

    fallback_context = context_bundle.get("selected_cities")
    if isinstance(fallback_context, list):
        fallback_values = [str(item) for item in fallback_context if isinstance(item, str)]
        resolved = dedupe_city_names(fallback_values)
        if resolved:
            return resolved
    return []


def resolve_analysis_mode(context_bundle: dict[str, object]) -> str:
    """Resolve analysis mode from context bundle with aggregate fallback."""
    raw_mode = context_bundle.get("analysis_mode")
    if isinstance(raw_mode, str):
        normalized = raw_mode.strip()
        if normalized in {"aggregate", "city_by_city"}:
            return normalized
    markdown_bundle = extract_markdown_bundle(context_bundle)
    markdown_mode = markdown_bundle.get("analysis_mode")
    if isinstance(markdown_mode, str):
        normalized = markdown_mode.strip()
        if normalized in {"aggregate", "city_by_city"}:
            return normalized
    return "aggregate"


def extract_ref_city_mapping(
    markdown_bundle: dict[str, object],
) -> tuple[dict[str, str], dict[str, str]]:
    """Return mapping ref_id -> city_key and city_key -> display name."""
    ref_to_city_key: dict[str, str] = {}
    city_display_by_key: dict[str, str] = {}
    excerpts = extract_markdown_excerpts(markdown_bundle)
    for excerpt in excerpts:
        ref_id = str(excerpt.get("ref_id", "")).strip()
        city_name = str(excerpt.get("city_name", "")).strip()
        raw_city_key = str(excerpt.get("city_key", "")).strip()
        if not ref_id or not is_valid_ref_id(ref_id):
            continue
        city_key_value = normalize_city_key(raw_city_key)
        city_display = city_display_name(city_name)
        if not city_key_value:
            city_key_value = city_key(city_display)
        if not city_key_value:
            continue
        if not city_display:
            city_display = format_city_display_name(city_key_value) or format_city_stem(
                city_key_value
            )
        ref_to_city_key[ref_id] = city_key_value
        city_display_by_key.setdefault(city_key_value, city_display)
    return ref_to_city_key, city_display_by_key


def extract_reference_tokens(content: str) -> set[str]:
    """Extract unique values from bracket tokens like [ref_1] without regex parsing."""
    tokens: set[str] = set()
    cursor = 0
    while cursor < len(content):
        start = content.find("[", cursor)
        if start < 0:
            break
        end = content.find("]", start + 1)
        if end < 0:
            break
        token = content[start + 1 : end].strip()
        if token:
            tokens.add(token)
        cursor = end + 1
    return tokens


def extract_cited_ref_ids(content: str) -> set[str]:
    """Extract unique valid [ref_n] tokens from generated markdown."""
    return {token for token in extract_reference_tokens(content) if is_valid_ref_id(token)}


def extract_city_coverage_sets(
    *,
    content: str,
    markdown_bundle: dict[str, object],
    selected_city_names: list[str],
) -> tuple[list[str], list[str], list[str], dict[str, str]]:
    """Return required, missing, and no-evidence city keys plus display-name mapping."""
    ref_to_city_key, city_display_by_key = extract_ref_city_mapping(markdown_bundle)
    excerpt_city_keys = set(city_display_by_key.keys())
    selected_city_keys: set[str] = set()
    for city_name in selected_city_names:
        resolved_key = city_key(city_name)
        if resolved_key:
            selected_city_keys.add(resolved_key)

    required_city_keys = sorted(selected_city_keys & excerpt_city_keys)
    if not required_city_keys:
        required_city_keys = sorted(excerpt_city_keys)

    cited_ref_ids = extract_cited_ref_ids(content)
    covered_city_keys = {
        ref_to_city_key[ref_id]
        for ref_id in cited_ref_ids
        if ref_id in ref_to_city_key
    }
    missing_coverage_keys = [key for key in required_city_keys if key not in covered_city_keys]
    no_evidence_keys = sorted(selected_city_keys - excerpt_city_keys)
    return required_city_keys, missing_coverage_keys, no_evidence_keys, city_display_by_key


def extract_missing_coverage(
    *,
    content: str,
    markdown_bundle: dict[str, object],
    selected_city_names: list[str],
) -> tuple[list[str], list[str], dict[str, str]]:
    """Return missing citation-coverage city keys and no-evidence city keys."""
    _required_city_keys, missing_coverage_keys, no_evidence_keys, city_display_by_key = (
        extract_city_coverage_sets(
            content=content,
            markdown_bundle=markdown_bundle,
            selected_city_names=selected_city_names,
        )
    )
    return missing_coverage_keys, no_evidence_keys, city_display_by_key



def render_no_evidence_section(no_evidence_city_names: list[str]) -> str:
    """Render deterministic section for selected cities with no excerpts."""
    if not no_evidence_city_names:
        return ""
    lines = [NO_EVIDENCE_HEADER]
    for city in no_evidence_city_names:
        lines.append(f"- {city}: no important evidence was found in the provided excerpts.")
    return "\n".join(lines)


def render_cities_considered_section(city_names: list[str]) -> str:
    """Render deterministic city list section for final answer footer."""
    lines = [CITIES_CONSIDERED_HEADER]
    if not city_names:
        lines.append("- (none)")
    else:
        for city in city_names:
            lines.append(f"- {city}")
    return "\n".join(lines)


def strip_existing_footer_sections(content: str) -> str:
    """Remove existing generated footer blocks before appending canonical sections."""
    cleaned_lines: list[str] = []
    skip_footer_block = False
    for line in content.splitlines():
        stripped = line.strip()
        if is_generated_footer_header(stripped):
            skip_footer_block = True
            continue

        if skip_footer_block:
            if not stripped or stripped.startswith("- ") or stripped.startswith("* "):
                continue
            if is_generated_footer_header(stripped):
                continue
            skip_footer_block = False

        if not skip_footer_block:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def append_sections(content: str, sections: list[str]) -> str:
    """Append non-empty sections to content with stable spacing."""
    content = strip_existing_footer_sections(content)
    rendered_sections: list[str] = []
    for section in sections:
        section_stripped = section.strip()
        if not section_stripped:
            continue
        rendered_sections.append(section)
    if not rendered_sections:
        return content.strip()
    base = content.strip()
    if not base:
        return "\n\n".join(rendered_sections).strip()
    return f"{base}\n\n" + "\n\n".join(rendered_sections)


def validate_writer_citations(content: str, context_bundle: dict[str, object]) -> None:
    """Emit warnings when writer citations are missing, malformed, or unknown."""
    markdown_bundle = extract_markdown_bundle(context_bundle)
    excerpt_count = extract_excerpt_count(markdown_bundle)
    if excerpt_count <= 0:
        return

    expected_ref_ids = extract_expected_ref_ids(context_bundle)
    cited_ref_ids = extract_cited_ref_ids(content)
    raw_reference_tokens = extract_reference_tokens(content)
    malformed_tokens = sorted(
        token
        for token in raw_reference_tokens
        if token.startswith("ref_") and not is_valid_ref_id(token)
    )

    if not cited_ref_ids:
        logger.warning(
            "Writer output contains no [ref_n] citations despite excerpt_count=%d.",
            excerpt_count,
        )

    if not expected_ref_ids:
        logger.warning(
            "Writer context has excerpt_count=%d but no valid ref ids in excerpts.",
            excerpt_count,
        )

    unknown_refs = sorted(ref_id for ref_id in cited_ref_ids if ref_id not in expected_ref_ids)
    if unknown_refs:
        logger.warning(
            "Writer output cites unknown reference ids: %s",
            ", ".join(unknown_refs),
        )

    if malformed_tokens:
        logger.warning(
            "Writer output contains malformed reference tokens: %s",
            ", ".join(malformed_tokens),
        )


__all__ = [
    "append_sections",
    "extract_city_coverage_sets",
    "extract_markdown_bundle",
    "extract_missing_coverage",
    "extract_ref_city_mapping",
    "extract_selected_city_names",
    "render_cities_considered_section",
    "render_no_evidence_section",
    "resolve_analysis_mode",
    "validate_writer_citations",
]
