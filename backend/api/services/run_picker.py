"""Helpers for compact run-picker labels and search ranking."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher

from backend.api.services.run_store import RunRecord
from backend.utils.city_normalization import format_city_display_name, normalize_city_key
from backend.utils.json_io import read_json_object

_MULTISPACE_PATTERN = re.compile(r"\s+")
_TOKEN_SPLIT_PATTERN = re.compile(r"[\W_]+", flags=re.UNICODE)
_MIN_FUZZY_CITY_LENGTH = 4
_FUZZY_CITY_RATIO_THRESHOLD = 0.84


@dataclass(frozen=True)
class RunPickerEntry:
    """Compact run metadata returned to picker-facing API routes."""

    run_id: str
    question: str
    status: str
    picker_timestamp: str


@dataclass(frozen=True)
class _RunSearchDocument:
    """Normalized run metadata used for picker filtering and ranking."""

    record: RunRecord
    run_id_text: str
    question: str
    picker_timestamp: str
    picker_timestamp_text: str
    city_keys: tuple[str, ...]
    city_token_sets: tuple[frozenset[str], ...]
    question_text: str
    combined_tokens: frozenset[str]


@dataclass(frozen=True)
class _RunSearchScore:
    """Sortable search score for one run document."""

    tier: int
    matched_tokens: int
    fuzzy_city_score: float


def build_picker_timestamp(started_at: datetime) -> str:
    """Format a compact UTC picker timestamp like ``0312-1954``."""
    aware_started_at = started_at
    if aware_started_at.tzinfo is None:
        aware_started_at = aware_started_at.replace(tzinfo=timezone.utc)
    return aware_started_at.astimezone(timezone.utc).strftime("%m%d-%H%M")


def list_run_picker_entries(
    records: list[RunRecord],
    *,
    search: str | None = None,
) -> list[RunPickerEntry]:
    """Return run picker entries, optionally filtered and ranked by search text."""
    documents = [_build_search_document(record) for record in records]
    normalized_search = _normalize_text(search or "")
    if normalized_search:
        documents = _filter_and_rank_documents(documents, normalized_search)
    return [
        RunPickerEntry(
            run_id=document.record.run_id,
            question=document.question,
            status=document.record.status,
            picker_timestamp=document.picker_timestamp,
        )
        for document in documents
    ]


def _build_search_document(record: RunRecord) -> _RunSearchDocument:
    """Normalize one run record into a search-ready document."""
    run_id_text = _normalize_text(record.run_id)
    question = record.question.strip()
    picker_timestamp = build_picker_timestamp(record.started_at)
    picker_timestamp_text = _normalize_text(picker_timestamp)
    city_labels = _load_city_labels(record)
    city_keys = tuple(
        key
        for key in (normalize_city_key(label) for label in city_labels)
        if key
    )
    city_token_sets = tuple(frozenset(key.split("_")) for key in city_keys)
    question_text = _normalize_text(question)
    combined_tokens = frozenset(
        _tokenize(" ".join([record.run_id, picker_timestamp, question, *city_labels]).strip())
    )
    return _RunSearchDocument(
        record=record,
        run_id_text=run_id_text,
        question=question,
        picker_timestamp=picker_timestamp,
        picker_timestamp_text=picker_timestamp_text,
        city_keys=city_keys,
        city_token_sets=city_token_sets,
        question_text=question_text,
        combined_tokens=combined_tokens,
    )


def _load_city_labels(record: RunRecord) -> tuple[str, ...]:
    """Load display-friendly city labels from ``api_state.json`` when available."""
    if record.api_state_path is None:
        return ()
    api_state = read_json_object(record.api_state_path)
    if not isinstance(api_state, dict):
        return ()
    inputs = api_state.get("inputs")
    if not isinstance(inputs, dict):
        return ()

    labels: list[str] = []
    seen: set[str] = set()
    _append_city_labels(
        labels,
        seen,
        inputs.get("selected_cities_planned"),
        transform_display=True,
    )
    _append_city_labels(
        labels,
        seen,
        inputs.get("selected_cities_found"),
        transform_display=True,
    )
    return tuple(labels)


def _append_city_labels(
    labels: list[str],
    seen: set[str],
    raw_values: object,
    *,
    transform_display: bool,
) -> None:
    """Append de-duplicated city labels from one persisted input field."""
    if not isinstance(raw_values, list):
        return
    for value in raw_values:
        if not isinstance(value, str):
            continue
        cleaned = value.strip()
        if not cleaned:
            continue
        label = format_city_display_name(cleaned) if transform_display else cleaned
        key = normalize_city_key(label)
        if not key or key in seen:
            continue
        seen.add(key)
        labels.append(label)


def _filter_and_rank_documents(
    documents: list[_RunSearchDocument],
    normalized_search: str,
) -> list[_RunSearchDocument]:
    """Filter run picker documents and rank stronger matches first."""
    search_tokens = _tokenize(normalized_search)
    search_key = normalize_city_key(normalized_search)
    scored_documents: list[tuple[_RunSearchScore, _RunSearchDocument]] = []
    for document in documents:
        score = _score_document(document, normalized_search, search_tokens, search_key)
        if score is None:
            continue
        scored_documents.append((score, document))

    scored_documents.sort(
        key=lambda item: (
            item[0].tier,
            -item[0].matched_tokens,
            -item[0].fuzzy_city_score,
            -item[1].record.started_at.timestamp(),
            item[1].record.run_id,
        )
    )
    return [document for _, document in scored_documents]


def _score_document(
    document: _RunSearchDocument,
    normalized_search: str,
    search_tokens: list[str],
    search_key: str,
) -> _RunSearchScore | None:
    """Return a sortable score when a run matches the picker search."""
    has_numeric_fragment = any(character.isdigit() for character in normalized_search)
    matched_tokens = sum(1 for token in search_tokens if token in document.combined_tokens)
    if (
        normalized_search in document.run_id_text
        or normalized_search in document.picker_timestamp_text
    ):
        return _RunSearchScore(
            tier=0,
            matched_tokens=max(matched_tokens, 1),
            fuzzy_city_score=0.0,
        )

    if _has_exact_city_match(document, search_key, search_tokens):
        return _RunSearchScore(
            tier=1,
            matched_tokens=max(matched_tokens, 1),
            fuzzy_city_score=1.0,
        )

    fuzzy_city_score = _best_fuzzy_city_score(document, search_key, search_tokens)
    if fuzzy_city_score is not None:
        return _RunSearchScore(
            tier=2,
            matched_tokens=max(matched_tokens, 1),
            fuzzy_city_score=fuzzy_city_score,
        )

    if normalized_search in document.question_text:
        return _RunSearchScore(
            tier=3,
            matched_tokens=max(matched_tokens, 1),
            fuzzy_city_score=0.0,
        )

    if search_tokens and all(token in document.combined_tokens for token in search_tokens):
        return _RunSearchScore(
            tier=4,
            matched_tokens=len(search_tokens),
            fuzzy_city_score=0.0,
        )

    if has_numeric_fragment:
        return None

    if matched_tokens > 0:
        return _RunSearchScore(
            tier=5,
            matched_tokens=matched_tokens,
            fuzzy_city_score=0.0,
        )
    return None


def _has_exact_city_match(
    document: _RunSearchDocument,
    search_key: str,
    search_tokens: list[str],
) -> bool:
    """Return True when the search directly resolves to a selected city."""
    if search_key and search_key in document.city_keys:
        return True
    if not search_tokens or any(len(token) < 3 for token in search_tokens):
        return False
    for city_tokens in document.city_token_sets:
        if all(token in city_tokens for token in search_tokens):
            return True
    return False


def _best_fuzzy_city_score(
    document: _RunSearchDocument,
    search_key: str,
    search_tokens: list[str],
) -> float | None:
    """Return a fuzzy city similarity score for typo-tolerant city search."""
    candidates = [search_key, *search_tokens]
    best_score: float | None = None
    for candidate in candidates:
        if len(candidate) < _MIN_FUZZY_CITY_LENGTH:
            continue
        for city_key in document.city_keys:
            ratio = SequenceMatcher(None, candidate, city_key).ratio()
            if best_score is None or ratio > best_score:
                best_score = ratio
            for city_token in city_key.split("_"):
                token_ratio = SequenceMatcher(None, candidate, city_token).ratio()
                if best_score is None or token_ratio > best_score:
                    best_score = token_ratio
    if best_score is None or best_score < _FUZZY_CITY_RATIO_THRESHOLD:
        return None
    return best_score


def _normalize_text(value: str) -> str:
    """Normalize whitespace and case for phrase matching."""
    return _MULTISPACE_PATTERN.sub(" ", value.casefold()).strip()


def _tokenize(value: str) -> list[str]:
    """Tokenize text for order-insensitive search matching."""
    normalized = _TOKEN_SPLIT_PATTERN.sub(" ", value.casefold())
    return [token for token in normalized.split() if token]


__all__ = ["RunPickerEntry", "build_picker_timestamp", "list_run_picker_entries"]
