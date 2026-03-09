"""Service layer for FastAPI run lifecycle endpoints."""

from backend.api.services.chat_memory import (
    ChatMemoryStore,
    ChatSessionExistsError,
    ChatSessionNotFoundError,
)
from backend.api.services.chat_followup_research import (
    CHAT_FOLLOWUP_CITY_UNAVAILABLE,
    CHAT_FOLLOWUP_SEARCH_FAILED,
    ChatFollowupSearchResult,
    followup_bundle_dir,
    run_chat_followup_search,
)
from backend.api.services.reference_artifacts import (
    build_reference_item,
    load_reference_records,
)
from backend.api.services.source_chunks import load_source_chunks, normalize_chunk_ids
from backend.api.services.assumptions_review import (
    apply_assumptions_and_regenerate,
    apply_assumptions_to_context,
    dedupe_missing_data_items,
    discover_missing_data,
    discover_missing_data_for_run,
    group_missing_data_by_city,
    load_latest_assumptions_payload,
    rewrite_document_with_assumptions,
)
from backend.api.services.city_catalog import (
    build_city_subset,
    index_city_markdown_files,
    list_city_names,
    load_city_groups,
)
from backend.api.services.context_chat import (
    generate_context_chat_reply,
    load_context_bundle,
    load_final_document,
    resolve_chat_token_cap,
)
from backend.api.services.run_executor import RunExecutor, StartRunCommand
from backend.api.services.run_store import (
    DuplicateRunIdError,
    IN_PROGRESS_STATUSES,
    SUCCESS_STATUSES,
    TERMINAL_STATUSES,
    RunRecord,
    RunStore,
)

__all__ = [
    "DuplicateRunIdError",
    "IN_PROGRESS_STATUSES",
    "SUCCESS_STATUSES",
    "TERMINAL_STATUSES",
    "ChatMemoryStore",
    "ChatSessionExistsError",
    "ChatSessionNotFoundError",
    "CHAT_FOLLOWUP_CITY_UNAVAILABLE",
    "CHAT_FOLLOWUP_SEARCH_FAILED",
    "ChatFollowupSearchResult",
    "build_city_subset",
    "build_reference_item",
    "followup_bundle_dir",
    "index_city_markdown_files",
    "load_reference_records",
    "load_source_chunks",
    "list_city_names",
    "load_city_groups",
    "normalize_chunk_ids",
    "resolve_chat_token_cap",
    "generate_context_chat_reply",
    "load_context_bundle",
    "load_final_document",
    "run_chat_followup_search",
    "RunExecutor",
    "RunRecord",
    "RunStore",
    "StartRunCommand",
    "apply_assumptions_and_regenerate",
    "apply_assumptions_to_context",
    "dedupe_missing_data_items",
    "discover_missing_data",
    "discover_missing_data_for_run",
    "group_missing_data_by_city",
    "load_latest_assumptions_payload",
    "rewrite_document_with_assumptions",
]
