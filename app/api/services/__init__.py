"""Service layer for FastAPI run lifecycle endpoints."""

from app.api.services.chat_memory import (
    ChatMemoryStore,
    ChatSessionExistsError,
    ChatSessionNotFoundError,
)
from app.api.services.assumptions_review import (
    apply_assumptions_and_regenerate,
    apply_assumptions_to_context,
    dedupe_missing_data_items,
    discover_missing_data,
    discover_missing_data_for_run,
    group_missing_data_by_city,
    load_latest_assumptions_payload,
    rewrite_document_with_assumptions,
)
from app.api.services.city_catalog import (
    build_city_subset,
    index_city_markdown_files,
    list_city_names,
    load_city_groups,
)
from app.api.services.context_chat import (
    CHAT_PROMPT_TOKEN_CAP,
    generate_context_chat_reply,
    load_context_bundle,
    load_final_document,
)
from app.api.services.run_executor import RunExecutor, StartRunCommand
from app.api.services.run_store import (
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
    "build_city_subset",
    "index_city_markdown_files",
    "list_city_names",
    "load_city_groups",
    "CHAT_PROMPT_TOKEN_CAP",
    "generate_context_chat_reply",
    "load_context_bundle",
    "load_final_document",
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
