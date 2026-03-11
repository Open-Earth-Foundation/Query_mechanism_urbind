"""Low-level token and message helpers for context chat orchestration."""

from __future__ import annotations

from pathlib import Path

from backend.utils.tokenization import count_tokens, get_encoding

CHAT_EVIDENCE_CACHE_FILENAME = "evidence_chunks.json"


def build_messages(
    system_prompt: str,
    history: list[dict[str, str]],
    user_content: str,
) -> list[dict[str, str]]:
    """Build the chat-completion message list."""
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_content})
    return messages


def estimate_messages_tokens(messages: list[dict[str, str]]) -> int:
    """Estimate token usage for chat-completion messages."""
    total = 0
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        total += count_tokens(f"{role}\n{content}") + 6
    return total + 2


def truncate_to_tokens(value: str, max_tokens: int) -> str:
    """Truncate text to a token budget while preserving leading content."""
    if max_tokens <= 0:
        return ""
    encoding = get_encoding()
    tokens = encoding.encode(value)
    if len(tokens) <= max_tokens:
        return value

    suffix = "\n\n[truncated due to prompt token budget]"
    suffix_tokens = encoding.encode(suffix)
    if max_tokens <= len(suffix_tokens):
        return encoding.decode(tokens[:max_tokens])
    head_tokens = tokens[: max_tokens - len(suffix_tokens)]
    return encoding.decode(head_tokens) + suffix


def chat_evidence_cache_path(runs_dir: Path, run_id: str) -> Path:
    """Return the cached compact evidence artifact path for one run."""
    return runs_dir / run_id / "chat_cache" / CHAT_EVIDENCE_CACHE_FILENAME


__all__ = [
    "build_messages",
    "chat_evidence_cache_path",
    "estimate_messages_tokens",
    "truncate_to_tokens",
]
