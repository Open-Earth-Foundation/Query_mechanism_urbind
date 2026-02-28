from backend.api.services import context_chat
from backend.utils.tokenization import count_tokens


def _catalog_entry(ref_id: str, token_repeats: int) -> dict[str, str]:
    return {
        "ref_id": ref_id,
        "city_name": "Munich",
        "quote": "evidence " * token_repeats,
        "partial_answer": "grounded partial answer",
    }


def test_fit_citation_catalog_to_budget_prunes_refs() -> None:
    citation_catalog = [
        _catalog_entry("ref_1", 8),
        _catalog_entry("ref_2", 180),
        _catalog_entry("ref_3", 8),
    ]
    prompt_header = context_chat._build_system_prompt_header(
        original_question="What is the policy status?",
        retry_missing_citation=False,
    )
    user_content = "Summarize the policy."
    fixed_tokens = context_chat._estimate_messages_tokens(
        [{"role": "user", "content": user_content}]
    )
    first_entry_budget = (
        count_tokens(context_chat._render_citation_catalog_block(citation_catalog[:1])) + 20
    )
    token_cap = (
        fixed_tokens
        + count_tokens(prompt_header)
        + context_chat.CHAT_PROMPT_TOKEN_BUFFER
        + first_entry_budget
    )

    fitted = context_chat._fit_citation_catalog_to_budget(
        citation_catalog=citation_catalog,
        prompt_header=prompt_header,
        history=[],
        user_content=user_content,
        token_cap=token_cap,
    )

    assert [item["ref_id"] for item in fitted] == ["ref_1"]


def test_render_citation_catalog_block_for_empty_entries() -> None:
    rendered = context_chat._render_citation_catalog_block([])
    assert "No citation entries fit within the prompt token budget" in rendered


def test_system_prompt_header_avoids_inline_allowed_ref_list() -> None:
    header = context_chat._build_system_prompt_header(
        original_question="What does Aachen do for rooftop solar?",
        retry_missing_citation=False,
    )
    assert "Allowed references for this turn:" not in header
    assert "present in that catalog" in header
