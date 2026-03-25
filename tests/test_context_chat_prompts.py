from pathlib import Path

import backend.api.services.prompts.context_chat as context_chat_prompts


def test_context_chat_system_prompt_template_uses_required_schema_sections() -> None:
    content = Path("backend/prompts/context_chat_system.md").read_text(encoding="utf-8")

    assert "<role>" in content
    assert "<task>" in content
    assert "<input>" in content
    assert "<output>" in content
    assert "<example_output>" in content
    assert "Original build question:" in content


def test_context_chat_overflow_prompt_templates_use_required_schema_sections() -> None:
    prompt_paths = [
        Path("backend/prompts/context_chat_evidence_map_system.md"),
        Path("backend/prompts/context_chat_evidence_reduce_system.md"),
        Path("backend/prompts/context_chat_empty_evidence_system.md"),
    ]

    for prompt_path in prompt_paths:
        content = prompt_path.read_text(encoding="utf-8")
        assert "<role>" in content
        assert "<task>" in content
        assert "<input>" in content
        assert "<output>" in content
        assert "<example_output>" in content


def test_context_chat_system_prompt_header_renders_markdown_template() -> None:
    prompt = context_chat_prompts.build_system_prompt_header(
        original_question="What does Aachen do for rooftop solar?",
        retry_missing_citation=True,
    )

    assert "<role>" in prompt
    assert "<task>" in prompt
    assert "<input>" in prompt
    assert "<output>" in prompt
    assert "<example_output>" in prompt
    assert "Prior response failed citation requirements" in prompt
    assert "What does Aachen do for rooftop solar?" in prompt


def test_context_chat_evidence_prompts_render_markdown_templates() -> None:
    prompt = context_chat_prompts.compose_evidence_map_prompt(
        prompt_header="HEADER",
        evidence_block="- [ref_1] City: Aachen",
        chunk_index=2,
        total_chunks=5,
    )

    assert "HEADER" in prompt
    assert "<role>" in prompt
    assert "<task>" in prompt
    assert "evidence chunk `2` of `5`" in prompt.lower()
    assert "- [ref_1] City: Aachen" in prompt


def test_context_chat_reduce_prompt_renders_markdown_template() -> None:
    prompt = context_chat_prompts.compose_evidence_reduce_prompt(
        prompt_header="HEADER",
        analyses_block="### Partial analysis 1\nGrounded answer [ref_1]",
        stage_index=2,
        batch_index=1,
        batch_count=3,
    )

    assert "HEADER" in prompt
    assert "<role>" in prompt
    assert "<task>" in prompt
    assert "reduce stage `2`, batch `1` of `3`" in prompt
    assert "Grounded answer [ref_1]" in prompt


def test_context_chat_empty_prompt_renders_markdown_template() -> None:
    prompt = context_chat_prompts.compose_empty_evidence_prompt("HEADER")

    assert "HEADER" in prompt
    assert "<role>" in prompt
    assert "<task>" in prompt
    assert "<input>" in prompt
    assert "<output>" in prompt
    assert "<example_output>" in prompt
    assert "does not provide extractable evidence" in prompt
