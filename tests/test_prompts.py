from pathlib import Path


EXPECTED_PROMPT_TOOLS = {
    "benchmark_fact_judge_system.md": ["submit_fact_judgement"],
    "benchmark_judge_system.md": ["submit_benchmark_judgement"],
    "chat_followup_router_system.md": ["submit_chat_followup_decision"],
    "context_chat_empty_evidence_system.md": [
        "sum_numbers",
        "subtract_numbers",
        "multiply_numbers",
        "divide_numbers",
    ],
    "context_chat_evidence_map_system.md": [
        "sum_numbers",
        "subtract_numbers",
        "multiply_numbers",
        "divide_numbers",
    ],
    "context_chat_evidence_reduce_system.md": [
        "sum_numbers",
        "subtract_numbers",
        "multiply_numbers",
        "divide_numbers",
    ],
    "context_chat_system.md": [
        "sum_numbers",
        "subtract_numbers",
        "multiply_numbers",
        "divide_numbers",
    ],
    "external_source_finalizer_system.md": [],
    "external_source_researcher_system.md": [
        "get_tag_options",
        "list_candidate_sources",
        "regex_search",
        "expand_hits",
        "add_evidence_candidates",
        "list_evidence_candidates",
        "mark_no_evidence_found",
    ],
    "initiative_extractor_system.md": [
        "submit_initiative_extractions",
        "stop_initiative_extraction",
    ],
    "initiative_semantic_dedupe_system.md": ["submit_semantic_dedupe"],
    "markdown_researcher_system.md": ["submit_markdown_excerpts"],
    "orchestrator_research_question_system.md": ["submit_research_question"],
    "orchestrator_system.md": ["decide_next_action"],
    "tef_mapper_sector_router_system.md": ["submit_tef_sector_route"],
    "tef_mapper_subsector_router_system.md": ["submit_tef_subsector_route"],
    "tef_mapper_transition_mapper_system.md": ["submit_tef_transition_mapping"],
    "tef_numeric_unit_classifier_system.md": ["submit_numeric_unit_classification"],
    "writer_system_aggregate.md": ["submit_writer_output"],
    "writer_system_city_by_city.md": ["submit_writer_output"],
    "writer_system_combine.md": ["submit_writer_output"],
}


def test_all_prompts_use_required_schema_sections() -> None:
    prompt_paths = sorted(Path("backend/prompts").glob("*.md"))
    assert prompt_paths
    assert {path.name for path in prompt_paths} == set(EXPECTED_PROMPT_TOOLS)

    for prompt_path in prompt_paths:
        content = prompt_path.read_text(encoding="utf-8")
        for section in ("<role>", "<task>", "<input>", "<output>"):
            assert section in content, f"{prompt_path} missing {section}"


def test_prompt_tool_sections_match_runtime_contracts() -> None:
    for prompt_name, tool_names in EXPECTED_PROMPT_TOOLS.items():
        content = Path("backend/prompts", prompt_name).read_text(encoding="utf-8")

        if not tool_names:
            assert "<tools>" not in content
            continue

        assert "<tools>" in content
        tools_section = content.split("<tools>", maxsplit=1)[1].split("</tools>", maxsplit=1)[0]
        for tool_name in tool_names:
            assert tool_name in tools_section


def test_writer_aggregate_prompt_mentions_grouped_requirements() -> None:
    prompt_path = Path("backend/prompts/writer_system_aggregate.md")
    content = prompt_path.read_text(encoding="utf-8")
    for section in ("<role>", "<task>", "<input>", "<tools>", "<output>"):
        assert section in content
    assert "submit_writer_output" in content
    assert "`analysis_mode` (`aggregate` | `city_by_city`)" in content
    assert "explicitly mention all numeric parts used in that calculation" in content
    assert "show the addition for the user" in content
    assert "Cities considered:" in content
    assert "Do not produce one section/bullet/paragraph per city" in content
    assert "final aggregation overview" in content
    assert "assumption-based estimate" in content


def test_writer_city_by_city_prompt_mentions_per_city_requirements() -> None:
    prompt_path = Path("backend/prompts/writer_system_city_by_city.md")
    content = prompt_path.read_text(encoding="utf-8")
    for section in ("<role>", "<task>", "<input>", "<tools>", "<output>"):
        assert section in content
    assert "submit_writer_output" in content
    assert "`analysis_mode` (`aggregate` | `city_by_city`)" in content
    assert "explicitly mention all numeric parts used in that calculation" in content
    assert "show the addition for the user" in content
    assert "Cities considered:" in content
    assert "Provide one clear section per city first." in content


def test_writer_combine_prompt_uses_required_schema_sections() -> None:
    prompt_path = Path("backend/prompts/writer_system_combine.md")
    content = prompt_path.read_text(encoding="utf-8")
    for section in ("<role>", "<task>", "<input>", "<tools>", "<output>", "<example_output>"):
        assert section in content
    assert "submit_writer_output" in content
    assert "`draft_answers`" in content
    assert "citation_coverage" not in content


def test_external_source_researcher_prompt_uses_tool_contract_sections() -> None:
    prompt_path = Path("backend/prompts/external_source_researcher_system.md")
    content = prompt_path.read_text(encoding="utf-8")
    for section in ("<role>", "<task>", "<input>", "<tools>", "<output>", "<example_output>"):
        assert section in content
    for tool_name in (
        "get_tag_options",
        "list_candidate_sources",
        "regex_search",
        "expand_hits",
        "add_evidence_candidates",
        "list_evidence_candidates",
        "mark_no_evidence_found",
    ):
        assert tool_name in content
    assert "ExternalSourceAgentResult" in content


def test_chat_followup_router_prompt_uses_required_schema_sections() -> None:
    prompt_path = Path("backend/prompts/chat_followup_router_system.md")
    content = prompt_path.read_text(encoding="utf-8")
    assert "<role>" in content
    assert "<task>" in content
    assert "<input>" in content
    assert "<output>" in content
    assert "<example_output>" in content
    assert "submit_chat_followup_decision" in content
    assert "selected_run_ids" not in content
    assert "selected_followup_bundle_ids" not in content
    assert "- `source_id`" not in content
