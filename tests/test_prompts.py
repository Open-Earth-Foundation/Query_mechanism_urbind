from pathlib import Path

import pytest

from backend.modules.writer.models import WriterSectionPlan


def test_writer_aggregate_prompt_mentions_grouped_requirements() -> None:
    prompt_path = Path("backend/prompts/writer_system_aggregate.md")
    content = prompt_path.read_text(encoding="utf-8")
    assert "`analysis_mode` (`aggregate` | `city_by_city`)" in content
    assert "explicitly mention all numeric parts used in that calculation" in content
    assert "show the addition for the user" in content
    assert "Cities considered:" in content
    assert "Do not produce one section/bullet/paragraph per city" in content
    assert "final aggregation overview" in content
    assert "assumption-based estimate" in content
    assert "first question verbatim" in content


def test_writer_city_by_city_prompt_mentions_per_city_requirements() -> None:
    prompt_path = Path("backend/prompts/writer_system_city_by_city.md")
    content = prompt_path.read_text(encoding="utf-8")
    assert "`analysis_mode` (`aggregate` | `city_by_city`)" in content
    assert "explicitly mention all numeric parts used in that calculation" in content
    assert "show the addition for the user" in content
    assert "Cities considered:" in content
    assert "Provide one clear section per city first." in content
    assert "first question verbatim" in content


def test_writer_combine_prompt_uses_required_schema_sections() -> None:
    prompt_path = Path("backend/prompts/writer_system_combine.md")
    content = prompt_path.read_text(encoding="utf-8")
    for section in ("<role>", "<task>", "<input>", "<tools>", "<output>", "<example_output>"):
        assert section in content
    assert "submit_writer_output" in content
    assert "`draft_answers`" in content
    assert "citation_coverage" not in content


@pytest.mark.parametrize(
    "prompt_name,expected_terms",
    [
        (
            "writer_section_planner_system.md",
            ("WriterSectionPlan", "`section_id`", "`writing_instructions`", "`evidence_catalog`"),
        ),
        (
            "writer_section_system.md",
            ("submit_writer_output", "`section`", "`context_bundle`"),
        ),
        (
            "writer_section_composer_system.md",
            ("submit_writer_output", "`section_drafts`", "`reconsideration`"),
        ),
    ],
)
def test_writer_section_prompts_use_required_schema_sections(
    prompt_name: str,
    expected_terms: tuple[str, ...],
) -> None:
    prompt_path = Path("backend/prompts") / prompt_name
    content = prompt_path.read_text(encoding="utf-8")
    for section in ("<role>", "<task>", "<input>", "<output>", "<example_output>"):
        assert section in content
    if prompt_name != "writer_section_planner_system.md":
        assert "<tools>" in content
    for term in expected_terms:
        assert term in content
    for forbidden in ("table" + "_columns", "max" + "_words"):
        assert forbidden not in content


def test_writer_section_plan_prompt_example_matches_runtime_schema() -> None:
    prompt_path = Path("backend/prompts/writer_section_planner_system.md")
    content = prompt_path.read_text(encoding="utf-8")
    example = content.split("<example_output>", 1)[1].split("</example_output>", 1)[0]

    parsed = WriterSectionPlan.model_validate_json(example)

    assert parsed.strategy == "section_first"
    assert parsed.analysis_mode == "aggregate"
    assert parsed.sections[0].required_ref_ids == ["ref_1", "ref_3"]


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
