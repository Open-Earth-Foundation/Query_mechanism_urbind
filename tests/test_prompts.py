from pathlib import Path


def test_writer_aggregate_prompt_mentions_grouped_requirements() -> None:
    prompt_path = Path("backend/prompts/writer_system_aggregate.md")
    content = prompt_path.read_text(encoding="utf-8")
    assert "`analysis_mode` (`aggregate` | `city_by_city`)" in content
    assert "sum_numbers" in content
    assert "subtract_numbers" in content
    assert "multiply_numbers" in content
    assert "divide_numbers" in content
    assert "Cities considered:" in content
    assert "Do not produce one section/bullet/paragraph per city" in content
    assert "final aggregation overview" in content
    assert "assumption-based estimate" in content


def test_writer_city_by_city_prompt_mentions_per_city_requirements() -> None:
    prompt_path = Path("backend/prompts/writer_system_city_by_city.md")
    content = prompt_path.read_text(encoding="utf-8")
    assert "`analysis_mode` (`aggregate` | `city_by_city`)" in content
    assert "sum_numbers" in content
    assert "subtract_numbers" in content
    assert "multiply_numbers" in content
    assert "divide_numbers" in content
    assert "Cities considered:" in content
    assert "Provide one clear section per city first." in content


def test_chat_followup_router_prompt_uses_required_schema_sections() -> None:
    prompt_path = Path("backend/prompts/chat_followup_router_system.md")
    content = prompt_path.read_text(encoding="utf-8")
    assert "<role>" in content
    assert "<task>" in content
    assert "<input>" in content
    assert "<output>" in content
    assert "<example_output>" in content
    assert "submit_chat_followup_decision" in content
