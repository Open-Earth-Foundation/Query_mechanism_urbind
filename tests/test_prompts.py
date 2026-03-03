from pathlib import Path


def test_writer_aggregate_prompt_mentions_grouped_requirements() -> None:
    prompt_path = Path("backend/prompts/writer_system_aggregate.md")
    content = prompt_path.read_text(encoding="utf-8")
    assert "`analysis_mode` (`aggregate` | `city_by_city`)" in content
    assert "run_calculation_subagent" in content
    assert "Do not call arithmetic tools directly in writer output generation." in content
    assert "Cities considered:" in content
    assert "Do not produce one section/bullet/paragraph per city" in content
    assert "final aggregation overview" in content
    assert "Assumptions used for calculation" in content


def test_writer_city_by_city_prompt_mentions_per_city_requirements() -> None:
    prompt_path = Path("backend/prompts/writer_system_city_by_city.md")
    content = prompt_path.read_text(encoding="utf-8")
    assert "`analysis_mode` (`aggregate` | `city_by_city`)" in content
    assert "run_calculation_subagent" in content
    assert "Do not call arithmetic tools directly in writer output generation." in content
    assert "Cities considered:" in content
    assert "Provide one clear section per city first." in content


def test_calculation_researcher_prompt_mentions_schema_and_submit_tool() -> None:
    prompt_path = Path("backend/prompts/calculation_researcher_system.md")
    content = prompt_path.read_text(encoding="utf-8")
    assert "request_schema" in content
    assert "submit_calculation_output" in content
    assert "included_cities[]" in content
    assert "excluded_policy_cities[]" in content
