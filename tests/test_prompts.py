from pathlib import Path


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


def test_writer_city_by_city_prompt_mentions_per_city_requirements() -> None:
    prompt_path = Path("backend/prompts/writer_system_city_by_city.md")
    content = prompt_path.read_text(encoding="utf-8")
    assert "`analysis_mode` (`aggregate` | `city_by_city`)" in content
    assert "explicitly mention all numeric parts used in that calculation" in content
    assert "show the addition for the user" in content
    assert "Cities considered:" in content
    assert "Provide one clear section per city first." in content
