from pathlib import Path

from app.modules.markdown_researcher.services import load_markdown_documents
from app.utils.config import MarkdownResearcherConfig


def test_load_markdown_documents_filters_selected_cities(tmp_path: Path) -> None:
    (tmp_path / "Munich.md").write_text("# Munich\n\nText", encoding="utf-8")
    (tmp_path / "Leipzig.md").write_text("# Leipzig\n\nText", encoding="utf-8")
    config = MarkdownResearcherConfig(model="test")

    docs = load_markdown_documents(
        tmp_path,
        config,
        selected_cities=["Munich"],
    )

    assert docs
    assert all(doc["city_name"] == "Munich" for doc in docs)


def test_load_markdown_documents_city_filter_is_case_insensitive(
    tmp_path: Path,
) -> None:
    (tmp_path / "Munich.md").write_text("# Munich\n\nText", encoding="utf-8")
    config = MarkdownResearcherConfig(model="test")

    docs = load_markdown_documents(
        tmp_path,
        config,
        selected_cities=["munich"],
    )

    assert docs
    assert all(doc["city_name"] == "Munich" for doc in docs)
