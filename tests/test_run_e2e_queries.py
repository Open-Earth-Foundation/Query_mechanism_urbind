from pathlib import Path

from backend.scripts.run_e2e_queries import load_questions


def test_load_questions_uses_overrides_only_when_provided(tmp_path: Path) -> None:
    questions_file = tmp_path / "questions.txt"
    questions_file.write_text("file question 1\nfile question 2\n", encoding="utf-8")

    result = load_questions(questions_file, ["cli question"])

    assert result == ["cli question"]


def test_load_questions_falls_back_to_file_when_no_overrides(tmp_path: Path) -> None:
    questions_file = tmp_path / "questions.txt"
    questions_file.write_text(
        "# comment\n\nfile question 1\nfile question 2\n", encoding="utf-8"
    )

    result = load_questions(questions_file, None)

    assert result == ["file question 1", "file question 2"]
