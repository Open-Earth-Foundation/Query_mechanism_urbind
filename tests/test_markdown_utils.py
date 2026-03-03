from backend.utils.markdown import render_question_section


def test_render_question_section_wraps_multiline_text_in_blockquote() -> None:
    question = (
        "Across the CCCs, what is the total number of vehicles explicitly targeted for electrification,\n"
        "\n"
        "disaggregated by:\n"
        "\n"
        "Municipal light-duty fleets, Public buses"
    )

    rendered = render_question_section(question)

    assert rendered.startswith("# Question\n")
    assert "> Across the CCCs, what is the total number of vehicles explicitly targeted for electrification," in rendered
    assert ">\n> disaggregated by:\n>\n> Municipal light-duty fleets, Public buses" in rendered
    assert rendered.endswith("\n\n")


def test_render_question_section_trims_outer_whitespace() -> None:
    rendered = render_question_section("\n  First line  \nSecond line\n\n")

    assert rendered == "# Question\n> First line\n> Second line\n\n"
