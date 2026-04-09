"""Markdown-to-PDF conversion service.

Uses ``fpdf2`` (pure Python, no system C libraries) to render
markdown content as a formatted PDF document.
"""

from __future__ import annotations

import re


def strip_citation_markers(text: str) -> str:
    """Remove ``[ref_n]`` citation markers from markdown text."""
    return re.sub(r"\[ref_\d+\]", "", text)


# ---------------------------------------------------------------------------
#  Unicode → latin-1 sanitization
# ---------------------------------------------------------------------------

_UNICODE_REPLACEMENTS: dict[str, str] = {
    "\u2018": "'",   "\u2019": "'",   # smart single quotes
    "\u201c": '"',   "\u201d": '"',   # smart double quotes
    "\u2013": "-",   "\u2014": "--",  # en/em dash
    "\u2026": "...",                   # ellipsis
    "\u2032": "'",   "\u2033": '"',   # prime / double prime
    "\u00a0": " ",                     # non-breaking space
    "\u200b": "",    "\u200c": "",    "\u200d": "",  # zero-width chars
    "\ufeff": "",                      # BOM
    "\u2022": "*",   "\u2023": ">",  "\u2043": "-",  # bullets
    "\u25aa": "*",   "\u25cf": "*",
    "\u2192": "->",  "\u2190": "<-", "\u2194": "<->",  # arrows
    "\u2264": "<=",  "\u2265": ">=", "\u2260": "!=",   # math
    "\u00b2": "2",   "\u00b3": "3",                     # superscripts
}

_UNICODE_RE = re.compile(
    "[" + re.escape("".join(_UNICODE_REPLACEMENTS.keys())) + "]"
)


def _sanitize_for_latin1(text: str) -> str:
    """Replace Unicode characters unsupported by latin-1 core fonts."""
    result = _UNICODE_RE.sub(lambda m: _UNICODE_REPLACEMENTS[m.group()], text)
    return result.encode("latin-1", errors="replace").decode("latin-1")


# ---------------------------------------------------------------------------
#  HTML simplification for fpdf2 compatibility
# ---------------------------------------------------------------------------

# Tags that fpdf2 cannot handle when nested inside <td>/<th>.
_INLINE_TAG_RE = re.compile(r"</?(?:strong|em|b|i|code|a|span|abbr|mark|sub|sup|del|ins)[^>]*>")


def _simplify_html_for_fpdf(html: str) -> str:
    """Strip inline formatting tags from table cells.

    fpdf2 raises ``NotImplementedError`` for nested tags inside ``<td>``
    and ``<th>`` elements.  We strip them to plain text so tables render.
    """

    def _clean_cell(match: re.Match[str]) -> str:
        open_tag = match.group(1)   # e.g. "<td>" or "<th ...>"
        content = match.group(2)
        close_tag = match.group(3)  # "</td>" or "</th>"
        cleaned = _INLINE_TAG_RE.sub("", content)
        return f"{open_tag}{cleaned}{close_tag}"

    return re.sub(
        r"(<t[dh][^>]*>)(.*?)(</t[dh]>)",
        _clean_cell,
        html,
        flags=re.DOTALL,
    )


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def markdown_to_pdf(markdown_content: str) -> bytes:
    """Convert markdown text to PDF bytes.

    Strips citation markers, sanitizes Unicode, simplifies HTML for fpdf2
    compatibility, then renders via fpdf2.  Pure Python — no system C libs.
    """
    import markdown as md
    from fpdf import FPDF

    cleaned = strip_citation_markers(markdown_content)
    sanitized = _sanitize_for_latin1(cleaned)
    html_body = md.markdown(
        sanitized,
        extensions=["tables", "fenced_code"],
    )
    html_body = _simplify_html_for_fpdf(html_body)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    pdf.write_html(html_body)
    return bytes(pdf.output())


__all__ = ["markdown_to_pdf", "strip_citation_markers"]
