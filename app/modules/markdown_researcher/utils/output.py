"""Output parsing and coercion utilities for markdown researcher."""

import ast
import logging

from pydantic import ValidationError

from app.modules.markdown_researcher.models import (
    MarkdownExcerpt,
    MarkdownResearchResult,
)


logger = logging.getLogger(__name__)


# DEAD CODE: Regex patterns for repr-string parsing
# With current agent setup (output_type=MarkdownResearchResult + strict_mode=True),
# these patterns are not used. Kept in case we switch to frameworks without strict output typing.
# _PY_STRING_LITERAL = r"(?:'[^'\\]*(?:\\.[^'\\]*)*'|\"[^\"\\]*(?:\\.[^\"\\]*)*\")"
# _EXCERPT_REPR_PATTERN = re.compile(
#     rf"MarkdownExcerpt\(\s*snippet=(?P<snippet>{_PY_STRING_LITERAL})\s*,\s*"
#     rf"city_name=(?P<city_name>{_PY_STRING_LITERAL})\s*,\s*"
#     rf"(?:partial_answer|answer)=(?P<partial_answer>{_PY_STRING_LITERAL})\s*,\s*"
#     rf"relevant=(?P<relevant>{_PY_STRING_LITERAL})\s*\)",
#     re.DOTALL,
# )
# _STATUS_REPR_PATTERN = re.compile(
#     r"status\s*=\s*(?P<status>'success'|'error'|\"success\"|\"error\")"
# )


# DEAD CODE: Repr-string parser function
# With current agent setup (output_type=MarkdownResearchResult + strict_mode=True),
# the agent framework guarantees structured JSON output, so this function should not be called.
# Kept in case we switch to models/frameworks without strict output typing.
# To re-enable: uncomment this function and the regex patterns above (lines 18-28).
#
# def parse_markdown_result_repr(text: str) -> MarkdownResearchResult | None:
#     """Parse a MarkdownResearchResult repr-like string into a model."""
#     status = "success"
#     status_match = _STATUS_REPR_PATTERN.search(text)
#     if status_match:
#         try:
#             status = ast.literal_eval(status_match.group("status"))
#         except Exception as exc:
#             logger.debug("Failed to parse status from repr: %s", exc)
#             status = "success"
#     if status not in ("success", "error"):
#         status = "success"
#
#     error = None
#
#     excerpts: list[MarkdownExcerpt] = []
#     for match in _EXCERPT_REPR_PATTERN.finditer(text):
#         try:
#             snippet = ast.literal_eval(match.group("snippet"))
#             city_name = ast.literal_eval(match.group("city_name"))
#             partial_answer = ast.literal_eval(match.group("partial_answer"))
#             relevant = ast.literal_eval(match.group("relevant"))
#             excerpts.append(
#                 MarkdownExcerpt(
#                     snippet=str(snippet),
#                     city_name=str(city_name),
#                     partial_answer=str(partial_answer),
#                     relevant=str(relevant),
#                 )
#             )
#         except Exception as exc:
#             logger.debug("Failed to parse excerpt from repr: %s", exc)
#             continue
#
#     if not excerpts and not status_match and error is None:
#         return None
#
#     return MarkdownResearchResult(status=status, excerpts=excerpts, error=error)


def coerce_markdown_result(output: object) -> MarkdownResearchResult | None:
    """Coerce raw tool output into a MarkdownResearchResult when possible.
    
    With the current agent setup (output_type=MarkdownResearchResult + strict_mode=True),
    the agent framework should always return a properly typed MarkdownResearchResult.
    This function provides fallback strategies for unexpected output formats,
    ordered by likelihood:
    1. Already a MarkdownResearchResult instance
    2. JSON string
    3. Python literal (dict/list)
    4. Object with model_dump() (Pydantic model)
    5. Object with __dict__ attribute
    """
    if isinstance(output, MarkdownResearchResult):
        return output

    if isinstance(output, str):
        try:
            return MarkdownResearchResult.model_validate_json(output)
        except ValidationError:
            pass
        try:
            parsed = ast.literal_eval(output)
        except (ValueError, SyntaxError):
            parsed = None
        if parsed is not None:
            return coerce_markdown_result(parsed)
        # If we can't parse as JSON or literal, return None (repr parsing is commented out)
        return None

    if isinstance(output, dict):
        if "arguments" in output:
            return coerce_markdown_result(output["arguments"])
        if "result" in output:
            return coerce_markdown_result(output["result"])
        try:
            return MarkdownResearchResult.model_validate(output)
        except ValidationError as exc:
            logger.debug("Dict coercion failed: %s", exc)
            return None

    model_dump = getattr(output, "model_dump", None)
    if callable(model_dump):
        return coerce_markdown_result(model_dump())

    value_dict = getattr(output, "__dict__", None)
    if isinstance(value_dict, dict):
        filtered = {
            key: item
            for key, item in value_dict.items()
            if not str(key).startswith("_")
        }
        return coerce_markdown_result(filtered)

    logger.debug(
        "No matching coercion strategy for output type: %s", type(output).__name__
    )
    return None


__all__ = [
    "coerce_markdown_result",
    # "parse_markdown_result_repr",  # DEAD CODE - commented out with function definition
]
