"""Output parsing and coercion utilities for markdown researcher."""

import ast
import logging
import re

from pydantic import ValidationError

from app.modules.markdown_researcher.models import (
    MarkdownExcerpt,
    MarkdownResearchResult,
)


logger = logging.getLogger(__name__)


_PY_STRING_LITERAL = r"(?:'[^'\\]*(?:\\.[^'\\]*)*'|\"[^\"\\]*(?:\\.[^\"\\]*)*\")"
_EXCERPT_REPR_PATTERN = re.compile(
    rf"MarkdownExcerpt\(\s*snippet=(?P<snippet>{_PY_STRING_LITERAL})\s*,\s*"
    rf"city_name=(?P<city_name>{_PY_STRING_LITERAL})\s*,\s*"
    rf"answer=(?P<answer>{_PY_STRING_LITERAL})\s*,\s*"
    rf"relevant=(?P<relevant>{_PY_STRING_LITERAL})\s*\)",
    re.DOTALL,
)
_STATUS_REPR_PATTERN = re.compile(
    r"status\s*=\s*(?P<status>'success'|'error'|\"success\"|\"error\")"
)


def parse_markdown_result_repr(text: str) -> MarkdownResearchResult | None:
    """Parse a MarkdownResearchResult repr-like string into a model."""
    status = "success"
    status_match = _STATUS_REPR_PATTERN.search(text)
    if status_match:
        try:
            status = ast.literal_eval(status_match.group("status"))
        except Exception as exc:
            logger.debug("Failed to parse status from repr: %s", exc)
            status = "success"
    if status not in ("success", "error"):
        status = "success"

    error = None

    excerpts: list[MarkdownExcerpt] = []
    for match in _EXCERPT_REPR_PATTERN.finditer(text):
        try:
            snippet = ast.literal_eval(match.group("snippet"))
            city_name = ast.literal_eval(match.group("city_name"))
            answer = ast.literal_eval(match.group("answer"))
            relevant = ast.literal_eval(match.group("relevant"))
            excerpts.append(
                MarkdownExcerpt(
                    snippet=str(snippet),
                    city_name=str(city_name),
                    answer=str(answer),
                    relevant=str(relevant),
                )
            )
        except Exception as exc:
            logger.debug("Failed to parse excerpt from repr: %s", exc)
            continue

    if not excerpts and not status_match and error is None:
        return None

    return MarkdownResearchResult(status=status, excerpts=excerpts, error=error)


def coerce_markdown_result(output: object) -> MarkdownResearchResult | None:
    """Coerce raw tool output into a MarkdownResearchResult when possible."""
    if output.__class__.__name__ == "MarkdownResearchResult":
        return output
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
        return parse_markdown_result_repr(output)

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

    parsed_repr = parse_markdown_result_repr(str(output))
    if parsed_repr is not None:
        return parsed_repr

    logger.debug(
        "No matching coercion strategy for output type: %s", type(output).__name__
    )
    return None


__all__ = [
    "coerce_markdown_result",
    "parse_markdown_result_repr",
]
