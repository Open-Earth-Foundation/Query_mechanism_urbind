"""Output parsing and coercion utilities for markdown researcher."""

import ast
import logging

from pydantic import ValidationError

from backend.modules.markdown_researcher.models import (
    MarkdownResearchResult,
)


logger = logging.getLogger(__name__)


def coerce_markdown_result(output: object) -> MarkdownResearchResult | None:
    """Coerce raw tool output into a MarkdownResearchResult when possible."""
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
]
