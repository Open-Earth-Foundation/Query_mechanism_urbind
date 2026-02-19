from __future__ import annotations

import json
import logging
import re
import time
from datetime import date, datetime
from decimal import Decimal
from typing import Iterable
from uuid import UUID

import tiktoken

from backend.modules.sql_researcher.models import (
    SqlQuery,
    SqlQueryResult,
    SqlResearchResult,
)
from backend.services.db_client import DbClient, is_select_query

logger = logging.getLogger(__name__)


SQL_KEYWORDS = {
    "select",
    "from",
    "join",
    "left",
    "right",
    "inner",
    "outer",
    "full",
    "on",
    "where",
    "and",
    "or",
    "as",
    "distinct",
    "group",
    "by",
    "order",
    "having",
    "limit",
    "offset",
    "desc",
    "asc",
    "nulls",
    "last",
    "first",
    "case",
    "when",
    "then",
    "else",
    "end",
    "in",
    "like",
    "ilike",
    "between",
    "is",
    "not",
    "null",
    "count",
    "min",
    "max",
    "avg",
    "sum",
    "coalesce",
    "string_agg",
}
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
TABLE_REF_RE = re.compile(
    r'(?i)\b(from|join)\s+("?[A-Za-z0-9_]+"?)(?:\s+(?:as\s+)?("?[A-Za-z0-9_]+"?))?'
)
QUALIFIED_COL_RE = re.compile(r'("?[A-Za-z0-9_]+"?)\s*\.\s*("?[A-Za-z0-9_]+"?)')
SELECT_CLAUSE_RE = re.compile(r"(?is)\bselect\b(.*?)\bfrom\b")
WHERE_CLAUSE_RE = re.compile(
    r"(?is)\bwhere\b(.*?)(\border\b|\bgroup\b|\bhaving\b|\blimit\b|$)"
)
WHERE_COL_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*)\b\s*(=|<>|!=|<=|>=|<|>|\blike\b|\bilike\b|\bin\b|\bbetween\b)",
    re.IGNORECASE,
)


def _ensure_query_ids(queries: list[SqlQuery]) -> list[SqlQuery]:
    updated: list[SqlQuery] = []
    for idx, query in enumerate(queries, start=1):
        query_id = query.query_id.strip() if query.query_id else ""
        if not query_id:
            query_id = f"q{idx}"
        updated.append(query.model_copy(update={"query_id": query_id}))
    return updated


def _count_tokens(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def _json_safe_value(value: object) -> str | int | float | None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (UUID, datetime, date, Decimal)):
        return str(value)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=True, default=str)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _normalize_rows(rows: list[list[object]]) -> list[list[str | int | float | None]]:
    return [[_json_safe_value(value) for value in row] for row in rows]


def _quote_identifiers(sql: str, identifiers: list[str]) -> str:
    if not identifiers:
        return sql
    updated = sql
    for name in sorted(set(identifiers), key=len, reverse=True):
        if not name:
            continue
        pattern = rf'(?<!")\b{re.escape(name)}\b(?!")'
        updated = re.sub(pattern, f'"{name}"', updated)
    return updated


def _escape_percent(sql: str) -> str:
    return re.sub(r"%(?!%)", "%%", sql)


def _strip_quotes(name: str) -> str:
    return name.strip('"')


def build_table_catalog(schema_summary: dict) -> list[str]:
    """Build a flat catalog of tables and columns with type information."""
    catalog: list[str] = []
    for table in schema_summary.get("tables", []):
        name = table.get("name")
        columns_with_types = table.get("columns_with_types", [])

        if name and columns_with_types:
            # Format: "TableName: col1 (Type), col2 (Type), ..."
            col_strings = [
                f"{col['name']} ({col['type']})" for col in columns_with_types
            ]
            catalog.append(f"{name}: {', '.join(col_strings)}")
        elif name and table.get("columns", []):
            # Fallback for backward compatibility if no type info
            columns = table.get("columns", [])
            catalog.append(f"{name}: {', '.join(columns)}")
    return catalog


def _extract_table_aliases(sql: str) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for _, table, alias in TABLE_REF_RE.findall(sql):
        table_name = _strip_quotes(table)
        alias_name = _strip_quotes(alias) if alias else table_name
        aliases[alias_name] = table_name
    return aliases


def _extract_unqualified_columns(sql: str, valid_columns: set[str]) -> set[str]:
    sanitized = re.sub(r"'(?:''|[^'])*'", "", sql)
    select_match = SELECT_CLAUSE_RE.search(sanitized)
    invalid: set[str] = set()
    if select_match:
        select_clause = select_match.group(1)
        parts: list[str] = []
        current = []
        depth = 0
        for char in select_clause:
            if char == "(":
                depth += 1
            elif char == ")":
                depth = max(depth - 1, 0)
            if char == "," and depth == 0:
                parts.append("".join(current))
                current = []
            else:
                current.append(char)
        if current:
            parts.append("".join(current))
        for part in parts:
            cleaned = re.sub(r"(?i)\bas\b\s+\w+", "", part)
            if "." in cleaned:
                continue
            tokens = TOKEN_RE.findall(cleaned)
            if not tokens:
                continue
            candidate = tokens[-1].lower()
            if candidate in SQL_KEYWORDS or candidate in valid_columns:
                continue
            invalid.add(candidate)

    where_match = WHERE_CLAUSE_RE.search(sanitized)
    if where_match:
        where_clause = where_match.group(1)
        for token, _ in WHERE_COL_RE.findall(where_clause):
            candidate = token.lower()
            if candidate in SQL_KEYWORDS or candidate in valid_columns:
                continue
            invalid.add(candidate)

    return invalid


def validate_queries(
    queries: list[SqlQuery], schema_summary: dict
) -> list[dict[str, object]]:
    table_map: dict[str, set[str]] = {}
    for table in schema_summary.get("tables", []):
        name = table.get("name")
        columns = table.get("columns", [])
        if name:
            table_map[name.lower()] = {col.lower() for col in columns}

    errors: list[dict[str, object]] = []
    for query in queries:
        alias_map = _extract_table_aliases(query.query)
        invalid_tables: set[str] = set()
        invalid_columns: set[str] = set()

        for alias, table in alias_map.items():
            if table.lower() not in table_map:
                invalid_tables.add(table)

        for alias, column in QUALIFIED_COL_RE.findall(query.query):
            alias_key = _strip_quotes(alias)
            column_name = _strip_quotes(column)
            table_name = alias_map.get(alias_key)
            if not table_name and alias_key.lower() in table_map:
                table_name = alias_key
            if not table_name or table_name.lower() not in table_map:
                invalid_tables.add(table_name or alias_key)
                continue
            if column_name.lower() not in table_map[table_name.lower()]:
                invalid_columns.add(f"{alias_key}.{column_name}")

        if len(alias_map) == 1:
            table_name = next(iter(alias_map.values()))
            valid_columns = table_map.get(table_name.lower(), set())
            invalid_columns.update(
                _extract_unqualified_columns(query.query, valid_columns)
            )

        if invalid_tables or invalid_columns:
            errors.append(
                {
                    "query_id": query.query_id,
                    "invalid_tables": sorted(invalid_tables),
                    "invalid_columns": sorted(invalid_columns),
                }
            )

    return errors


def _split_sql_list(section: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for char in section:
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(depth - 1, 0)
        if char == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    if current:
        parts.append("".join(current))
    return parts


def _select_item_is_valid(item: str, valid_columns: set[str]) -> bool:
    if "*" in item:
        return True
    cleaned = re.sub(r"(?i)\bas\b\s+\w+", "", item)
    tokens = [token.lower() for token in TOKEN_RE.findall(cleaned)]
    tokens = [token for token in tokens if token not in SQL_KEYWORDS]
    if not tokens:
        return True
    if any(token not in valid_columns for token in tokens):
        return False
    return True


def _sanitize_select_clause(sql: str, valid_columns: set[str]) -> str:
    match = SELECT_CLAUSE_RE.search(sql)
    if not match:
        return sql
    select_clause = match.group(1)
    items = _split_sql_list(select_clause)
    kept = [item for item in items if _select_item_is_valid(item, valid_columns)]
    if not kept:
        kept = ["*"]
    new_select = "SELECT " + ", ".join(item.strip() for item in kept) + " "
    return sql[: match.start()] + new_select + sql[match.end() - 4 :]


def _sanitize_where_clause(sql: str, valid_columns: set[str]) -> str:
    match = WHERE_CLAUSE_RE.search(sql)
    if not match:
        return sql
    where_clause = match.group(1)
    tail_start = match.start(2) if match.group(2) else match.end()
    parts = re.split(r"(\bAND\b|\bOR\b)", where_clause, flags=re.IGNORECASE)
    rebuilt: list[str] = []
    pending_connector: str | None = None
    for part in parts:
        token = part.strip()
        if not token:
            continue
        if token.upper() in {"AND", "OR"}:
            pending_connector = token.upper()
            continue
        candidates = [col.lower() for col, _ in WHERE_COL_RE.findall(token)]
        if candidates and any(col not in valid_columns for col in candidates):
            continue
        if rebuilt and pending_connector:
            rebuilt.append(pending_connector)
        rebuilt.append(token)
        pending_connector = None
    if not rebuilt:
        return sql[: match.start()] + sql[tail_start:]
    new_where = "WHERE " + " ".join(rebuilt) + " "
    return sql[: match.start()] + new_where + sql[tail_start:]


def _sanitize_order_by_clause(sql: str, valid_columns: set[str]) -> str:
    match = re.search(r"(?is)\border\s+by\b(.*?)(\blimit\b|$)", sql)
    if not match:
        return sql
    order_clause = match.group(1)
    tail = (
        sql[match.end() - len(match.group(2)) :]
        if match.group(2)
        else sql[match.end() :]
    )
    items = _split_sql_list(order_clause)
    kept: list[str] = []
    for item in items:
        cleaned = re.sub(r"(?i)\s+(asc|desc)\b", "", item).strip()
        if not cleaned:
            continue
        if "." in cleaned:
            _, col = cleaned.split(".", 1)
            col_name = _strip_quotes(col.strip()).lower()
        else:
            tokens = [token.lower() for token in TOKEN_RE.findall(cleaned)]
            tokens = [token for token in tokens if token not in SQL_KEYWORDS]
            col_name = tokens[-1] if tokens else ""
        if col_name and col_name in valid_columns:
            kept.append(item.strip())
    if not kept:
        return sql[: match.start()] + tail
    new_order = "ORDER BY " + ", ".join(kept) + " "
    return sql[: match.start()] + new_order + tail


def sanitize_queries(queries: list[SqlQuery], schema_summary: dict) -> list[SqlQuery]:
    table_map: dict[str, set[str]] = {}
    for table in schema_summary.get("tables", []):
        name = table.get("name")
        columns = table.get("columns", [])
        if name:
            table_map[name.lower()] = {col.lower() for col in columns}

    sanitized: list[SqlQuery] = []
    for query in queries:
        alias_map = _extract_table_aliases(query.query)
        if len(alias_map) != 1:
            sanitized.append(query)
            continue
        table_name = next(iter(alias_map.values()))
        valid_columns = table_map.get(table_name.lower())
        if not valid_columns:
            sanitized.append(query)
            continue
        updated = _sanitize_select_clause(query.query, valid_columns)
        updated = _sanitize_where_clause(updated, valid_columns)
        updated = _sanitize_order_by_clause(updated, valid_columns)
        sanitized.append(query.model_copy(update={"query": updated}))
    return sanitized


def execute_queries(
    client: DbClient,
    queries: list[SqlQuery],
    max_rows: int,
    identifiers: list[str] | None = None,
) -> list[SqlQueryResult]:
    results: list[SqlQueryResult] = []
    identifier_list = identifiers or []
    for query in _ensure_query_ids(queries):
        if not is_select_query(query.query):
            logger.warning("Skipping non-SELECT query: %s", query.query)
            continue
        sql = _escape_percent(_quote_identifiers(query.query, identifier_list))
        start = time.perf_counter()
        try:
            columns, rows = client.query(sql)
            rows = _normalize_rows(rows)
            if max_rows and len(rows) > max_rows:
                rows = rows[:max_rows]
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            results.append(
                SqlQueryResult(
                    query_id=query.query_id,
                    columns=columns,
                    rows=rows,
                    row_count=len(rows),
                    elapsed_ms=elapsed_ms,
                    token_count=_count_tokens(
                        json.dumps(rows, ensure_ascii=True, default=str)
                    ),
                    truncated=False,
                )
            )
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            logger.exception("SQL query failed: %s", query.query_id)
            error_row: list[str | int | float | None] = [f"ERROR: {exc}"]
            results.append(
                SqlQueryResult(
                    query_id=query.query_id,
                    columns=["error"],
                    rows=[error_row],
                    row_count=1,
                    elapsed_ms=elapsed_ms,
                    token_count=_count_tokens(
                        json.dumps([error_row], ensure_ascii=True, default=str)
                    ),
                    truncated=False,
                )
            )
    return results


def cap_results(
    results: list[SqlQueryResult],
    max_tokens: int,
) -> tuple[list[SqlQueryResult], int, bool]:
    capped: list[SqlQueryResult] = []
    total_tokens = 0
    truncated = False

    for result in results:
        rows: list[list[str | int | float | None]] = []
        row_tokens = 0
        for row in result.rows:
            token_count = _count_tokens(json.dumps(row, ensure_ascii=True, default=str))
            if total_tokens + token_count > max_tokens:
                truncated = True
                break
            rows.append(row)
            row_tokens += token_count
            total_tokens += token_count

        capped.append(
            result.model_copy(
                update={
                    "rows": rows,
                    "row_count": len(rows),
                    "token_count": row_tokens,
                    "truncated": truncated,
                }
            )
        )
        if truncated:
            break

    return capped, total_tokens, truncated


def build_sql_research_result(
    queries: list[SqlQuery],
    results: list[SqlQueryResult],
    total_tokens: int,
    truncated: bool,
) -> SqlResearchResult:
    """Build a structured SQL research result bundle."""
    return SqlResearchResult(
        queries=queries,
        results=results,
        total_token_count=total_tokens,
        truncation_applied=truncated,
    )


__all__ = [
    "execute_queries",
    "cap_results",
    "build_sql_research_result",
    "build_table_catalog",
    "validate_queries",
    "sanitize_queries",
]
