from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Iterable, Protocol

import psycopg

logger = logging.getLogger(__name__)


def is_select_query(sql: str) -> bool:
    stripped = _strip_leading_sql_comments(sql).strip().lower()
    return stripped.startswith("select") or stripped.startswith("with")


def _strip_leading_sql_comments(sql: str) -> str:
    text = sql.lstrip()
    while text:
        if text.startswith("--"):
            newline = text.find("\n")
            if newline == -1:
                return ""
            text = text[newline + 1 :].lstrip()
            continue
        if text.startswith("/*"):
            end = text.find("*/")
            if end == -1:
                return ""
            text = text[end + 2 :].lstrip()
            continue
        break
    return text


class DbClient(Protocol):
    def query(self, sql: str, params: Iterable | None = None) -> tuple[list[str], list[list[object]]]:
        ...


class SQLiteClient:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def connect(self) -> sqlite3.Connection:
        if not self.db_path.exists():
            raise FileNotFoundError(f"Source DB not found: {self.db_path}")
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def query(self, sql: str, params: Iterable | None = None) -> tuple[list[str], list[list[object]]]:
        if not is_select_query(sql):
            raise ValueError("Only SELECT queries are allowed.")

        logger.info("Executing SQL query")
        with self.connect() as conn:
            cursor = conn.execute(sql, params or [])
            rows = cursor.fetchall()
            columns = [col[0] for col in cursor.description] if cursor.description else []
            data = [list(row) for row in rows]
        return columns, data


class PostgresClient:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url

    def connect(self) -> psycopg.Connection:
        return psycopg.connect(normalize_database_url(self.database_url))

    def query(self, sql: str, params: Iterable | None = None) -> tuple[list[str], list[list[object]]]:
        if not is_select_query(sql):
            raise ValueError("Only SELECT queries are allowed.")

        logger.info("Executing SQL query")
        with self.connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, params or [])
                rows = cursor.fetchall()
                columns = [col.name for col in cursor.description] if cursor.description else []
                data = [list(row) for row in rows]
        return columns, data


def get_db_client(source_db_path: Path, source_db_url: str | None) -> DbClient:
    if source_db_url:
        logger.info("Using DATABASE_URL for source DB")
        return PostgresClient(source_db_url)
    logger.warning("DATABASE_URL not set; falling back to SOURCE_DB_PATH")
    return SQLiteClient(source_db_path)


def normalize_database_url(database_url: str) -> str:
    if database_url.startswith("postgresql+psycopg://"):
        return "postgresql://" + database_url[len("postgresql+psycopg://") :]
    if database_url.startswith("postgresql+psycopg2://"):
        return "postgresql://" + database_url[len("postgresql+psycopg2://") :]
    return database_url


__all__ = [
    "DbClient",
    "SQLiteClient",
    "PostgresClient",
    "get_db_client",
    "normalize_database_url",
    "is_select_query",
]
