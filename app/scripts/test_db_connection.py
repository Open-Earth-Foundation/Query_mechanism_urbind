"""
Brief: Test connectivity to the database using DATABASE_URL.

Inputs:
- DATABASE_URL (env var)
- --database-url (optional override)

Outputs:
- Logs success or failure

Usage (from project root):
- python -m app.scripts.test_db_connection
- python -m app.scripts.test_db_connection --database-url "postgresql+psycopg://..."
"""

from __future__ import annotations

import argparse
import logging

import psycopg

from app.services.db_client import normalize_database_url
from app.utils.config import get_database_url
from app.utils.logging_config import setup_logger

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Test database connectivity.")
    parser.add_argument(
        "--database-url",
        help="Override DATABASE_URL environment variable.",
    )
    return parser.parse_args()


def check_connection(database_url: str) -> None:
    """Attempt to connect and run a simple query."""
    normalized_url = normalize_database_url(database_url)
    logger.info("Testing database connection")

    with psycopg.connect(normalized_url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()

    logger.info("Database connection succeeded")


def main() -> None:
    """Script entry point."""
    args = parse_args()
    setup_logger()

    database_url = args.database_url or get_database_url()
    check_connection(database_url)


if __name__ == "__main__":
    main()
