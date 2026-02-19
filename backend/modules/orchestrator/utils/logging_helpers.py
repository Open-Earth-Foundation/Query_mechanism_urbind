"""Logging utilities for orchestrator."""

import logging
from pathlib import Path


def attach_run_file_logger(run_dir: Path) -> logging.FileHandler:
    """Attach file handler to root logger for run-specific logging."""
    log_path = run_dir / "run.log"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.getLogger().level)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(filename)s:%(lineno)d - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logging.getLogger().addHandler(handler)
    return handler


__all__ = ["attach_run_file_logger"]
