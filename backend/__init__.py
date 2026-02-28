"""Application package initialization and runtime defaults."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# This project uses OpenRouter as the model backend.
# Disable OpenAI Agents tracing globally to prevent non-fatal 401 tracing errors.
os.environ["OPENAI_AGENTS_DISABLE_TRACING"] = "1"

# Disable Chroma anonymized telemetry (avoids INFO log on startup).
os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")
