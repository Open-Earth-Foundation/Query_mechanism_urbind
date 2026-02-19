FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir uv

WORKDIR /app

# Install locked dependencies first for better build caching.
COPY pyproject.toml uv.lock /app/
RUN uv sync --frozen --no-dev --no-install-project

# Copy project sources and runtime assets, then install the local project.
COPY backend /app/backend
COPY assets /app/assets
COPY llm_config.yaml /app/llm_config.yaml
COPY documents /app/documents
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["sh", "-c", "uv run python -m uvicorn backend.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
