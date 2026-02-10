FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md /app/
COPY app /app/app
COPY llm_config.yaml /app/llm_config.yaml
COPY documents /app/documents

RUN uv sync --frozen

ENTRYPOINT ["python", "-m", "app.scripts.run_pipeline"]
