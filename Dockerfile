FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY backend /app/backend
COPY llm_config.yaml /app/llm_config.yaml
COPY documents /app/documents

RUN pip install --no-cache-dir -e .

ENTRYPOINT ["python", "-m", "backend.scripts.run_pipeline"]
