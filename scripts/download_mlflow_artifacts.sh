#!/usr/bin/env bash

# Brief: Download all artifacts for one MLflow run from the URBIND experiment.
#
# Inputs:
# - CLI args:
#   - <RUN_ID>: MLflow run id (32-char hex), e.g. bcebd2f925264a90a0c7f2fa7ab7fb91
# - Env vars:
#   - DEST_DIR: Optional download directory override. Default: output/remote_artifact_downloads/<RUN_ID>
#
# Outputs:
# - Downloads the full run artifact tree under DEST_DIR.
# - Prints the resolved local path on success.
#
# Preferred usage (from project root; no venv activation needed):
# - bash scripts/download_mlflow_artifacts.sh bcebd2f925264a90a0c7f2fa7ab7fb91
#
# The script runs Python through `uv run`, so dependencies come from the project
# environment managed by uv. Install them once with `uv sync` if needed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# URBIND dev MLflow tracking server and experiment (same as deployed backend).
MLFLOW_TRACKING_URI="https://mlflow-dev.openearth.dev"
MLFLOW_EXPERIMENT_NAME="URBIND"

usage() {
  cat <<'EOF'
Usage (preferred, from project root; no venv activation needed):
  bash scripts/download_mlflow_artifacts.sh <RUN_ID>

Example:
  bash scripts/download_mlflow_artifacts.sh bcebd2f925264a90a0c7f2fa7ab7fb91

The script uses `uv run python` internally against the project environment.
Run `uv sync` once if dependencies are missing.

Downloads all artifacts for one run from the URBIND experiment on mlflow-dev.openearth.dev.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$#" -ne 1 ]]; then
  usage >&2
  exit 1
fi

RUN_ID="$1"
DEST_DIR="${DEST_DIR:-./output/remote_artifact_downloads/${RUN_ID}}"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf 'Required command not found: %s\n' "$1" >&2
    exit 1
  fi
}

require_command uv

printf 'MLflow tracking URI: %s\n' "$MLFLOW_TRACKING_URI"
printf 'MLflow experiment: %s\n' "$MLFLOW_EXPERIMENT_NAME"
printf 'Run ID: %s\n' "$RUN_ID"
printf 'Destination: %s\n' "$DEST_DIR"

cd "$REPO_ROOT"

uv run python - "$RUN_ID" "$MLFLOW_TRACKING_URI" "$MLFLOW_EXPERIMENT_NAME" "$DEST_DIR" <<'PYTHON'
import sys
from pathlib import Path

from mlflow import MlflowClient


def main() -> None:
    if len(sys.argv) != 5:
        print("ERROR: internal usage: download_mlflow_artifacts.py <run_id> <tracking_uri> <experiment_name> <dest_dir>")
        sys.exit(1)

    run_id, tracking_uri, experiment_name, dest_dir = sys.argv[1:5]

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"ERROR: MLflow experiment not found: {experiment_name}")
        sys.exit(1)

    try:
        run = client.get_run(run_id)
    except Exception as exc:  # noqa: BLE001 - surface MLflow lookup errors to the CLI
        print(f"ERROR: MLflow run not found: {run_id} ({exc})")
        sys.exit(1)

    if run.info.experiment_id != experiment.experiment_id:
        actual = client.get_experiment(run.info.experiment_id)
        actual_name = actual.name if actual is not None else run.info.experiment_id
        print(
            "ERROR: Run belongs to a different experiment. "
            f"Expected {experiment_name!r}, got {actual_name!r}."
        )
        sys.exit(1)

    print("Downloading all artifacts...")
    destination = Path(dest_dir)
    local_path = client.download_artifacts(
        run_id,
        "",
        dst_path=str(destination),
    )
    print(f"Done. Artifacts downloaded to: {local_path}")


if __name__ == "__main__":
    main()
PYTHON
