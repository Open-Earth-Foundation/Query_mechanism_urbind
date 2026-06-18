#!/usr/bin/env bash

# Brief: Run the maintenance-only Kubernetes vector-store refresh workflow.
#
# Inputs:
# - Env vars:
#   - NAMESPACE: Kubernetes namespace. Default: default
#   - BACKEND_DEPLOYMENT: Backend deployment to scale down/up. Default: urbind-query-mechanism-backend
#   - JOB_MANIFEST: Job manifest applied for the rebuild. Default: k8s/backend-build-vector-index-job.yml
#   - JOB_NAME: Kubernetes Job name created by the manifest. Default: urbind-query-mechanism-build-vector-index
#   - SCALE_UP_REPLICAS: Replicas restored after the Job completes. Default: 1
#   - JOB_TIMEOUT_SECONDS: Wait timeout for Job completion. Default: 7200
#   - ROLLOUT_TIMEOUT_SECONDS: Wait timeout for backend rollout after scale-up. Default: 600
# - CLI args:
#   - --namespace <name>: Override NAMESPACE
#   - --deployment <name>: Override BACKEND_DEPLOYMENT
#   - --job-manifest <path>: Override JOB_MANIFEST
#   - --job-name <name>: Override JOB_NAME
#   - --replicas <int>: Override SCALE_UP_REPLICAS
#   - --job-timeout <seconds>: Override JOB_TIMEOUT_SECONDS
#   - --rollout-timeout <seconds>: Override ROLLOUT_TIMEOUT_SECONDS
#
# Outputs:
# - Scales the backend deployment to 0, deletes any previous maintenance Job with the same
#   name, applies the maintenance Job manifest, waits for completion, streams logs, then
#   scales the backend deployment back up and waits for rollout readiness.
# - Prints progress to stdout/stderr. Does not mutate anything outside the targeted
#   namespace/deployment/job workflow.
#
# Usage (from project root):
# - bash scripts/update_vector_store_maintenance.sh
# - bash scripts/update_vector_store_maintenance.sh --namespace default --replicas 1

set -euo pipefail

NAMESPACE="${NAMESPACE:-default}"
BACKEND_DEPLOYMENT="${BACKEND_DEPLOYMENT:-urbind-query-mechanism-backend}"
JOB_MANIFEST="${JOB_MANIFEST:-k8s/backend-build-vector-index-job.yml}"
JOB_NAME="${JOB_NAME:-urbind-query-mechanism-build-vector-index}"
SCALE_UP_REPLICAS="${SCALE_UP_REPLICAS:-1}"
JOB_TIMEOUT_SECONDS="${JOB_TIMEOUT_SECONDS:-7200}"
ROLLOUT_TIMEOUT_SECONDS="${ROLLOUT_TIMEOUT_SECONDS:-600}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/update_vector_store_maintenance.sh [options]

Options:
  --namespace <name>         Kubernetes namespace (default: default)
  --deployment <name>        Backend deployment to scale (default: urbind-query-mechanism-backend)
  --job-manifest <path>      Job manifest path (default: k8s/backend-build-vector-index-job.yml)
  --job-name <name>          Job name from the manifest (default: urbind-query-mechanism-build-vector-index)
  --replicas <int>           Replicas to restore after completion (default: 1)
  --job-timeout <seconds>    Wait timeout for Job completion (default: 7200)
  --rollout-timeout <sec>    Wait timeout for backend rollout (default: 600)
  -h, --help                 Show this help
EOF
}

log() {
  printf '[vector-maintenance] %s\n' "$*"
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf 'Required command not found: %s\n' "$1" >&2
    exit 1
  fi
}

delete_existing_job_if_present() {
  if kubectl -n "$NAMESPACE" get job "$JOB_NAME" >/dev/null 2>&1; then
    log "Deleting existing Job $JOB_NAME so the manifest can create a fresh run."
    kubectl -n "$NAMESPACE" delete job "$JOB_NAME"
    kubectl -n "$NAMESPACE" wait --for=delete "job/$JOB_NAME" --timeout=180s
  fi
}

stream_job_logs() {
  set +e
  kubectl -n "$NAMESPACE" logs "job/$JOB_NAME" -f
  local log_exit=$?
  set -e
  if [[ "$log_exit" -ne 0 ]]; then
    log "Job log stream exited before completion; continuing with status checks."
  fi
}

scale_backend() {
  local replicas="$1"
  log "Scaling deployment/$BACKEND_DEPLOYMENT to $replicas replica(s)."
  kubectl -n "$NAMESPACE" scale "deployment/$BACKEND_DEPLOYMENT" --replicas="$replicas"
}

wait_for_backend_rollout() {
  log "Waiting for deployment/$BACKEND_DEPLOYMENT rollout."
  kubectl -n "$NAMESPACE" rollout status "deployment/$BACKEND_DEPLOYMENT" --timeout="${ROLLOUT_TIMEOUT_SECONDS}s"
}

cleanup_on_error() {
  local exit_code=$?
  if [[ "$exit_code" -ne 0 ]]; then
    printf '[vector-maintenance] Maintenance failed. Attempting to scale deployment/%s back to %s replica(s).\n' \
      "$BACKEND_DEPLOYMENT" "$SCALE_UP_REPLICAS" >&2
    kubectl -n "$NAMESPACE" scale "deployment/$BACKEND_DEPLOYMENT" --replicas="$SCALE_UP_REPLICAS" >/dev/null 2>&1 || true
  fi
  exit "$exit_code"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --namespace)
      NAMESPACE="$2"
      shift 2
      ;;
    --deployment)
      BACKEND_DEPLOYMENT="$2"
      shift 2
      ;;
    --job-manifest)
      JOB_MANIFEST="$2"
      shift 2
      ;;
    --job-name)
      JOB_NAME="$2"
      shift 2
      ;;
    --replicas)
      SCALE_UP_REPLICAS="$2"
      shift 2
      ;;
    --job-timeout)
      JOB_TIMEOUT_SECONDS="$2"
      shift 2
      ;;
    --rollout-timeout)
      ROLLOUT_TIMEOUT_SECONDS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown option: %s\n\n' "$1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

require_command kubectl

if [[ ! -f "$JOB_MANIFEST" ]]; then
  printf 'Job manifest not found: %s\n' "$JOB_MANIFEST" >&2
  exit 1
fi

trap cleanup_on_error EXIT

log "Starting maintenance vector-store refresh in namespace $NAMESPACE."
scale_backend 0
delete_existing_job_if_present

log "Applying Job manifest $JOB_MANIFEST."
kubectl -n "$NAMESPACE" apply -f "$JOB_MANIFEST"

stream_job_logs

log "Waiting for job/$JOB_NAME to complete."
kubectl -n "$NAMESPACE" wait --for=condition=complete "job/$JOB_NAME" --timeout="${JOB_TIMEOUT_SECONDS}s"

scale_backend "$SCALE_UP_REPLICAS"
wait_for_backend_rollout

trap - EXIT
log "Vector-store maintenance refresh completed successfully."
