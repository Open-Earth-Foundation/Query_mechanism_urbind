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
#   - STATUS_POLL_SECONDS: Poll interval for Job/Pod status output. Default: 10
#   - PENDING_TIMEOUT_SECONDS: Max time a Job pod may stay Pending/ContainerCreating before failing. Default: 180
# - CLI args:
#   - --namespace <name>: Override NAMESPACE
#   - --deployment <name>: Override BACKEND_DEPLOYMENT
#   - --job-manifest <path>: Override JOB_MANIFEST
#   - --job-name <name>: Override JOB_NAME
#   - --replicas <int>: Override SCALE_UP_REPLICAS
#   - --job-timeout <seconds>: Override JOB_TIMEOUT_SECONDS
#   - --rollout-timeout <seconds>: Override ROLLOUT_TIMEOUT_SECONDS
#   - --status-poll <seconds>: Override STATUS_POLL_SECONDS
#   - --pending-timeout <seconds>: Override PENDING_TIMEOUT_SECONDS
#
# Outputs:
# - Scales the backend deployment to 0, deletes any previous maintenance Job with the same
#   name, applies the maintenance Job manifest, prints live Job/Pod status while waiting,
#   surfaces attach/mount failures clearly, streams logs when available, then scales the
#   backend deployment back up and waits for rollout readiness.
# - Prints progress to stdout/stderr. Does not mutate anything outside the targeted
#   namespace/deployment/job workflow.
#
# Usage (from project root):
# - bash scripts/update_vector_store_maintenance.sh
# - bash scripts/update_vector_store_maintenance.sh --namespace default --replicas 1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

NAMESPACE="${NAMESPACE:-default}"
BACKEND_DEPLOYMENT="${BACKEND_DEPLOYMENT:-urbind-query-mechanism-backend}"
JOB_MANIFEST="${JOB_MANIFEST:-k8s/backend-build-vector-index-job.yml}"
JOB_NAME="${JOB_NAME:-urbind-query-mechanism-build-vector-index}"
SCALE_UP_REPLICAS="${SCALE_UP_REPLICAS:-1}"
JOB_TIMEOUT_SECONDS="${JOB_TIMEOUT_SECONDS:-7200}"
ROLLOUT_TIMEOUT_SECONDS="${ROLLOUT_TIMEOUT_SECONDS:-600}"
STATUS_POLL_SECONDS="${STATUS_POLL_SECONDS:-10}"
PENDING_TIMEOUT_SECONDS="${PENDING_TIMEOUT_SECONDS:-180}"
JOB_LOG_PID=""

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
  --status-poll <seconds>    Poll interval for status output (default: 10)
  --pending-timeout <sec>    Max Pending/ContainerCreating wait before failing (default: 180)
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

stop_job_log_stream() {
  if [[ -n "$JOB_LOG_PID" ]] && kill -0 "$JOB_LOG_PID" >/dev/null 2>&1; then
    kill "$JOB_LOG_PID" >/dev/null 2>&1 || true
    wait "$JOB_LOG_PID" >/dev/null 2>&1 || true
  fi
  JOB_LOG_PID=""
}

maybe_start_job_log_stream() {
  if [[ -n "$JOB_LOG_PID" ]] && kill -0 "$JOB_LOG_PID" >/dev/null 2>&1; then
    return
  fi
  stream_job_logs &
  JOB_LOG_PID=$!
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
  stop_job_log_stream
  if [[ "$exit_code" -ne 0 ]]; then
    printf '[vector-maintenance] Maintenance failed. Attempting to scale deployment/%s back to %s replica(s).\n' \
      "$BACKEND_DEPLOYMENT" "$SCALE_UP_REPLICAS" >&2
    kubectl -n "$NAMESPACE" scale "deployment/$BACKEND_DEPLOYMENT" --replicas="$SCALE_UP_REPLICAS" >/dev/null 2>&1 || true
  fi
  exit "$exit_code"
}

get_job_pod_name() {
  kubectl -n "$NAMESPACE" get pods -l "job-name=$JOB_NAME" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true
}

get_job_counter() {
  local field="$1"
  kubectl -n "$NAMESPACE" get job "$JOB_NAME" -o "jsonpath={.status.$field}" 2>/dev/null || true
}

get_pod_field() {
  local pod_name="$1"
  local field="$2"
  kubectl -n "$NAMESPACE" get pod "$pod_name" -o "jsonpath=$field" 2>/dev/null || true
}

print_pod_events() {
  local pod_name="$1"
  kubectl -n "$NAMESPACE" get events \
    --field-selector "involvedObject.name=$pod_name" \
    --sort-by=.lastTimestamp \
    --no-headers 2>/dev/null || true
}

print_failure_diagnostics() {
  local pod_name="$1"
  if [[ -n "$pod_name" ]]; then
    log "Recent events for pod/$pod_name:"
    local pod_events
    pod_events="$(print_pod_events "$pod_name")"
    if [[ -n "$pod_events" ]]; then
      printf '%s\n' "$pod_events"
    else
      log "No pod events found for $pod_name."
    fi
    log "Describe summary for pod/$pod_name:"
    kubectl -n "$NAMESPACE" describe pod "$pod_name" || true
    log "Current logs for pod/$pod_name:"
    kubectl -n "$NAMESPACE" logs "$pod_name" --tail=200 || true
  else
    log "No Job pod created yet; printing Job description instead."
    kubectl -n "$NAMESPACE" describe job "$JOB_NAME" || true
  fi
}

wait_for_job_completion() {
  local start_epoch
  start_epoch="$(date +%s)"
  local pending_since_epoch=""

  while true; do
    local now_epoch
    now_epoch="$(date +%s)"
    local elapsed_seconds=$((now_epoch - start_epoch))
    if (( elapsed_seconds > JOB_TIMEOUT_SECONDS )); then
      log "Timed out after ${JOB_TIMEOUT_SECONDS}s waiting for job/$JOB_NAME."
      print_failure_diagnostics "$(get_job_pod_name)"
      return 1
    fi

    local succeeded failed active
    succeeded="$(get_job_counter succeeded)"
    failed="$(get_job_counter failed)"
    active="$(get_job_counter active)"
    succeeded="${succeeded:-0}"
    failed="${failed:-0}"
    active="${active:-0}"

    local pod_name phase waiting_reason terminated_reason
    pod_name="$(get_job_pod_name)"
    phase="missing"
    waiting_reason=""
    terminated_reason=""

    if [[ -n "$pod_name" ]]; then
      phase="$(get_pod_field "$pod_name" '{.status.phase}')"
      waiting_reason="$(get_pod_field "$pod_name" '{.status.containerStatuses[0].state.waiting.reason}')"
      terminated_reason="$(get_pod_field "$pod_name" '{.status.containerStatuses[0].state.terminated.reason}')"
    fi

    log "Job status active=${active} succeeded=${succeeded} failed=${failed} pod=${pod_name:-none} phase=${phase:-unknown} waiting=${waiting_reason:-none} terminated=${terminated_reason:-none} elapsed=${elapsed_seconds}s"

    if [[ "$succeeded" -ge 1 ]]; then
      stop_job_log_stream
      return 0
    fi

    if [[ "$failed" -ge 1 ]] || [[ "$phase" == "Failed" ]]; then
      log "job/$JOB_NAME reported failure."
      print_failure_diagnostics "$pod_name"
      return 1
    fi

    if [[ "$phase" == "Running" ]] || [[ "$phase" == "Succeeded" ]]; then
      pending_since_epoch=""
      maybe_start_job_log_stream
    elif [[ "$phase" == "Pending" ]] || [[ "$waiting_reason" == "ContainerCreating" ]]; then
      if [[ -z "$pending_since_epoch" ]]; then
        pending_since_epoch="$now_epoch"
      fi

      local pod_events
      pod_events="$(print_pod_events "$pod_name")"
      if grep -q 'FailedAttachVolume' <<<"$pod_events"; then
        log "Detected FailedAttachVolume for pod/$pod_name. The PVC is still attached elsewhere."
        printf '%s\n' "$pod_events"
        print_failure_diagnostics "$pod_name"
        return 1
      fi

      local pending_elapsed=$((now_epoch - pending_since_epoch))
      if (( pending_elapsed > PENDING_TIMEOUT_SECONDS )); then
        log "Pod stayed Pending/ContainerCreating for ${pending_elapsed}s, exceeding ${PENDING_TIMEOUT_SECONDS}s."
        if [[ -n "$pod_events" ]]; then
          printf '%s\n' "$pod_events"
        fi
        print_failure_diagnostics "$pod_name"
        return 1
      fi
    else
      pending_since_epoch=""
    fi

    sleep "$STATUS_POLL_SECONDS"
  done
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
    --status-poll)
      STATUS_POLL_SECONDS="$2"
      shift 2
      ;;
    --pending-timeout)
      PENDING_TIMEOUT_SECONDS="$2"
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

if [[ "$JOB_MANIFEST" != /* ]]; then
  JOB_MANIFEST="${REPO_ROOT}/${JOB_MANIFEST}"
fi

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

log "Waiting for job/$JOB_NAME to complete."
wait_for_job_completion

scale_backend "$SCALE_UP_REPLICAS"
wait_for_backend_rollout

trap - EXIT
stop_job_log_stream
log "Vector-store maintenance refresh completed successfully."
