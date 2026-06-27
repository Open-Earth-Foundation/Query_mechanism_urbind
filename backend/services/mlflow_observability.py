"""Optional MLflow run mirroring and trace creation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from backend.services.llm_observability import LlmCallRecorder, safe_serialize

logger = logging.getLogger(__name__)


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk, returning an empty dict on failure."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _metric_value(value: object) -> float | None:
    """Return a numeric metric value accepted by MLflow."""
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _build_tags(api_state: dict[str, Any], experiment_name: str) -> dict[str, str]:
    """Build stable MLflow tags from api_state.json."""
    inputs = api_state.get("inputs")
    if not isinstance(inputs, dict):
        inputs = {}
    tags = {
        "run_id": str(api_state.get("run_id") or ""),
        "status": str(api_state.get("status") or ""),
        "finish_reason": str(api_state.get("finish_reason") or ""),
        "experiment": experiment_name,
        "query_mode": str(inputs.get("query_mode") or ""),
        "analysis_mode": str(inputs.get("analysis_mode") or ""),
        "city_scope_mode": str(inputs.get("city_scope_mode") or ""),
    }
    return {key: value for key, value in tags.items() if value}


def _build_metrics(api_state: dict[str, Any], call_count: int) -> dict[str, float]:
    """Build numeric MLflow metrics from api_state.json and call artifacts."""
    inputs = api_state.get("inputs")
    if not isinstance(inputs, dict):
        inputs = {}
    metrics: dict[str, float] = {
        "llm_call_artifact_count": float(call_count),
    }
    for metric_name, key in (
        ("selected_city_count", "selected_cities_planned"),
        ("markdown_file_count", "markdown_file_count"),
        ("markdown_chunk_count", "markdown_chunk_count"),
        ("markdown_excerpt_count", "markdown_excerpt_count"),
    ):
        value = inputs.get(key)
        if isinstance(value, list):
            metrics[metric_name] = float(len(value))
            continue
        numeric = _metric_value(value)
        if numeric is not None:
            metrics[metric_name] = numeric

    llm_usage = api_state.get("llm_usage")
    if isinstance(llm_usage, dict):
        calls = _metric_value(llm_usage.get("calls"))
        if calls is not None:
            metrics["llm_calls"] = calls
        totals = llm_usage.get("totals")
        if isinstance(totals, dict):
            for name in ("input_tokens", "output_tokens", "total_tokens"):
                value = _metric_value(totals.get(name))
                if value is not None:
                    metrics[f"llm_{name}"] = value

    retry_summary = api_state.get("retry_summary")
    if isinstance(retry_summary, dict):
        for source_key, metric_name in (
            ("total_events", "retry_events"),
            ("exhausted_events", "retry_exhausted_events"),
        ):
            value = _metric_value(retry_summary.get(source_key))
            if value is not None:
                metrics[metric_name] = value
    return metrics


def _span_attributes(call: dict[str, Any]) -> dict[str, object]:
    """Return compact attributes for one MLflow LLM span."""
    usage = call.get("response")
    if isinstance(usage, dict):
        usage = usage.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    attrs: dict[str, object] = {
        "run_id": call.get("run_id"),
        "stage_number": call.get("stage_number"),
        "stage_name": call.get("stage_name"),
        "stage_family": call.get("stage_family"),
        "agent": call.get("agent"),
        "call_kind": call.get("call_kind"),
        "provider": call.get("provider"),
        "model": call.get("model"),
        "status": call.get("status"),
    }
    for key in ("input_tokens", "prompt_tokens", "output_tokens", "completion_tokens", "total_tokens"):
        if key in usage:
            attrs[f"usage.{key}"] = usage[key]
    return {key: value for key, value in attrs.items() if value is not None}


def _set_span_payload(span: Any, *, inputs: object, outputs: object) -> None:
    """Set span inputs and outputs when supported by the MLflow version."""
    set_inputs = getattr(span, "set_inputs", None)
    if callable(set_inputs):
        set_inputs(safe_serialize(inputs))
    set_outputs = getattr(span, "set_outputs", None)
    if callable(set_outputs):
        set_outputs(safe_serialize(outputs))


def _create_trace(
    mlflow_module: Any,
    *,
    run_id: str,
    trace_family: str,
    calls: list[dict[str, Any]],
    api_state: dict[str, Any],
) -> str | None:
    """Create one MLflow trace from recorded LLM calls."""
    if not calls:
        return None
    root_name = f"{run_id}:{trace_family}"
    attributes = {
        "run_id": run_id,
        "trace_family": trace_family,
        "trace_group": run_id,
        "status": api_state.get("status"),
    }
    with mlflow_module.start_span(root_name, span_type="CHAIN", attributes=attributes) as root:
        update_current_trace = getattr(mlflow_module, "update_current_trace", None)
        if callable(update_current_trace):
            update_current_trace(
                tags={
                    "run_id": run_id,
                    "trace_family": trace_family,
                    "trace_group": run_id,
                }
            )
        _set_span_payload(
            root,
            inputs={
                "run_id": run_id,
                "question": api_state.get("question"),
                "trace_family": trace_family,
            },
            outputs={"llm_call_count": len(calls), "status": api_state.get("status")},
        )
        for call in calls:
            span_name = (
                f"{call.get('call_index', 'unknown')}:"
                f"{call.get('stage_family', 'unknown')}:"
                f"{call.get('agent', 'unknown')}"
            )
            with mlflow_module.start_span(
                span_name,
                span_type="LLM",
                attributes=_span_attributes(call),
            ) as span:
                outputs = call.get("response")
                if call.get("status") == "error":
                    outputs = {"error": call.get("error")}
                _set_span_payload(
                    span,
                    inputs=call.get("request"),
                    outputs=outputs,
                )
    return str(getattr(root, "trace_id", "") or "") or None


def _create_traces(
    mlflow_module: Any,
    *,
    config: Any,
    run_id: str,
    calls: list[dict[str, Any]],
    api_state: dict[str, Any],
) -> dict[str, object]:
    """Create consolidated trace with split fallback for markdown and assumptions."""
    trace_result: dict[str, object] = {
        "mode": getattr(config, "trace_mode", "consolidated"),
        "trace_ids": [],
        "fallback_used": False,
    }
    if not calls:
        return trace_result

    try:
        trace_id = _create_trace(
            mlflow_module,
            run_id=run_id,
            trace_family="pipeline",
            calls=calls,
            api_state=api_state,
        )
        if trace_id:
            trace_result["trace_ids"] = [trace_id]
        return trace_result
    except Exception as exc:  # noqa: BLE001
        logger.warning("Consolidated MLflow trace failed; falling back. error=%s", exc)
        trace_result["fallback_used"] = True

    fallback_ids: list[str] = []
    for family in ("markdown", "assumptions"):
        family_calls = [call for call in calls if call.get("stage_family") == family]
        try:
            trace_id = _create_trace(
                mlflow_module,
                run_id=run_id,
                trace_family=family,
                calls=family_calls,
                api_state=api_state,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("MLflow %s fallback trace failed. error=%s", family, exc)
            continue
        if trace_id:
            fallback_ids.append(trace_id)
    trace_result["trace_ids"] = fallback_ids
    return trace_result


def sync_run_to_mlflow(
    *,
    run_logger: Any,
    config: Any,
    recorder: LlmCallRecorder | None,
    mlflow_module: Any | None = None,
) -> dict[str, object]:
    """Mirror a finalized run directory to MLflow when enabled."""
    if not bool(getattr(config, "enabled", False)):
        return {"enabled": False, "sync_status": "disabled"}

    run_dir = run_logger.run_paths.base_dir
    run_id = run_dir.name
    experiment_name = str(getattr(config, "experiment_name", "URBIND"))
    artifact_path = str(getattr(config, "artifact_path", "run_artifacts"))
    metadata: dict[str, object] = {
        "enabled": True,
        "sync_status": "started",
        "run_id": run_id,
        "experiment_name": experiment_name,
        "artifact_path": artifact_path,
    }

    try:
        if mlflow_module is None:
            import mlflow as mlflow_module  # type: ignore[no-redef]

        tracking_uri = getattr(config, "tracking_uri", None)
        if tracking_uri:
            mlflow_module.set_tracking_uri(str(tracking_uri))
        mlflow_module.set_experiment(experiment_name)

        api_state = _read_json(run_logger.run_paths.api_state)
        calls = recorder.read_call_records() if recorder is not None else []
        with mlflow_module.start_run(run_name=run_id) as active_run:
            mlflow_run_id = str(getattr(active_run.info, "run_id", ""))
            metadata["mlflow_run_id"] = mlflow_run_id
            mlflow_module.set_tags(_build_tags(api_state, experiment_name))
            mlflow_module.log_metrics(_build_metrics(api_state, len(calls)))
            trace_payload = _create_traces(
                mlflow_module,
                config=config,
                run_id=run_id,
                calls=calls,
                api_state=api_state,
            )
            metadata["traces"] = trace_payload
            metadata["sync_status"] = "uploading"
            run_logger.record_mlflow_metadata(metadata)
            mlflow_module.log_artifacts(str(run_dir), artifact_path=artifact_path)
            metadata["sync_status"] = "completed"
            run_logger.record_mlflow_metadata(metadata)
            mlflow_module.log_artifact(
                str(run_logger.run_paths.api_state),
                artifact_path=artifact_path,
            )
            mlflow_module.log_artifact(
                str(run_logger.run_paths.manifest),
                artifact_path=artifact_path,
            )
        return metadata
    except Exception as exc:  # noqa: BLE001
        metadata["sync_status"] = "failed"
        metadata["error"] = {"type": type(exc).__name__, "message": str(exc)}
        try:
            run_logger.record_mlflow_metadata(metadata)
        except Exception:  # noqa: BLE001
            logger.warning("Could not persist failed MLflow metadata.", exc_info=True)
        if bool(getattr(config, "fail_on_error", False)):
            raise
        logger.warning("MLflow sync failed for run_id=%s: %s", run_id, exc)
        return metadata


__all__ = ["sync_run_to_mlflow"]
