from __future__ import annotations

import asyncio
import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from backend.modules.markdown_researcher.models import (
    MarkdownExcerpt,
    MarkdownResearchResult,
)
from backend.modules.orchestrator.module import run_pipeline
from backend.modules.web_researcher.assumptions_estimator import _call_estimator
from backend.modules.web_researcher.models import (
    EnrichedField,
    FieldClassification,
    GapManifest,
)
from backend.modules.writer.models import WriterOutput
from backend.services.agents import LlmCallRecordingHooks, run_agent_sync
from backend.services.llm_observability import LlmCallContext, LlmCallRecorder
from backend.services.mlflow_observability import sync_run_to_mlflow
from backend.services.run_logger import RunLogger
from backend.utils.paths import create_run_paths
from tests.support import build_test_app_config


class _FakeMlflowRunInfo:
    def __init__(self, run_id: str = "mlflow-run-id") -> None:
        self.run_id = run_id


class _FakeMlflowRun:
    def __init__(self, run_id: str = "mlflow-run-id") -> None:
        self.info = _FakeMlflowRunInfo(run_id)

    def __enter__(self) -> "_FakeMlflowRun":
        return self

    def __exit__(self, *_args: object) -> None:
        return None


class _FakeSpan:
    def __init__(self, name: str) -> None:
        self.name = name
        self.trace_id = f"trace-{name}"
        self.inputs: object | None = None
        self.outputs: object | None = None

    def __enter__(self) -> "_FakeSpan":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def set_inputs(self, inputs: object) -> None:
        self.inputs = inputs

    def set_outputs(self, outputs: object) -> None:
        self.outputs = outputs


class _FakeMlflow:
    def __init__(self, *, fail_pipeline_trace: bool = False, fail_log_artifacts: bool = False) -> None:
        self.fail_pipeline_trace = fail_pipeline_trace
        self.fail_log_artifacts = fail_log_artifacts
        self.started_run_id: str | None = None
        self.run_name: str | None = None
        self.tracking_uri: str | None = None
        self.experiment_name: str | None = None
        self.tags: dict[str, str] = {}
        self.metrics: dict[str, float] = {}
        self.logged_artifacts: list[tuple[str, str | None]] = []
        self.logged_artifact_files: list[tuple[str, str | None]] = []
        self.spans: list[dict[str, object]] = []
        self.span_objects: list[_FakeSpan] = []
        self.trace_tags: list[dict[str, str]] = []

    def set_tracking_uri(self, uri: str) -> None:
        self.tracking_uri = uri

    def set_experiment(self, name: str) -> None:
        self.experiment_name = name

    def start_run(
        self,
        run_name: str | None = None,
        run_id: str | None = None,
    ) -> _FakeMlflowRun:
        self.run_name = run_name
        self.started_run_id = run_id
        return _FakeMlflowRun(run_id or "mlflow-run-id")

    def set_tags(self, tags: dict[str, str]) -> None:
        self.tags.update(tags)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self.metrics.update(metrics)

    def log_artifacts(self, local_dir: str, artifact_path: str | None = None) -> None:
        if self.fail_log_artifacts:
            raise RuntimeError("artifact upload failed")
        self.logged_artifacts.append((local_dir, artifact_path))

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        self.logged_artifact_files.append((local_path, artifact_path))

    def start_span(
        self,
        name: str,
        *,
        span_type: str,
        attributes: dict[str, object],
    ) -> _FakeSpan:
        if self.fail_pipeline_trace and name.endswith(":pipeline"):
            raise RuntimeError("pipeline trace failed")
        self.spans.append(
            {
                "name": name,
                "span_type": span_type,
                "attributes": attributes,
            }
        )
        span = _FakeSpan(name)
        self.span_objects.append(span)
        return span

    def update_current_trace(self, *, tags: dict[str, str]) -> None:
        self.trace_tags.append(tags)


def _join_content_parts(value: object) -> str:
    """Join trace-only content parts produced for MLflow display."""
    assert isinstance(value, dict)
    content_parts = value["content_parts"]
    assert isinstance(content_parts, list)
    joined = []
    for part in content_parts:
        assert isinstance(part, dict)
        text = part["text"]
        assert isinstance(text, str)
        joined.append(text)
    return "".join(joined)


def _finalized_run_logger(tmp_path: Path) -> RunLogger:
    paths = create_run_paths(tmp_path, "mlflow-test-run", "context_bundle.json")
    run_logger = RunLogger(paths, "Question?")
    paths.final_output.write_text("# Final\nAnswer", encoding="utf-8")
    run_logger.finalize(
        "completed",
        final_output_path=paths.final_output,
        finish_reason="completed (write)",
    )
    return run_logger


def test_llm_call_recorder_writes_stage_payload_and_thread_safe_index(
    tmp_path: Path,
) -> None:
    recorder = LlmCallRecorder(tmp_path, "run-1")

    def _record(index: int) -> None:
        recorder.record_call(
            LlmCallContext(
                stage_name="markdown_extraction",
                stage_family="markdown",
                agent="markdown_researcher",
                call_kind="batch_extraction",
                model="test-model",
                metadata={"worker": index},
            ),
            request={"messages": [{"role": "user", "content": f"q{index}"}]},
            response={"usage": {"total_tokens": index + 1}},
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(_record, range(8)))

    index_rows = [
        json.loads(line)
        for line in (tmp_path / "llm_calls" / "index.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    assert len(index_rows) == 8
    assert sorted(row["call_index"] for row in index_rows) == list(range(1, 9))
    assert all(row["stage_family"] == "markdown" for row in index_rows)
    first_payload = json.loads((tmp_path / index_rows[0]["path"]).read_text(encoding="utf-8"))
    assert first_payload["stage_number"] == 6
    assert first_payload["agent"] == "markdown_researcher"
    assert first_payload["request"]["messages"][0]["role"] == "user"


def test_agents_sdk_hook_records_fake_model_response(tmp_path: Path) -> None:
    recorder = LlmCallRecorder(tmp_path, "run-2")
    hook = LlmCallRecordingHooks(
        recorder,
        LlmCallContext(
            stage_name="writer",
            stage_family="writer",
            agent="writer",
            call_kind="write_markdown",
            model="test-model",
        ),
    )
    agent = type("Agent", (), {"name": "Writer"})()
    class _Response:
        def __init__(self) -> None:
            self.response_id = "resp-1"
            self.usage = {"total_tokens": 11}
            self.output = [
                {"type": "message", "content": [{"type": "text", "text": "ok"}]}
            ]

    response = _Response()

    async def _run_hook() -> None:
        await hook.on_llm_start(None, agent, "system", [{"role": "user", "content": "hi"}])
        await hook.on_llm_end(None, agent, response)

    asyncio.run(_run_hook())

    records = recorder.read_call_records()
    assert len(records) == 1
    assert records[0]["stage_family"] == "writer"
    assert records[0]["request"]["system_prompt"] == "system"
    assert records[0]["response"]["response_id"] == "resp-1"


def test_run_agent_sync_records_pending_llm_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = LlmCallRecorder(tmp_path, "run-error")
    agent = type("Agent", (), {"name": "Writer"})()

    async def _raise_after_llm_start(
        agent: object,
        input_data: str,
        *,
        max_turns: int,
        hooks: object,
    ) -> object:
        del input_data, max_turns
        await hooks.on_llm_start(
            None,
            agent,
            "system",
            [{"role": "user", "content": "hi"}],
        )
        raise RuntimeError("provider down")

    monkeypatch.setattr(
        "backend.services.agents.Runner.run",
        _raise_after_llm_start,
    )

    with pytest.raises(RuntimeError, match="provider down"):
        run_agent_sync(
            agent,
            "input",
            llm_recorder=recorder,
            llm_call_context=LlmCallContext(
                stage_name="writer",
                stage_family="writer",
                agent="writer",
                call_kind="write_markdown",
                model="test-model",
            ),
        )

    records = recorder.read_call_records()
    assert len(records) == 1
    assert records[0]["status"] == "error"
    assert records[0]["error"]["type"] == "RuntimeError"
    assert records[0]["request"]["system_prompt"] == "system"


def test_assumptions_estimator_direct_openai_call_is_recorded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = build_test_app_config(runs_dir=tmp_path / "output")
    recorder = LlmCallRecorder(tmp_path, "run-3")

    class _StubCompletions:
        def create(self, **_kwargs: object) -> object:
            return type(
                "Response",
                (),
                {
                    "choices": [
                        type(
                            "Choice",
                            (),
                            {
                                "message": type(
                                    "Message",
                                    (),
                                    {
                                        "content": (
                                            '{"assumptions":[{"city":"Aachen",'
                                            '"field_name":"vehicle_count",'
                                            '"gap_description":"Missing vehicles",'
                                            '"method_used":"peer_city_proxy",'
                                            '"estimate":{"low":1,"mid":2,"high":3},'
                                            '"confidence":"MEDIUM",'
                                            '"reference_data":"Munich",'
                                            '"rationale":"test","basis":"test",'
                                            '"is_replaceable":true}]}'
                                        )
                                    },
                                )()
                            },
                        )()
                    ],
                    "usage": {"total_tokens": 42},
                },
            )()

    class _StubOpenAI:
        def __init__(self, **_kwargs: object) -> None:
            self.chat = type("Chat", (), {"completions": _StubCompletions()})()

    monkeypatch.setattr(
        "backend.modules.web_researcher.assumptions_estimator.OpenAI",
        _StubOpenAI,
    )
    gap_manifest = GapManifest(
        query_fields=[
            FieldClassification(
                field="vehicle_count",
                classification="estimable_numerical",
                searchable=True,
                rationale="test",
            )
        ],
        city_gaps=[],
        non_estimable_fields=[],
    )
    records = _call_estimator(
        question="Question?",
        context_bundle={"markdown": {"excerpt_count": 1}},
        gap_manifest=gap_manifest,
        estimable_fields=[
            EnrichedField(
                city="Aachen",
                field="vehicle_count",
                status="still_missing",
                value=None,
                source="none",
                provenance={},
            )
        ],
        all_enriched_fields=[],
        config=config,
        api_key="sk-test",
        pass_name="generate",
        llm_recorder=recorder,
    )

    assert len(records) == 1
    llm_records = recorder.read_call_records()
    assert llm_records[0]["stage_family"] == "assumptions"
    assert llm_records[0]["agent"] == "assumptions_estimator"
    assert llm_records[0]["call_kind"] == "generate_estimates"


def test_mlflow_sync_logs_full_run_dir_and_consolidated_trace(tmp_path: Path) -> None:
    run_logger = _finalized_run_logger(tmp_path)
    recorder = LlmCallRecorder(run_logger.run_paths.base_dir, run_logger.run_paths.base_dir.name)
    recorder.record_call(
        LlmCallContext(
            stage_name="markdown_extraction",
            stage_family="markdown",
            agent="markdown_researcher",
            call_kind="batch_extraction",
            model="test-model",
        ),
        request={"messages": []},
        response={"usage": {"total_tokens": 3}},
    )
    config = build_test_app_config(
        runs_dir=tmp_path,
        mlflow_overrides={
            "enabled": True,
            "tracking_uri": "file:///tmp/mlruns",
            "experiment_name": "URBIND_TEST",
            "environment": "test",
            "artifact_path": "run_artifacts",
        },
    ).mlflow
    fake_mlflow = _FakeMlflow()

    metadata = sync_run_to_mlflow(
        run_logger=run_logger,
        config=config,
        recorder=recorder,
        mlflow_module=fake_mlflow,
    )

    assert metadata["sync_status"] == "completed"
    assert fake_mlflow.logged_artifacts == [
        (str(run_logger.run_paths.base_dir), "run_artifacts")
    ]
    assert fake_mlflow.tags["run_id"] == run_logger.run_paths.base_dir.name
    assert fake_mlflow.tags["environment"] == "test"
    assert fake_mlflow.metrics["llm_call_artifact_count"] == 1.0
    assert any(span["name"].endswith(":pipeline") for span in fake_mlflow.spans)
    api_state = json.loads(run_logger.run_paths.api_state.read_text(encoding="utf-8"))
    manifest = json.loads(run_logger.run_paths.manifest.read_text(encoding="utf-8"))
    assert api_state["mlflow"]["mlflow_run_id"] == "mlflow-run-id"
    assert api_state["mlflow"]["environment"] == "test"
    assert manifest["metadata"]["mlflow"]["sync_status"] == "completed"


def test_mlflow_sync_formats_agents_payloads_for_trace_view(
    tmp_path: Path,
) -> None:
    run_logger = _finalized_run_logger(tmp_path)
    recorder = LlmCallRecorder(
        run_logger.run_paths.base_dir,
        run_logger.run_paths.base_dir.name,
    )
    system_prompt = "System instructions " * 220
    long_request = "request content " * 220
    long_response = "response content " * 220
    request_json = json.dumps(
        {
            "question": "How much solar storage?",
            "chunks": [
                {
                    "path": "documents/Lodz.md",
                    "content": long_request,
                }
            ],
        }
    )
    call_path = recorder.record_call(
        LlmCallContext(
            stage_name="markdown_extraction",
            stage_family="markdown",
            agent="markdown_researcher",
            call_kind="batch_extraction",
            model="test-model",
        ),
        request={
            "system_prompt": system_prompt,
            "input_items": [{"role": "user", "content": request_json}],
            "agent": "Markdown Researcher",
        },
        response={"output": [{"type": "text", "text": long_response}]},
    )
    config = build_test_app_config(
        runs_dir=tmp_path,
        mlflow_overrides={"enabled": True},
    ).mlflow
    fake_mlflow = _FakeMlflow()

    metadata = sync_run_to_mlflow(
        run_logger=run_logger,
        config=config,
        recorder=recorder,
        mlflow_module=fake_mlflow,
    )

    assert metadata["sync_status"] == "completed"
    llm_span = next(
        span for span in fake_mlflow.span_objects if span.name.startswith("1:markdown")
    )
    assert isinstance(llm_span.inputs, dict)
    assert "input_items" not in llm_span.inputs
    assert "system_prompt" not in llm_span.inputs
    assert llm_span.inputs["agent"] == "Markdown Researcher"
    messages = llm_span.inputs["messages"]
    assert isinstance(messages, list)
    assert messages[0]["role"] == "system"
    assert _join_content_parts(messages[0]["content"]) == system_prompt
    assert messages[1]["role"] == "user"
    user_content = messages[1]["content"]
    assert isinstance(user_content, dict)
    assert user_content["question"] == "How much solar storage?"
    chunks = user_content["chunks"]
    assert isinstance(chunks, list)
    first_chunk = chunks[0]
    assert isinstance(first_chunk, dict)
    assert first_chunk["path"] == "documents/Lodz.md"
    assert _join_content_parts(first_chunk["content"]) == long_request

    assert isinstance(llm_span.outputs, dict)
    output = llm_span.outputs["output"]
    assert isinstance(output, list)
    first_output = output[0]
    assert isinstance(first_output, dict)
    assert _join_content_parts(first_output["text"]) == long_response

    raw_call = json.loads(call_path.read_text(encoding="utf-8"))
    assert raw_call["request"]["input_items"][0]["content"] == request_json
    assert raw_call["response"]["output"][0]["text"] == long_response


def test_mlflow_sync_reconfigures_charmap_stdout_for_mlflow_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_logger = _finalized_run_logger(tmp_path)
    config = build_test_app_config(
        runs_dir=tmp_path,
        mlflow_overrides={"enabled": True},
    ).mlflow

    class _EmojiMlflow(_FakeMlflow):
        def start_run(
            self,
            run_name: str | None = None,
            run_id: str | None = None,
        ) -> _FakeMlflowRun:
            print("\U0001f3c3 View run")
            return super().start_run(run_name=run_name, run_id=run_id)

    stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
    monkeypatch.setattr(sys, "stdout", stream)

    metadata = sync_run_to_mlflow(
        run_logger=run_logger,
        config=config,
        recorder=None,
        mlflow_module=_EmojiMlflow(),
    )

    assert metadata["sync_status"] == "completed"


def test_mlflow_sync_reuses_failed_run_and_trace_on_retry(tmp_path: Path) -> None:
    run_logger = _finalized_run_logger(tmp_path)
    existing_metadata = {
        "enabled": True,
        "sync_status": "failed",
        "run_id": run_logger.run_paths.base_dir.name,
        "experiment_name": "URBIND_TEST",
        "artifact_path": "run_artifacts",
        "mlflow_run_id": "existing-mlflow-run",
        "traces": {
            "mode": "consolidated",
            "trace_ids": ["trace-existing"],
            "fallback_used": False,
        },
        "error": {"type": "UnicodeEncodeError", "message": "console encoding"},
    }
    run_logger.record_mlflow_metadata(existing_metadata)
    config = build_test_app_config(
        runs_dir=tmp_path,
        mlflow_overrides={
            "enabled": True,
            "experiment_name": "URBIND_TEST",
            "artifact_path": "run_artifacts",
        },
    ).mlflow
    fake_mlflow = _FakeMlflow()

    metadata = sync_run_to_mlflow(
        run_logger=run_logger,
        config=config,
        recorder=None,
        mlflow_module=fake_mlflow,
    )

    assert metadata["sync_status"] == "completed"
    assert metadata["mlflow_run_id"] == "existing-mlflow-run"
    assert metadata["traces"] == existing_metadata["traces"]
    assert fake_mlflow.started_run_id == "existing-mlflow-run"
    assert fake_mlflow.spans == []


def test_force_mlflow_sync_creates_post_run_assumptions_trace(tmp_path: Path) -> None:
    run_logger = _finalized_run_logger(tmp_path)
    existing_metadata = {
        "enabled": True,
        "sync_status": "completed",
        "run_id": run_logger.run_paths.base_dir.name,
        "experiment_name": "URBIND_TEST",
        "artifact_path": "run_artifacts",
        "mlflow_run_id": "existing-mlflow-run",
        "traces": {
            "mode": "consolidated",
            "trace_ids": ["trace-existing"],
            "fallback_used": False,
        },
    }
    run_logger.record_mlflow_metadata(existing_metadata)
    recorder = LlmCallRecorder(
        run_logger.run_paths.base_dir,
        run_logger.run_paths.base_dir.name,
    )
    recorder.record_call(
        LlmCallContext(
            stage_name="assumptions_apply",
            stage_family="assumptions",
            agent="assumptions_apply_writer",
            call_kind="apply_assumptions",
            model="test-model",
        ),
        request={"messages": []},
        response={"usage": {"total_tokens": 1}},
    )
    config = build_test_app_config(
        runs_dir=tmp_path,
        mlflow_overrides={"enabled": True},
    ).mlflow
    fake_mlflow = _FakeMlflow()

    metadata = sync_run_to_mlflow(
        run_logger=run_logger,
        config=config,
        recorder=recorder,
        mlflow_module=fake_mlflow,
        force=True,
    )

    supplemental_trace_id = (
        f"trace-{run_logger.run_paths.base_dir.name}:post_run_assumptions"
    )
    traces = metadata["traces"]
    assert isinstance(traces, dict)
    assert traces["trace_ids"] == ["trace-existing", supplemental_trace_id]
    assert traces["supplemental_trace_ids"] == [supplemental_trace_id]
    assert traces["post_run_trace_max_call_index"] == 1
    assert traces["post_run_trace_call_count"] == 1
    assert fake_mlflow.started_run_id == "existing-mlflow-run"
    assert any(
        span["name"].endswith(":post_run_assumptions")
        for span in fake_mlflow.spans
    )

    second_fake_mlflow = _FakeMlflow()
    second_metadata = sync_run_to_mlflow(
        run_logger=run_logger,
        config=config,
        recorder=recorder,
        mlflow_module=second_fake_mlflow,
        force=True,
    )

    assert second_metadata["traces"] == traces
    assert second_fake_mlflow.spans == []


def test_mlflow_sync_falls_back_to_markdown_and_assumptions_traces(
    tmp_path: Path,
) -> None:
    run_logger = _finalized_run_logger(tmp_path)
    recorder = LlmCallRecorder(run_logger.run_paths.base_dir, run_logger.run_paths.base_dir.name)
    for family, agent in (
        ("markdown", "markdown_researcher"),
        ("assumptions", "assumptions_estimator"),
    ):
        recorder.record_call(
            LlmCallContext(
                stage_name="assumptions" if family == "assumptions" else "markdown_extraction",
                stage_family=family,
                agent=agent,
                call_kind="test_call",
                model="test-model",
            ),
            request={"messages": []},
            response={"usage": {"total_tokens": 1}},
        )
    config = build_test_app_config(
        runs_dir=tmp_path,
        mlflow_overrides={"enabled": True},
    ).mlflow
    fake_mlflow = _FakeMlflow(fail_pipeline_trace=True)

    metadata = sync_run_to_mlflow(
        run_logger=run_logger,
        config=config,
        recorder=recorder,
        mlflow_module=fake_mlflow,
    )

    traces = metadata["traces"]
    assert isinstance(traces, dict)
    assert traces["fallback_used"] is True
    assert any(span["name"].endswith(":markdown") for span in fake_mlflow.spans)
    assert any(span["name"].endswith(":assumptions") for span in fake_mlflow.spans)


def test_mlflow_sync_is_best_effort_when_upload_fails(tmp_path: Path) -> None:
    run_logger = _finalized_run_logger(tmp_path)
    config = build_test_app_config(
        runs_dir=tmp_path,
        mlflow_overrides={"enabled": True, "fail_on_error": False},
    ).mlflow

    metadata = sync_run_to_mlflow(
        run_logger=run_logger,
        config=config,
        recorder=None,
        mlflow_module=_FakeMlflow(fail_log_artifacts=True),
    )

    assert metadata["sync_status"] == "failed"
    assert metadata["error"]["type"] == "RuntimeError"
    api_state = json.loads(run_logger.run_paths.api_state.read_text(encoding="utf-8"))
    assert api_state["mlflow"]["sync_status"] == "failed"


def test_run_pipeline_persists_mlflow_metadata_with_fake_sync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    (docs_dir / "Munich.md").write_text("# Munich\n\nSample", encoding="utf-8")
    config = build_test_app_config(
        runs_dir=tmp_path / "output",
        markdown_dir=docs_dir,
        vector_store_overrides={"enabled": False},
        mlflow_overrides={"enabled": True},
    )

    def _stub_markdown(
        question: str,
        documents: list[dict[str, str]],
        config: object,
        api_key: str,
        **kwargs: object,
    ) -> MarkdownResearchResult:
        del question, documents, config, api_key
        recorder = kwargs.get("llm_recorder")
        if isinstance(recorder, LlmCallRecorder):
            recorder.record_call(
                LlmCallContext(
                    stage_name="markdown_extraction",
                    stage_family="markdown",
                    agent="markdown_researcher",
                    call_kind="batch_extraction",
                    model="test-model",
                ),
                request={"messages": []},
                response={"usage": {"total_tokens": 1}},
            )
        return MarkdownResearchResult(
            excerpts=[
                MarkdownExcerpt(
                    quote="Munich sample",
                    city_name="Munich",
                    partial_answer="Munich sample",
                )
            ]
        )

    def _stub_writer(
        question: str,
        context_bundle: dict[str, object],
        config: object,
        api_key: str,
        **_kwargs: object,
    ) -> WriterOutput:
        del question, context_bundle, config, api_key
        return WriterOutput(content="# Answer\n\nStub")

    def _fake_sync(*, run_logger: RunLogger, config: object, recorder: object) -> dict[str, object]:
        metadata = {
            "enabled": True,
            "sync_status": "completed",
            "mlflow_run_id": "fake-run",
            "experiment_name": "URBIND",
            "artifact_path": "run_artifacts",
            "traces": {"trace_ids": ["trace-1"], "fallback_used": False},
        }
        run_logger.record_mlflow_metadata(metadata)
        return metadata

    monkeypatch.setattr(
        "backend.modules.orchestrator.module.sync_run_to_mlflow",
        _fake_sync,
    )

    paths = run_pipeline(
        question="What initiatives exist for Munich?",
        config=config,
        markdown_func=_stub_markdown,
        writer_func=_stub_writer,
    )

    api_state = json.loads(paths.api_state.read_text(encoding="utf-8"))
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    assert api_state["mlflow"]["mlflow_run_id"] == "fake-run"
    assert manifest["metadata"]["mlflow"]["traces"]["trace_ids"] == ["trace-1"]
    assert (paths.base_dir / "llm_calls" / "index.jsonl").exists()
