import os

import pytest

from backend.api.services.kubernetes_vector_job import (
    KubernetesJobError,
    build_vector_store_update_job,
)


def test_build_vector_store_update_job_requires_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kubernetes updater Job creation should fail fast without an image."""
    monkeypatch.delenv("VECTOR_STORE_UPDATE_JOB_IMAGE", raising=False)

    with pytest.raises(KubernetesJobError):
        build_vector_store_update_job(job_name="vector-update", trigger="startup")


def test_build_vector_store_update_job_uses_configured_image_and_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Job payload should use deployment-configured image, PVC, and memory limits."""
    monkeypatch.setenv(
        "VECTOR_STORE_UPDATE_JOB_IMAGE",
        "ghcr.io/open-earth-foundation/query_mechanism_urbind-backend:test",
    )
    monkeypatch.setenv("VECTOR_STORE_UPDATE_JOB_MEMORY_REQUEST", "6Gi")
    monkeypatch.setenv("VECTOR_STORE_UPDATE_JOB_MEMORY_LIMIT", "10Gi")
    monkeypatch.setenv("VECTOR_STORE_UPDATE_JOB_PVC", "backend-output")

    payload = build_vector_store_update_job(job_name="vector-update", trigger="run")

    container = payload["spec"]["template"]["spec"]["containers"][0]
    assert container["image"] == os.environ["VECTOR_STORE_UPDATE_JOB_IMAGE"]
    assert container["resources"]["requests"]["memory"] == "6Gi"
    assert container["resources"]["limits"]["memory"] == "10Gi"
    command = " ".join(container["command"])
    assert "--trigger run" in command
    assert "MARKDOWN_DIR" in command
    volume = payload["spec"]["template"]["spec"]["volumes"][0]
    assert volume["persistentVolumeClaim"]["claimName"] == "backend-output"


def test_build_vector_store_update_job_passes_through_runtime_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Updater Jobs should follow the backend pod's active runtime path overrides."""
    monkeypatch.setenv(
        "VECTOR_STORE_UPDATE_JOB_IMAGE",
        "ghcr.io/open-earth-foundation/query_mechanism_urbind-backend:test",
    )
    monkeypatch.setenv("MARKDOWN_DIR", "/data/vector-smoke-docs")
    monkeypatch.setenv("LLM_CONFIG_PATH", "/app/llm_config.yaml")
    monkeypatch.setenv("CHROMA_PERSIST_PATH", "/data/vector-smoke-chroma")
    monkeypatch.setenv("CHROMA_COLLECTION_NAME", "markdown_chunks")

    payload = build_vector_store_update_job(job_name="vector-update", trigger="startup")

    env_entries = payload["spec"]["template"]["spec"]["containers"][0]["env"]
    env_by_name = {entry["name"]: entry for entry in env_entries if "name" in entry}
    assert env_by_name["MARKDOWN_DIR"]["value"] == "/data/vector-smoke-docs"
    assert env_by_name["LLM_CONFIG_PATH"]["value"] == "/app/llm_config.yaml"
    assert env_by_name["CHROMA_PERSIST_PATH"]["value"] == "/data/vector-smoke-chroma"
