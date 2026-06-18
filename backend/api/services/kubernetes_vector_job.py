"""Create one-off Kubernetes Jobs for vector-store updates."""

from __future__ import annotations

import json
import logging
import os
import ssl
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

SERVICE_ACCOUNT_DIR = Path("/var/run/secrets/kubernetes.io/serviceaccount")


class KubernetesJobError(RuntimeError):
    """Raised when the backend cannot create the vector-store updater Job."""


def _service_account_namespace() -> str:
    """Return the current pod namespace from service-account files or env."""
    env_namespace = os.getenv("VECTOR_STORE_UPDATE_JOB_NAMESPACE", "").strip()
    if env_namespace:
        return env_namespace
    namespace_path = SERVICE_ACCOUNT_DIR / "namespace"
    if namespace_path.exists():
        return namespace_path.read_text(encoding="utf-8").strip()
    return "default"


def _service_account_token() -> str:
    """Read the mounted Kubernetes service-account token."""
    token_path = SERVICE_ACCOUNT_DIR / "token"
    if not token_path.exists():
        raise KubernetesJobError("Kubernetes service-account token is not mounted.")
    return token_path.read_text(encoding="utf-8").strip()


def _ssl_context() -> ssl.SSLContext:
    """Return an SSL context using the service-account CA when available."""
    ca_path = SERVICE_ACCOUNT_DIR / "ca.crt"
    if ca_path.exists():
        return ssl.create_default_context(cafile=str(ca_path))
    return ssl.create_default_context()


def _env(name: str, default: str) -> str:
    """Return a stripped environment value with a default."""
    return os.getenv(name, default).strip() or default


def _optional_env_entry(name: str) -> dict[str, str] | None:
    """Return one plain env entry when the current backend pod defines it."""
    value = os.getenv(name, "").strip()
    if not value:
        return None
    return {"name": name, "value": value}


def _job_name(trigger: str) -> str:
    """Build a unique Kubernetes Job name for one vector update."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    safe_trigger = "".join(char for char in trigger.lower() if char.isalnum() or char == "-")
    return f"urbind-vector-store-update-{safe_trigger or 'manual'}-{timestamp}"


def build_vector_store_update_job(*, job_name: str, trigger: str) -> dict[str, object]:
    """Build the Kubernetes Job payload for a vector-store update."""
    image = os.getenv("VECTOR_STORE_UPDATE_JOB_IMAGE", "").strip()
    if not image:
        raise KubernetesJobError("VECTOR_STORE_UPDATE_JOB_IMAGE is required.")

    pvc_name = _env(
        "VECTOR_STORE_UPDATE_JOB_PVC",
        "urbind-query-mechanism-backend-output",
    )
    config_map_name = _env(
        "VECTOR_STORE_UPDATE_JOB_CONFIGMAP",
        "urbind-query-mechanism-backend-config",
    )
    secret_name = _env(
        "VECTOR_STORE_UPDATE_JOB_SECRET",
        "urbind-query-mechanism-backend-secrets",
    )
    cpu_request = _env("VECTOR_STORE_UPDATE_JOB_CPU_REQUEST", "250m")
    cpu_limit = _env("VECTOR_STORE_UPDATE_JOB_CPU_LIMIT", "500m")
    memory_request = _env("VECTOR_STORE_UPDATE_JOB_MEMORY_REQUEST", "16Gi")
    memory_limit = _env("VECTOR_STORE_UPDATE_JOB_MEMORY_LIMIT", "20Gi")
    passthrough_env = [
        entry
        for entry in (
            _optional_env_entry("MARKDOWN_DIR"),
            _optional_env_entry("LLM_CONFIG_PATH"),
            _optional_env_entry("CHROMA_PERSIST_PATH"),
            _optional_env_entry("CHROMA_COLLECTION_NAME"),
            _optional_env_entry("OPENROUTER_BASE_URL"),
            _optional_env_entry("ANONYMIZED_TELEMETRY"),
        )
        if entry is not None
    ]

    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name,
            "labels": {
                "app": "urbind-query-mechanism",
                "component": "vector-store-updater",
                "trigger": trigger,
            },
        },
        "spec": {
            "ttlSecondsAfterFinished": 86400,
            "backoffLimit": 3,
            "activeDeadlineSeconds": 3600,
            "template": {
                "metadata": {
                    "labels": {
                        "app": "urbind-query-mechanism",
                        "component": "vector-store-updater",
                    },
                    "annotations": {
                        "karpenter.sh/do-not-disrupt": "true",
                    },
                },
                "spec": {
                    "restartPolicy": "Never",
                    "automountServiceAccountToken": False,
                    "containers": [
                        {
                            "name": "update-vector-store",
                            "image": image,
                            "imagePullPolicy": "Always",
                            "securityContext": {
                                "runAsUser": 0,
                                "runAsGroup": 0,
                                "allowPrivilegeEscalation": False,
                                "capabilities": {"add": ["DAC_READ_SEARCH"]},
                            },
                            "envFrom": [
                                {"configMapRef": {"name": config_map_name}},
                            ],
                            "env": [
                                {"name": "PYTHONPATH", "value": "/app"},
                                {
                                    "name": "OPENROUTER_API_KEY",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": secret_name,
                                            "key": "OPENROUTER_API_KEY",
                                        }
                                    },
                                },
                            ]
                            + passthrough_env,
                            "workingDir": "/app",
                            "command": [
                                "sh",
                                "-c",
                                (
                                    "python -m backend.scripts.update_vector_store "
                                    f"--trigger {trigger} "
                                    '--docs-dir "${MARKDOWN_DIR:-/app/documents}" '
                                    '--config "${LLM_CONFIG_PATH:-/app/llm_config.yaml}"'
                                ),
                            ],
                            "volumeMounts": [
                                {"name": "output-volume", "mountPath": "/data"},
                            ],
                            "resources": {
                                "requests": {
                                    "cpu": cpu_request,
                                    "memory": memory_request,
                                },
                                "limits": {
                                    "cpu": cpu_limit,
                                    "memory": memory_limit,
                                },
                            },
                        }
                    ],
                    "volumes": [
                        {
                            "name": "output-volume",
                            "persistentVolumeClaim": {"claimName": pvc_name},
                        }
                    ],
                },
            },
        },
    }


def create_vector_store_update_job(*, trigger: str) -> str:
    """Create a Kubernetes updater Job and return its name."""
    namespace = _service_account_namespace()
    job_name = _job_name(trigger)
    payload = build_vector_store_update_job(job_name=job_name, trigger=trigger)
    api_host = _env("KUBERNETES_SERVICE_HOST", "kubernetes.default.svc")
    api_port = _env("KUBERNETES_SERVICE_PORT", "443")
    url = f"https://{api_host}:{api_port}/apis/batch/v1/namespaces/{namespace}/jobs"
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {_service_account_token()}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, context=_ssl_context(), timeout=10) as response:
            response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise KubernetesJobError(
            f"Kubernetes updater Job creation failed ({exc.code}): {detail}"
        ) from exc
    except OSError as exc:
        raise KubernetesJobError(f"Kubernetes updater Job creation failed: {exc}") from exc

    logger.info(
        "Created vector-store updater Job namespace=%s job_name=%s trigger=%s",
        namespace,
        job_name,
        trigger,
    )
    return job_name


__all__ = [
    "KubernetesJobError",
    "build_vector_store_update_job",
    "create_vector_store_update_job",
]
