from pathlib import Path

import yaml


def _container_env_by_name(manifest_path: str, container_name: str) -> dict[str, dict]:
    """Return one Kubernetes container env list keyed by variable name."""
    manifest = yaml.safe_load(Path(manifest_path).read_text(encoding="utf-8"))
    containers = manifest["spec"]["template"]["spec"]["containers"]
    container = next(item for item in containers if item["name"] == container_name)
    return {entry["name"]: entry for entry in container["env"]}


def test_backend_kubernetes_manifests_enable_mlflow_for_production() -> None:
    """Production backend pods should receive the same MLflow controls as local env."""
    configmap = yaml.safe_load(
        Path("k8s/backend-configmap.yml").read_text(encoding="utf-8")
    )

    assert configmap["data"]["MLFLOW_ENABLED"] == "true"
    assert configmap["data"]["MLFLOW_EXPERIMENT_NAME"] == "URBIND"
    assert configmap["data"]["MLFLOW_ENVIRONMENT"] == "production"
    assert configmap["data"]["MLFLOW_ARTIFACT_PATH"] == "run_artifacts"
    assert configmap["data"]["MLFLOW_TRACE_MODE"] == "consolidated"
    assert configmap["data"]["MLFLOW_FAIL_ON_ERROR"] == "false"

    backend_env = _container_env_by_name("k8s/backend-deployment.yml", "backend")
    for variable in (
        "MLFLOW_TRACKING_URI",
        "MLFLOW_TRACKING_USERNAME",
        "MLFLOW_TRACKING_PASSWORD",
    ):
        mlflow_secret = backend_env[variable]["valueFrom"]["secretKeyRef"]
        assert mlflow_secret["name"] == "urbind-query-mechanism-backend-secrets"
        assert mlflow_secret["key"] == variable


def test_vector_index_job_receives_mlflow_secrets() -> None:
    """The maintenance Job should use the same backend secret contract."""
    job_env = _container_env_by_name(
        "k8s/backend-build-vector-index-job.yml",
        "build-index",
    )

    for variable in (
        "MLFLOW_TRACKING_URI",
        "MLFLOW_TRACKING_USERNAME",
        "MLFLOW_TRACKING_PASSWORD",
    ):
        mlflow_secret = job_env[variable]["valueFrom"]["secretKeyRef"]
        assert mlflow_secret["name"] == "urbind-query-mechanism-backend-secrets"
        assert mlflow_secret["key"] == variable


def test_deploy_workflow_creates_mlflow_secrets() -> None:
    """The deploy workflow should copy GitHub's MLflow secrets into Kubernetes."""
    workflow = yaml.safe_load(
        Path(".github/workflows/develop.yml").read_text(encoding="utf-8")
    )
    steps = workflow["jobs"]["deploy"]["steps"]
    backend_secret_step = next(
        step for step in steps if step["name"] == "Create or update backend secret"
    )

    for variable in (
        "MLFLOW_TRACKING_URI",
        "MLFLOW_TRACKING_USERNAME",
        "MLFLOW_TRACKING_PASSWORD",
    ):
        assert backend_secret_step["env"][variable] == (
            f"${{{{ secrets.{variable} }}}}"
        )
        assert f"Set GitHub secret {variable}" in backend_secret_step["run"]
        assert f"--from-literal={variable}=" in backend_secret_step["run"]
