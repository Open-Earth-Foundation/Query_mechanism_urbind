from pathlib import Path

import yaml


def test_backend_chroma_mount_uses_env_selectable_host_path() -> None:
    """Docker Compose should allow local host-side Chroma folder switching."""
    compose = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))
    backend = compose["services"]["backend"]

    assert backend["environment"]["CHROMA_PERSIST_PATH"] == "/data/chroma"
    assert backend["environment"]["INDEX_MANIFEST_PATH"] == "/data/chroma/index_manifest.json"
    assert "${CHROMA_HOST_PATH:-./.chroma}:/data/chroma" in backend["volumes"]
