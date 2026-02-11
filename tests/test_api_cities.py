import json
from pathlib import Path

from fastapi.testclient import TestClient

from backend.api.main import create_app


def test_cities_endpoint_lists_unique_markdown_stems(tmp_path: Path) -> None:
    runs_dir = tmp_path / "output"
    markdown_dir = tmp_path / "documents"
    nested = markdown_dir / "Germany"
    nested.mkdir(parents=True, exist_ok=True)

    (markdown_dir / "Munich.md").write_text("# Munich", encoding="utf-8")
    (markdown_dir / "Berlin.md").write_text("# Berlin", encoding="utf-8")
    (nested / "Munich.md").write_text("# Munich duplicate", encoding="utf-8")
    (markdown_dir / "README.txt").write_text("ignore", encoding="utf-8")

    app = create_app(runs_dir=runs_dir, max_workers=1, markdown_dir=markdown_dir)
    with TestClient(app) as client:
        response = client.get("/api/v1/cities")
        assert response.status_code == 200
        payload = response.json()
        assert payload["cities"] == ["Berlin", "Munich"]
        assert payload["total"] == 2
        assert payload["markdown_dir"] == str(markdown_dir)


def test_cities_endpoint_returns_empty_for_missing_dir(tmp_path: Path) -> None:
    runs_dir = tmp_path / "output"
    missing_markdown_dir = tmp_path / "missing-documents"

    app = create_app(
        runs_dir=runs_dir,
        max_workers=1,
        markdown_dir=missing_markdown_dir,
    )
    with TestClient(app) as client:
        response = client.get("/api/v1/cities")
        assert response.status_code == 200
        payload = response.json()
        assert payload["cities"] == []
        assert payload["total"] == 0
        assert payload["markdown_dir"] == str(missing_markdown_dir)


def test_city_groups_endpoint_filters_to_available_cities(tmp_path: Path) -> None:
    runs_dir = tmp_path / "output"
    markdown_dir = tmp_path / "documents"
    markdown_dir.mkdir(parents=True, exist_ok=True)
    (markdown_dir / "Munich.md").write_text("# Munich", encoding="utf-8")
    (markdown_dir / "Berlin.md").write_text("# Berlin", encoding="utf-8")

    groups_path = tmp_path / "city_groups.json"
    groups_path.write_text(
        json.dumps(
            {
                "groups": [
                    {
                        "id": "core",
                        "name": "Core",
                        "description": "Core cities",
                        "cities": ["Munich", "Berlin", "MissingCity"],
                    },
                    {
                        "id": "invalid",
                        "name": "Invalid",
                        "cities": ["MissingCityOnly"],
                    },
                ]
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    app = create_app(
        runs_dir=runs_dir,
        max_workers=1,
        markdown_dir=markdown_dir,
        city_groups_path=groups_path,
    )
    with TestClient(app) as client:
        response = client.get("/api/v1/city-groups")
        assert response.status_code == 200
        payload = response.json()
        assert payload["total"] == 1
        assert payload["groups_path"] == str(groups_path)
        assert payload["groups"][0]["id"] == "core"
        assert payload["groups"][0]["cities"] == ["Munich", "Berlin"]


def test_city_groups_endpoint_returns_empty_when_file_missing(tmp_path: Path) -> None:
    runs_dir = tmp_path / "output"
    markdown_dir = tmp_path / "documents"
    markdown_dir.mkdir(parents=True, exist_ok=True)
    (markdown_dir / "Munich.md").write_text("# Munich", encoding="utf-8")

    missing_groups = tmp_path / "missing-city-groups.json"
    app = create_app(
        runs_dir=runs_dir,
        max_workers=1,
        markdown_dir=markdown_dir,
        city_groups_path=missing_groups,
    )
    with TestClient(app) as client:
        response = client.get("/api/v1/city-groups")
        assert response.status_code == 200
        payload = response.json()
        assert payload["total"] == 0
        assert payload["groups"] == []
        assert payload["groups_path"] == str(missing_groups)
