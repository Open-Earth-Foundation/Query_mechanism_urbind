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


def test_city_markdown_endpoint_returns_city_content(tmp_path: Path) -> None:
    runs_dir = tmp_path / "output"
    markdown_dir = tmp_path / "documents"
    markdown_dir.mkdir(parents=True, exist_ok=True)
    source_path = markdown_dir / "Munich.md"
    source_path.write_text("# Munich\n\nClimate content", encoding="utf-8")

    app = create_app(runs_dir=runs_dir, max_workers=1, markdown_dir=markdown_dir)
    with TestClient(app) as client:
        response = client.get("/api/v1/cities/Munich/markdown")
        assert response.status_code == 200
        payload = response.json()
        assert payload == {
            "city_name": "Munich",
            "content": "# Munich\n\nClimate content",
            "source_paths": [str(source_path)],
        }


def test_city_markdown_endpoint_matches_normalized_city_names(tmp_path: Path) -> None:
    runs_dir = tmp_path / "output"
    markdown_dir = tmp_path / "documents"
    markdown_dir.mkdir(parents=True, exist_ok=True)
    (markdown_dir / "Vitoria_Gasteiz.md").write_text(
        "# Vitoria Gasteiz\n\nContent",
        encoding="utf-8",
    )

    app = create_app(runs_dir=runs_dir, max_workers=1, markdown_dir=markdown_dir)
    with TestClient(app) as client:
        response = client.get("/api/v1/cities/Vitoria-Gasteiz/markdown")
        assert response.status_code == 200
        payload = response.json()
        assert payload["city_name"] == "Vitoria_Gasteiz"
        assert payload["content"] == "# Vitoria Gasteiz\n\nContent"


def test_city_markdown_endpoint_concatenates_duplicate_city_files_in_path_order(
    tmp_path: Path,
) -> None:
    runs_dir = tmp_path / "output"
    markdown_dir = tmp_path / "documents"
    annex_dir = markdown_dir / "Annex"
    annex_dir.mkdir(parents=True, exist_ok=True)
    primary_path = markdown_dir / "Munich.md"
    annex_path = annex_dir / "Munich.md"
    primary_path.write_text("# Primary\n\nOne", encoding="utf-8")
    annex_path.write_text("# Annex\n\nTwo", encoding="utf-8")

    app = create_app(runs_dir=runs_dir, max_workers=1, markdown_dir=markdown_dir)
    with TestClient(app) as client:
        response = client.get("/api/v1/cities/Munich/markdown")
        assert response.status_code == 200
        payload = response.json()
        assert payload["content"] == "# Primary\n\nOne\n\n# Annex\n\nTwo"
        assert payload["source_paths"] == [str(primary_path), str(annex_path)]


def test_city_markdown_endpoint_returns_404_for_missing_city(tmp_path: Path) -> None:
    runs_dir = tmp_path / "output"
    markdown_dir = tmp_path / "documents"
    markdown_dir.mkdir(parents=True, exist_ok=True)
    (markdown_dir / "Munich.md").write_text("# Munich", encoding="utf-8")

    app = create_app(runs_dir=runs_dir, max_workers=1, markdown_dir=markdown_dir)
    with TestClient(app) as client:
        response = client.get("/api/v1/cities/Berlin/markdown")
        assert response.status_code == 404
        assert response.json()["detail"] == "City markdown for `Berlin` was not found."


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


def test_default_city_groups_catalog_exposes_all_12_regional_groups(tmp_path: Path) -> None:
    runs_dir = tmp_path / "output"
    markdown_dir = tmp_path / "documents"
    markdown_dir.mkdir(parents=True, exist_ok=True)

    seed_cities = [
        "Aachen",
        "Amsterdam",
        "Dublin",
        "Paris",
        "Lisbon",
        "Rome",
        "Stockholm",
        "Helsinki",
        "Riga",
        "Krakow",
        "Bratislava",
        "Athens",
    ]
    for city in seed_cities:
        (markdown_dir / f"{city}.md").write_text(f"# {city}", encoding="utf-8")

    app = create_app(
        runs_dir=runs_dir,
        max_workers=1,
        markdown_dir=markdown_dir,
    )
    with TestClient(app) as client:
        response = client.get("/api/v1/city-groups")
        assert response.status_code == 200
        payload = response.json()
        assert payload["total"] == 12
        ids = {group["id"] for group in payload["groups"]}
        assert ids == {
            "dach-germany",
            "benelux-lux",
            "uk-ireland",
            "france-core",
            "iberia",
            "italy-core",
            "scandinavia",
            "finland-iceland",
            "baltics",
            "poland",
            "central-europe",
            "balkans-east-med",
        }
