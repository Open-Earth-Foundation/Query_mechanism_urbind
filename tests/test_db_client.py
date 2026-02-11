import sqlite3

import pytest

from backend.services.db_client import SQLiteClient, get_db_client, is_select_query


def test_is_select_query() -> None:
    assert is_select_query("select * from City")
    assert is_select_query("WITH cte AS (SELECT 1) SELECT * FROM cte")
    assert is_select_query("-- comment\nSELECT * FROM City")
    assert is_select_query("/* comment */ SELECT * FROM City")
    assert not is_select_query("delete from City")


def test_sqlite_client_query(tmp_path) -> None:
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE City (cityId TEXT, cityName TEXT)")
    conn.execute("INSERT INTO City (cityId, cityName) VALUES ('1', 'Munich')")
    conn.commit()
    conn.close()

    client = SQLiteClient(db_path)
    columns, rows = client.query("SELECT cityName FROM City")

    assert columns == ["cityName"]
    assert rows == [["Munich"]]

    with pytest.raises(ValueError):
        client.query("DELETE FROM City")


def test_get_db_client_prefers_sqlite(tmp_path) -> None:
    db_path = tmp_path / "test.db"
    db_path.write_text("", encoding="utf-8")
    client = get_db_client(db_path, None)
    assert isinstance(client, SQLiteClient)
