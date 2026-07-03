from qwen_qlora_sql_benchmark.api.app import create_app


class StaticGenerator:
    def __init__(self, sql: str) -> None:
        self.sql = sql

    def generate_sql(self, schema: str, question: str) -> str:
        assert schema == "CREATE TABLE users (name TEXT)"
        assert question == "List all user names"
        return self.sql


def test_health_returns_status() -> None:
    from fastapi.testclient import TestClient

    client = TestClient(create_app(generator=StaticGenerator("SELECT name FROM users")))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_generate_sql_returns_validation_metadata() -> None:
    from fastapi.testclient import TestClient

    client = TestClient(create_app(generator=StaticGenerator("SELECT name FROM users")))

    response = client.post(
        "/generate-sql",
        json={
            "schema": "CREATE TABLE users (name TEXT)",
            "question": "List all user names",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["sql"] == "SELECT name FROM users"
    assert payload["parse_valid"] is True
    assert payload["is_select_only"] is True
    assert payload["error"] is None
    assert payload["latency_ms"] >= 0


def test_generate_sql_flags_non_select_sql() -> None:
    from fastapi.testclient import TestClient

    client = TestClient(create_app(generator=StaticGenerator("DROP TABLE users")))

    response = client.post(
        "/generate-sql",
        json={
            "schema": "CREATE TABLE users (name TEXT)",
            "question": "List all user names",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["parse_valid"] is True
    assert payload["is_select_only"] is False
    assert payload["error"] == "generated SQL is not a read-only SELECT query"


def test_generate_sql_rejects_blank_schema() -> None:
    from fastapi.testclient import TestClient

    client = TestClient(create_app(generator=StaticGenerator("SELECT 1")))

    response = client.post(
        "/generate-sql",
        json={"schema": " ", "question": "List all user names"},
    )

    assert response.status_code == 422
