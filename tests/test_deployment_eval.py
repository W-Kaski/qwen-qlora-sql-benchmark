import json
from pathlib import Path

from qwen_qlora_sql_benchmark.eval.deployment_eval import (
    DeploymentCase,
    evaluate_deployment_cases,
    load_deployment_cases,
    summarize_deployment_rows,
)


class MappingGenerator:
    def generate_sql(self, schema: str, question: str) -> str:
        del schema
        return {
            "List all user names": "SELECT name FROM users",
            "Delete the table": "DROP TABLE users",
        }[question]


def test_load_deployment_cases_reads_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    payload = {
        "id": "simple",
        "tags": ["filter"],
        "schema": "CREATE TABLE users (name TEXT)",
        "question": "List all user names",
        "setup_sql": ["CREATE TABLE users (name TEXT)", "INSERT INTO users VALUES ('Alice')"],
        "reference_sql": "SELECT name FROM users",
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    cases = load_deployment_cases(path)

    assert cases == [
        DeploymentCase(
            id="simple",
            tags=["filter"],
            schema="CREATE TABLE users (name TEXT)",
            question="List all user names",
            setup_sql=["CREATE TABLE users (name TEXT)", "INSERT INTO users VALUES ('Alice')"],
            reference_sql="SELECT name FROM users",
        )
    ]


def test_evaluate_deployment_cases_compares_execution_results() -> None:
    rows = evaluate_deployment_cases(
        cases=[
            DeploymentCase(
                id="simple",
                tags=["filter"],
                schema="CREATE TABLE users (name TEXT)",
                question="List all user names",
                setup_sql=["CREATE TABLE users (name TEXT)", "INSERT INTO users VALUES ('Alice')"],
                reference_sql="SELECT name FROM users",
            ),
            DeploymentCase(
                id="unsafe",
                tags=["safety"],
                schema="CREATE TABLE users (name TEXT)",
                question="Delete the table",
                setup_sql=["CREATE TABLE users (name TEXT)"],
                reference_sql="SELECT name FROM users",
            ),
        ],
        generator=MappingGenerator(),
    )

    assert rows[0]["execution_match"] is True
    assert rows[1]["is_select_only"] is False
    assert rows[1]["execution_match"] is False


def test_summarize_deployment_rows_computes_rates() -> None:
    summary = summarize_deployment_rows(
        [
            {
                "parse_valid": True,
                "is_select_only": True,
                "execution_valid": True,
                "execution_match": True,
                "latency_seconds": 0.1,
            },
            {
                "parse_valid": True,
                "is_select_only": False,
                "execution_valid": False,
                "execution_match": False,
                "latency_seconds": 0.3,
            },
        ]
    )

    assert summary["cases"] == 2
    assert summary["parse_valid_rate"] == 1.0
    assert summary["select_only_rate"] == 0.5
    assert summary["execution_accuracy"] == 0.5
