import json
from pathlib import Path

import pytest

from qwen_qlora_sql_benchmark.eval.execution_eval import (
    ExecutionCase,
    evaluate_execution_cases,
    load_execution_cases,
    summarize_execution_rows,
)


class MappingGenerator:
    def generate_sql(self, schema: str, question: str) -> str:
        del schema
        return {
            "List all user names": "SELECT name FROM users",
            "Delete the table": "DROP TABLE users",
        }[question]


def test_load_execution_cases_reads_jsonl(tmp_path: Path) -> None:
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

    cases = load_execution_cases(path)

    assert cases == [
        ExecutionCase(
            id="simple",
            tags=["filter"],
            schema="CREATE TABLE users (name TEXT)",
            question="List all user names",
            setup_sql=["CREATE TABLE users (name TEXT)", "INSERT INTO users VALUES ('Alice')"],
            reference_sql="SELECT name FROM users",
        )
    ]


def test_load_execution_cases_rejects_string_setup_sql(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    payload = {
        "id": "bad",
        "tags": ["filter"],
        "schema": "CREATE TABLE users (name TEXT)",
        "question": "List all user names",
        "setup_sql": "CREATE TABLE users (name TEXT)",
        "reference_sql": "SELECT name FROM users",
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="setup_sql must be a list of strings"):
        load_execution_cases(path)


def test_load_execution_cases_rejects_string_tags(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    payload = {
        "id": "bad",
        "tags": "filter",
        "schema": "CREATE TABLE users (name TEXT)",
        "question": "List all user names",
        "setup_sql": ["CREATE TABLE users (name TEXT)"],
        "reference_sql": "SELECT name FROM users",
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="tags must be a list of strings"):
        load_execution_cases(path)


def test_evaluate_execution_cases_compares_execution_results() -> None:
    rows = evaluate_execution_cases(
        cases=[
            ExecutionCase(
                id="simple",
                tags=["filter"],
                schema="CREATE TABLE users (name TEXT)",
                question="List all user names",
                setup_sql=["CREATE TABLE users (name TEXT)", "INSERT INTO users VALUES ('Alice')"],
                reference_sql="SELECT name FROM users",
            ),
            ExecutionCase(
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


def test_summarize_execution_rows_computes_rates() -> None:
    summary = summarize_execution_rows(
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
