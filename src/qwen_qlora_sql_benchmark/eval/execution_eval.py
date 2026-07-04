from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from qwen_qlora_sql_benchmark.api.generator import LazyPeftSqlGenerator, SqlGenerator
from qwen_qlora_sql_benchmark.api.sql_validation import is_select_only, parse_valid
from qwen_qlora_sql_benchmark.api.sqlite_sandbox import execute_select_query
from qwen_qlora_sql_benchmark.benchmark.serving_benchmark import percentile


@dataclass(frozen=True)
class ExecutionCase:
    id: str
    tags: list[str]
    schema: str
    question: str
    setup_sql: list[str]
    reference_sql: str


def load_execution_cases(path: Path) -> list[ExecutionCase]:
    cases: list[ExecutionCase] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            cases.append(
                ExecutionCase(
                    id=_require_string(payload, "id", path, line_number),
                    tags=_require_string_list(payload, "tags", path, line_number),
                    schema=_require_string(payload, "schema", path, line_number),
                    question=_require_string(payload, "question", path, line_number),
                    setup_sql=_require_string_list(payload, "setup_sql", path, line_number),
                    reference_sql=_require_string(payload, "reference_sql", path, line_number),
                )
            )
    return cases


def evaluate_execution_cases(
    cases: list[ExecutionCase],
    generator: SqlGenerator,
    timeout_ms: int = 1000,
    max_rows: int = 20,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        started = time.perf_counter()
        prediction = generator.generate_sql(case.schema, case.question).strip()
        latency_seconds = time.perf_counter() - started
        reference_result = execute_select_query(
            sql=case.reference_sql,
            setup_sql=case.setup_sql,
            timeout_ms=timeout_ms,
            max_rows=max_rows,
        )
        prediction_result = execute_select_query(
            sql=prediction,
            setup_sql=case.setup_sql,
            timeout_ms=timeout_ms,
            max_rows=max_rows,
        )
        execution_match = (
            reference_result.execution_valid
            and prediction_result.execution_valid
            and prediction_result.rows == reference_result.rows
        )
        rows.append(
            {
                "id": case.id,
                "tags": "|".join(case.tags),
                "parse_valid": parse_valid(prediction),
                "is_select_only": is_select_only(prediction),
                "execution_valid": prediction_result.execution_valid,
                "execution_match": execution_match,
                "latency_seconds": latency_seconds,
                "execution_error": prediction_result.execution_error,
                "prediction": prediction,
                "reference_sql": case.reference_sql,
            }
        )
    return rows


def summarize_execution_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(row["latency_seconds"]) for row in rows]
    total = len(rows)
    return {
        "cases": total,
        "parse_valid_rate": _mean_bool(rows, "parse_valid"),
        "select_only_rate": _mean_bool(rows, "is_select_only"),
        "execution_valid_rate": _mean_bool(rows, "execution_valid"),
        "execution_accuracy": _mean_bool(rows, "execution_match"),
        "p50_latency_seconds": percentile(latencies, 0.50) if latencies else 0.0,
        "p95_latency_seconds": percentile(latencies, 0.95) if latencies else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SQLite-backed execution evaluation.")
    parser.add_argument("--cases", type=Path, default=Path("data/execution_eval/cases.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/tables/execution_eval.csv"))
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("results/tables/execution_eval_summary.csv"),
    )
    parser.add_argument("--adapter-path", type=Path, default=Path("outputs/adapters/lora_r32"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_execution_cases(args.cases)
    generator = LazyPeftSqlGenerator(adapter_path=args.adapter_path)
    rows = evaluate_execution_cases(cases, generator)
    summary = summarize_execution_rows(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    pd.DataFrame([summary]).to_csv(args.summary_output, index=False)
    print(pd.DataFrame([summary]).to_csv(index=False))


def _require_string(payload: dict[str, Any], key: str, path: Path, line_number: int) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}:{line_number}.{key} must be a non-empty string")
    return value.strip()


def _require_string_list(
    payload: dict[str, Any], key: str, path: Path, line_number: int
) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{path}:{line_number}.{key} must be a list of strings")
    strings: list[str] = []
    for item_number, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                f"{path}:{line_number}.{key}[{item_number}] must be a non-empty string"
            )
        strings.append(item.strip())
    return strings


def _mean_bool(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(bool(row[key]) for row in rows) / len(rows)


if __name__ == "__main__":
    main()
