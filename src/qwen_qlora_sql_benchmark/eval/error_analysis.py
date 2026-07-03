from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import sqlglot
from sqlglot import exp

from qwen_qlora_sql_benchmark.eval.exact_match import exact_match, normalize_sql
from qwen_qlora_sql_benchmark.eval.sql_metrics import sql_parse_valid


def classify_prediction_error(prediction: str, reference: str) -> str:
    if exact_match(prediction, reference):
        return "exact_match"

    stripped = prediction.strip()
    if not stripped:
        return "empty_output"
    if not stripped.lower().startswith("select"):
        return "non_sql_output"
    if not sql_parse_valid(prediction):
        return "invalid_sql"

    prediction_expr = _parse_select(prediction)
    reference_expr = _parse_select(reference)
    if prediction_expr is None or reference_expr is None:
        return "other_sql_mismatch"

    if _select_projection(prediction_expr) != _select_projection(reference_expr):
        return "projection_mismatch"
    if _where_clause(prediction_expr) != _where_clause(reference_expr):
        return "filter_or_condition_mismatch"
    if _from_clause(prediction_expr) != _from_clause(reference_expr):
        return "table_or_join_mismatch"

    return "other_sql_mismatch"


def summarize_prediction_errors(prediction_path: Path) -> dict[str, Any]:
    rows = _read_jsonl(prediction_path)
    counts = Counter(
        classify_prediction_error(row["prediction"], row["reference"]) for row in rows
    )
    total = len(rows)
    return {
        "prediction_path": str(prediction_path),
        "rows": total,
        "counts": dict(sorted(counts.items())),
        "rates": {
            key: value / total if total else 0.0
            for key, value in sorted(counts.items())
        },
    }


def _parse_select(sql: str) -> exp.Select | None:
    try:
        expression = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None
    if isinstance(expression, exp.Select):
        return expression
    return None


def _select_projection(expression: exp.Select) -> tuple[str, ...]:
    return tuple(normalize_sql(item.sql(dialect="sqlite")) for item in expression.expressions)


def _where_clause(expression: exp.Select) -> str:
    where = expression.args.get("where")
    if where is None:
        return ""
    return normalize_sql(where.sql(dialect="sqlite"))


def _from_clause(expression: exp.Select) -> str:
    from_clause = expression.args.get("from_")
    if from_clause is None:
        return ""
    joins = expression.args.get("joins") or []
    parts = [from_clause.sql(dialect="sqlite")]
    parts.extend(join.sql(dialect="sqlite") for join in joins)
    return normalize_sql(" ".join(parts))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
