from __future__ import annotations

from qwen_qlora_sql_benchmark.eval.exact_match import exact_match


def sql_parse_valid(sql: str) -> bool:
    import sqlglot

    try:
        expression = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return False
    return expression is not None and expression.key == "select"


def compute_prediction_metrics(prediction: str, reference: str) -> dict[str, bool]:
    stripped = prediction.strip()
    return {
        "exact_match": exact_match(prediction, reference),
        "sql_parse_valid": sql_parse_valid(prediction),
        "starts_select": stripped.lower().startswith("select"),
        "empty": not stripped,
    }
