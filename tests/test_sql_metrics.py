from qwen_qlora_sql_benchmark.eval.sql_metrics import (
    compute_prediction_metrics,
    sql_parse_valid,
)


def test_sql_parse_valid_accepts_select_query() -> None:
    assert sql_parse_valid("SELECT id FROM users WHERE age > 18")


def test_sql_parse_valid_rejects_incomplete_query() -> None:
    assert not sql_parse_valid("SELECT FROM WHERE")


def test_compute_prediction_metrics_includes_exact_and_validity() -> None:
    metrics = compute_prediction_metrics("SELECT id FROM users;", "select id from users")

    assert metrics["exact_match"] is True
    assert metrics["sql_parse_valid"] is True
    assert metrics["starts_select"] is True
