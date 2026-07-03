import json

from qwen_qlora_sql_benchmark.eval.error_analysis import (
    classify_prediction_error,
    summarize_prediction_errors,
)


def test_classify_prediction_error_marks_exact_matches() -> None:
    error_type = classify_prediction_error(
        prediction="SELECT name FROM users;",
        reference="select name from users",
    )

    assert error_type == "exact_match"


def test_classify_prediction_error_detects_non_sql_output() -> None:
    error_type = classify_prediction_error(
        prediction="I cannot answer this question.",
        reference="SELECT name FROM users",
    )

    assert error_type == "non_sql_output"


def test_classify_prediction_error_detects_invalid_sql() -> None:
    error_type = classify_prediction_error(
        prediction="SELECT FROM WHERE",
        reference="SELECT name FROM users",
    )

    assert error_type == "invalid_sql"


def test_classify_prediction_error_detects_projection_mismatch() -> None:
    error_type = classify_prediction_error(
        prediction="SELECT age FROM users WHERE country = 'ca'",
        reference="SELECT name FROM users WHERE country = 'ca'",
    )

    assert error_type == "projection_mismatch"


def test_classify_prediction_error_detects_filter_mismatch() -> None:
    error_type = classify_prediction_error(
        prediction="SELECT name FROM users WHERE country = 'us'",
        reference="SELECT name FROM users WHERE country = 'ca'",
    )

    assert error_type == "filter_or_condition_mismatch"


def test_classify_prediction_error_detects_table_mismatch() -> None:
    error_type = classify_prediction_error(
        prediction="SELECT name FROM customers WHERE country = 'ca'",
        reference="SELECT name FROM users WHERE country = 'ca'",
    )

    assert error_type == "table_or_join_mismatch"


def test_summarize_prediction_errors_counts_rows(tmp_path) -> None:
    prediction_path = tmp_path / "predictions.jsonl"
    rows = [
        {
            "prediction": "SELECT name FROM users",
            "reference": "SELECT name FROM users",
        },
        {
            "prediction": "SELECT age FROM users",
            "reference": "SELECT name FROM users",
        },
    ]
    prediction_path.write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )

    summary = summarize_prediction_errors(prediction_path)

    assert summary["rows"] == 2
    assert summary["counts"]["exact_match"] == 1
    assert summary["counts"]["projection_mismatch"] == 1
