from qwen_qlora_sql_benchmark.api.sql_validation import is_allowed_setup_statement
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


def test_is_allowed_setup_statement_accepts_create_table_and_insert() -> None:
    assert is_allowed_setup_statement("CREATE TABLE users (name TEXT)") is True
    assert is_allowed_setup_statement("INSERT INTO users VALUES ('Alice')") is True


def test_is_allowed_setup_statement_rejects_other_sql() -> None:
    assert is_allowed_setup_statement("DROP TABLE users") is False
    assert is_allowed_setup_statement("CREATE INDEX idx_users_name ON users(name)") is False
