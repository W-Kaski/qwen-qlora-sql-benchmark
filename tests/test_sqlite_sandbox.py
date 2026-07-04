from qwen_qlora_sql_benchmark.api.sqlite_sandbox import execute_select_query


def test_execute_select_query_returns_limited_rows() -> None:
    result = execute_select_query(
        sql="SELECT name FROM users ORDER BY name",
        setup_sql=[
            "CREATE TABLE users (name TEXT)",
            "INSERT INTO users VALUES ('Alice'), ('Bob'), ('Chao')",
        ],
        max_rows=2,
    )

    assert result.execution_valid is True
    assert result.row_count == 2
    assert result.rows == [["Alice"], ["Bob"]]
    assert result.execution_error is None


def test_execute_select_query_rejects_non_select_query() -> None:
    result = execute_select_query(
        sql="DELETE FROM users",
        setup_sql=["CREATE TABLE users (name TEXT)"],
    )

    assert result.execution_valid is False
    assert result.execution_error == "only one read-only SELECT statement is allowed"


def test_execute_select_query_rejects_multi_statement_query() -> None:
    result = execute_select_query(
        sql="SELECT name FROM users; SELECT name FROM users",
        setup_sql=["CREATE TABLE users (name TEXT)"],
    )

    assert result.execution_valid is False
    assert result.execution_error == "only one read-only SELECT statement is allowed"


def test_execute_select_query_rejects_unsafe_setup_sql() -> None:
    result = execute_select_query(
        sql="SELECT name FROM users",
        setup_sql=["CREATE TABLE users (name TEXT)", "DROP TABLE users"],
    )

    assert result.execution_valid is False
    assert result.execution_error == (
        "setup_sql allows only single CREATE TABLE or INSERT statements"
    )


def test_execute_select_query_rejects_multi_statement_setup_sql() -> None:
    result = execute_select_query(
        sql="SELECT name FROM users",
        setup_sql=["CREATE TABLE users (name TEXT); INSERT INTO users VALUES ('Alice')"],
    )

    assert result.execution_valid is False
    assert result.execution_error == (
        "setup_sql allows only single CREATE TABLE or INSERT statements"
    )


def test_execute_select_query_reports_sqlite_errors() -> None:
    result = execute_select_query(sql="SELECT missing FROM users", setup_sql=[])

    assert result.execution_valid is False
    assert "no such table" in result.execution_error
