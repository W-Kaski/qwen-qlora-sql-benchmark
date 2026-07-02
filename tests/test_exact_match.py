from qwen_qlora_sql_benchmark.eval.exact_match import exact_match, normalize_sql


def test_normalize_sql_removes_extra_whitespace_and_trailing_semicolon() -> None:
    assert normalize_sql(" SELECT   id\nFROM users; ") == "select id from users"


def test_exact_match_is_case_and_whitespace_insensitive() -> None:
    assert exact_match("SELECT id FROM users;", "select   id from users")


def test_exact_match_detects_different_sql() -> None:
    assert not exact_match("SELECT id FROM users;", "SELECT name FROM users;")
