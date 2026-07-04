from __future__ import annotations

import sqlglot
from sqlglot import exp


def parse_valid(sql: str) -> bool:
    try:
        expressions = sqlglot.parse(sql, read="sqlite")
    except Exception:
        return False
    return bool(expressions)


def is_select_only(sql: str) -> bool:
    try:
        expressions = sqlglot.parse(sql, read="sqlite")
    except Exception:
        return False
    return len(expressions) == 1 and isinstance(expressions[0], exp.Select)


def is_allowed_setup_statement(sql: str) -> bool:
    try:
        expressions = sqlglot.parse(sql, read="sqlite")
    except Exception:
        return False
    if len(expressions) != 1:
        return False
    expression = expressions[0]
    if isinstance(expression, exp.Insert):
        return True
    return isinstance(expression, exp.Create) and expression.args.get("kind") == "TABLE"
