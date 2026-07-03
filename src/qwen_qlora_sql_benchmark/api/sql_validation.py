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
