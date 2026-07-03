from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from typing import Any

from qwen_qlora_sql_benchmark.api.sql_validation import is_select_only


@dataclass(frozen=True)
class SqlExecutionResult:
    execution_valid: bool
    row_count: int
    rows: list[list[Any]]
    execution_error: str | None


def execute_select_query(
    sql: str,
    setup_sql: list[str],
    timeout_ms: int = 1000,
    max_rows: int = 20,
) -> SqlExecutionResult:
    if not is_select_only(sql):
        return SqlExecutionResult(
            execution_valid=False,
            row_count=0,
            rows=[],
            execution_error="only one read-only SELECT statement is allowed",
        )

    started = time.monotonic()
    connection = sqlite3.connect(":memory:")
    connection.set_progress_handler(
        lambda: 1 if (time.monotonic() - started) * 1000 > timeout_ms else 0,
        1000,
    )
    try:
        cursor = connection.cursor()
        for statement in setup_sql:
            cursor.execute(statement)
        cursor.execute(sql)
        rows = [list(row) for row in cursor.fetchmany(max_rows)]
        return SqlExecutionResult(
            execution_valid=True,
            row_count=len(rows),
            rows=rows,
            execution_error=None,
        )
    except Exception as exc:
        return SqlExecutionResult(
            execution_valid=False,
            row_count=0,
            rows=[],
            execution_error=str(exc),
        )
    finally:
        connection.close()
