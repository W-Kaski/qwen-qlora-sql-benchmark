from __future__ import annotations

import time
from typing import Annotated

from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

from qwen_qlora_sql_benchmark.api.generator import LazyPeftSqlGenerator, SqlGenerator
from qwen_qlora_sql_benchmark.api.sql_validation import is_select_only, parse_valid

MAX_SCHEMA_CHARS = 12000
MAX_QUESTION_CHARS = 1000


class GenerateSqlRequest(BaseModel):
    sql_schema: Annotated[str, Field(alias="schema", min_length=1, max_length=MAX_SCHEMA_CHARS)]
    question: Annotated[str, Field(min_length=1, max_length=MAX_QUESTION_CHARS)]

    @field_validator("sql_schema", "question")
    @classmethod
    def reject_blank_strings(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be blank")
        return stripped


class GenerateSqlResponse(BaseModel):
    sql: str | None
    parse_valid: bool
    is_select_only: bool
    latency_ms: float
    error: str | None


def create_app(generator: SqlGenerator | None = None) -> FastAPI:
    app = FastAPI(title="Qwen QLoRA Text-to-SQL API", version="0.1.0")
    sql_generator = generator or LazyPeftSqlGenerator()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/generate-sql", response_model=GenerateSqlResponse)
    def generate_sql(request: GenerateSqlRequest) -> GenerateSqlResponse:
        started = time.perf_counter()
        try:
            sql = sql_generator.generate_sql(request.sql_schema, request.question).strip()
        except Exception as exc:
            return GenerateSqlResponse(
                sql=None,
                parse_valid=False,
                is_select_only=False,
                latency_ms=_elapsed_ms(started),
                error=f"generation failed: {exc}",
            )

        valid = parse_valid(sql)
        select_only = is_select_only(sql)
        error = None
        if not valid:
            error = "generated SQL is not parseable"
        elif not select_only:
            error = "generated SQL is not a read-only SELECT query"

        return GenerateSqlResponse(
            sql=sql,
            parse_valid=valid,
            is_select_only=select_only,
            latency_ms=_elapsed_ms(started),
            error=error,
        )

    return app


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)


app = create_app()
