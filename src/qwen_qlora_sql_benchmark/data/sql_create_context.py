from __future__ import annotations

from typing import Any

from qwen_qlora_sql_benchmark.data.prompt_completion import PromptCompletionRecord


def build_text_to_sql_prompt(context: str, question: str) -> str:
    return "\n".join(
        [
            "You are a Text-to-SQL assistant.",
            "Use the SQL schema to answer the question with one SQL query.",
            "",
            "Schema:",
            context.strip(),
            "",
            "Question:",
            question.strip(),
            "",
            "SQL:",
        ]
    )


def convert_sql_create_context_record(record: dict[str, Any]) -> PromptCompletionRecord:
    for key in ("answer", "question", "context"):
        if key not in record:
            raise KeyError(key)
        if not isinstance(record[key], str):
            raise TypeError(f"{key} must be a string")
        if not record[key].strip():
            raise ValueError(f"{key} must not be blank")

    return PromptCompletionRecord(
        prompt=build_text_to_sql_prompt(context=record["context"], question=record["question"]),
        completion=record["answer"].strip(),
    )
