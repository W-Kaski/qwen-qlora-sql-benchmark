import pytest

from qwen_qlora_sql_benchmark.data.prompt_completion import (
    PromptCompletionRecord,
    validate_prompt_completion_record,
)


def test_validate_prompt_completion_record_accepts_prompt_and_completion() -> None:
    record = validate_prompt_completion_record(
        {
            "prompt": "Schema: table users(id)\nQuestion: list ids",
            "completion": "SELECT id FROM users;",
        }
    )

    assert record == PromptCompletionRecord(
        prompt="Schema: table users(id)\nQuestion: list ids",
        completion="SELECT id FROM users;",
    )


def test_validate_prompt_completion_record_rejects_blank_completion() -> None:
    with pytest.raises(ValueError, match="completion"):
        validate_prompt_completion_record({"prompt": "Question", "completion": "   "})


def test_validate_prompt_completion_record_rejects_missing_prompt() -> None:
    with pytest.raises(KeyError, match="prompt"):
        validate_prompt_completion_record({"completion": "SELECT 1;"})
