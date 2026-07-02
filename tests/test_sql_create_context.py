from qwen_qlora_sql_benchmark.data.sql_create_context import (
    build_text_to_sql_prompt,
    convert_sql_create_context_record,
)


def test_build_text_to_sql_prompt_includes_context_and_question() -> None:
    prompt = build_text_to_sql_prompt(
        context="CREATE TABLE head (age INTEGER)",
        question="How many heads of the departments are older than 56 ?",
    )

    assert "CREATE TABLE head (age INTEGER)" in prompt
    assert "How many heads of the departments are older than 56 ?" in prompt
    assert prompt.endswith("SQL:")


def test_convert_sql_create_context_record_uses_exact_dataset_fields() -> None:
    record = convert_sql_create_context_record(
        {
            "answer": "SELECT COUNT(*) FROM head WHERE age > 56",
            "question": "How many heads of the departments are older than 56 ?",
            "context": "CREATE TABLE head (age INTEGER)",
        }
    )

    assert record.completion == "SELECT COUNT(*) FROM head WHERE age > 56"
    assert "CREATE TABLE head (age INTEGER)" in record.prompt
