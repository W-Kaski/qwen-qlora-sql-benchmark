import json
from pathlib import Path

from qwen_qlora_sql_benchmark.eval.baseline_generation import (
    EvalRecord,
    build_prediction_record,
    load_eval_records,
    strip_generated_sql,
    write_prediction_records,
)


def test_load_eval_records_reads_prompt_completion_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    path.write_text(
        json.dumps({"prompt": "Question:\nQ\n\nSQL:", "completion": "SELECT 1;"}) + "\n",
        encoding="utf-8",
    )

    records = load_eval_records(path)

    assert records == [EvalRecord(id="0", prompt="Question:\nQ\n\nSQL:", reference="SELECT 1;")]


def test_load_eval_records_rejects_blank_completion(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    path.write_text(
        json.dumps({"prompt": "Question:\nQ\n\nSQL:", "completion": " "}) + "\n",
        encoding="utf-8",
    )

    try:
        load_eval_records(path)
    except ValueError as exc:
        assert "completion must not be blank" in str(exc)
    else:
        raise AssertionError("expected blank completion to be rejected")


def test_build_prediction_record_uses_artifact_contract() -> None:
    record = EvalRecord(id="7", prompt="Question:\nQ\n\nSQL:", reference="SELECT 1;")

    payload = build_prediction_record(record, "SELECT 1;")

    assert payload == {
        "id": "7",
        "prompt": "Question:\nQ\n\nSQL:",
        "prediction": "SELECT 1;",
        "reference": "SELECT 1;",
    }


def test_strip_generated_sql_removes_prompt_prefix_when_present() -> None:
    prompt = "Question:\nQ\n\nSQL:"
    generated = "Question:\nQ\n\nSQL: SELECT 1;"

    assert strip_generated_sql(prompt, generated) == "SELECT 1;"


def test_write_prediction_records_writes_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"

    write_prediction_records(path, [{"id": "0", "prediction": "SELECT 1;"}])

    assert path.read_text(encoding="utf-8").strip() == '{"id": "0", "prediction": "SELECT 1;"}'
