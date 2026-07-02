from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qwen_qlora_sql_benchmark.data.prompt_completion import validate_prompt_completion_record


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            records.append(payload)
    return records


def write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def convert_records(
    records: list[dict[str, Any]], prompt_field: str, completion_field: str
) -> list[dict[str, str]]:
    converted: list[dict[str, str]] = []
    for index, record in enumerate(records):
        if prompt_field not in record:
            raise KeyError(f"record[{index}].{prompt_field}")
        if completion_field not in record:
            raise KeyError(f"record[{index}].{completion_field}")
        validated = validate_prompt_completion_record(
            {"prompt": record[prompt_field], "completion": record[completion_field]}
        )
        converted.append({"prompt": validated.prompt, "completion": validated.completion})
    return converted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert explicit JSONL fields to prompt-completion JSONL."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prompt-field", required=True)
    parser.add_argument("--completion-field", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.input)
    converted = convert_records(records, args.prompt_field, args.completion_field)
    write_jsonl(args.output, converted)


if __name__ == "__main__":
    main()
