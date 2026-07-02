from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

from qwen_qlora_sql_benchmark.data.sql_create_context import convert_sql_create_context_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download b-mc2/sql-create-context and write prompt-completion splits."
    )
    parser.add_argument("--dataset-name", default="b-mc2/sql-create-context")
    parser.add_argument("--split", default="train")
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--eval-size", type=int, default=500)
    parser.add_argument("--benchmark-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    return parser.parse_args()


def write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def main() -> None:
    args = parse_args()
    required = args.train_size + args.eval_size + args.benchmark_size
    dataset = load_dataset(args.dataset_name, split=args.split)
    if len(dataset) < required:
        raise ValueError(f"dataset has {len(dataset)} rows, but {required} rows were requested")

    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    selected = indices[:required]

    converted = []
    for index in selected:
        record = convert_sql_create_context_record(dataset[index])
        converted.append({"prompt": record.prompt, "completion": record.completion})

    train_end = args.train_size
    eval_end = train_end + args.eval_size
    train_records = converted[:train_end]
    eval_records = converted[train_end:eval_end]
    benchmark_records = [{"prompt": row["prompt"]} for row in converted[eval_end:]]

    write_jsonl(args.output_dir / "train.jsonl", train_records)
    write_jsonl(args.output_dir / "eval.jsonl", eval_records)
    write_jsonl(args.output_dir / "benchmark_prompts.jsonl", benchmark_records)


if __name__ == "__main__":
    main()
