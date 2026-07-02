from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from qwen_qlora_sql_benchmark.eval.exact_match import exact_match


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score prediction JSONL with Exact Match.")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prediction-field", default="prediction")
    parser.add_argument("--reference-field", default="reference")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    with args.predictions.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = json.loads(line)
            prediction = payload[args.prediction_field]
            reference = payload[args.reference_field]
            matched = exact_match(prediction, reference)
            rows.append({"line_number": line_number, "exact_match": matched})
    frame = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [{"count": len(frame), "exact_match": float(frame["exact_match"].mean())}]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
