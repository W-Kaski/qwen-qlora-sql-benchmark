from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from qwen_qlora_sql_benchmark.eval.sql_metrics import compute_prediction_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score prediction JSONL with quality metrics.")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-label", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    with args.predictions.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = json.loads(line)
            metrics = compute_prediction_metrics(payload["prediction"], payload["reference"])
            rows.append({"line_number": line_number, **metrics})
    frame = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "model": args.model_label,
                "count": len(frame),
                "exact_match": float(frame["exact_match"].mean()),
                "sql_parse_valid": float(frame["sql_parse_valid"].mean()),
                "starts_select": float(frame["starts_select"].mean()),
                "empty": float(frame["empty"].mean()),
            }
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
