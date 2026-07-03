from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from qwen_qlora_sql_benchmark.eval.error_analysis import summarize_prediction_errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Text-to-SQL prediction error types.")
    parser.add_argument(
        "--prediction",
        action="append",
        nargs=2,
        metavar=("MODEL", "PATH"),
        required=True,
        help="Model label and prediction JSONL path. Can be provided more than once.",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for model_label, prediction_path in args.prediction:
        summary = summarize_prediction_errors(Path(prediction_path))
        for error_type, count in summary["counts"].items():
            rows.append(
                {
                    "model": model_label,
                    "error_type": error_type,
                    "count": count,
                    "rate": summary["rates"][error_type],
                    "rows": summary["rows"],
                }
            )

    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(frame.to_csv(index=False))


if __name__ == "__main__":
    main()
