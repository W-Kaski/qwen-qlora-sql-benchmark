from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def lora_eval_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame[(frame["rank"] != "baseline") & (frame["eval_rows"] == 500)].copy()
    rows["rank"] = rows["rank"].astype(int)
    return rows.sort_values("rank")


def save_exact_match_plot(eval_summary: pd.DataFrame, output: Path) -> None:
    baseline = eval_summary[eval_summary["rank"] == "baseline"].iloc[0]
    lora_rows = lora_eval_rows(eval_summary)
    labels = ["baseline"] + [f"r{rank}" for rank in lora_rows["rank"]]
    values = [baseline["exact_match"]] + list(lora_rows["exact_match"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values, color=["#6B7280", "#2563EB", "#059669", "#DC2626"])
    ax.set_ylabel("Exact Match")
    ax.set_ylim(0, 0.8)
    ax.set_title("Text-to-SQL Exact Match")
    for index, value in enumerate(values):
        ax.text(index, value + 0.015, f"{value:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def save_training_time_plot(training_summary: pd.DataFrame, output: Path) -> None:
    rows = training_summary.sort_values("rank")
    labels = [f"r{rank}" for rank in rows["rank"]]
    values = list(rows["wall_time_seconds"] / 60.0)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values, color="#2563EB")
    ax.set_ylabel("Minutes")
    ax.set_title("Training Wall Time")
    for index, value in enumerate(values):
        ax.text(index, value + 0.15, f"{value:.1f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def save_sql_validity_plot(quality_metrics: pd.DataFrame, output: Path) -> None:
    labels = list(quality_metrics["model"])
    values = list(quality_metrics["sql_parse_valid"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values, color="#059669")
    ax.set_ylabel("SQL Parse Valid")
    ax.set_ylim(0.94, 1.0)
    ax.set_title("SQL Parse Validity")
    ax.tick_params(axis="x", rotation=20)
    for index, value in enumerate(values):
        ax.text(index, value + 0.001, f"{value:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate result plots.")
    parser.add_argument(
        "--eval-summary", type=Path, default=Path("results/tables/eval_summary.csv")
    )
    parser.add_argument(
        "--training-summary", type=Path, default=Path("results/tables/training_summary.csv")
    )
    parser.add_argument(
        "--quality-metrics", type=Path, default=Path("results/tables/quality_metrics.csv")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    eval_summary = pd.read_csv(args.eval_summary)
    training_summary = pd.read_csv(args.training_summary)
    quality_metrics = pd.read_csv(args.quality_metrics)
    save_exact_match_plot(eval_summary, args.output_dir / "exact_match_by_rank.png")
    save_training_time_plot(training_summary, args.output_dir / "training_time_by_rank.png")
    save_sql_validity_plot(quality_metrics, args.output_dir / "sql_validity_by_model.png")


if __name__ == "__main__":
    main()
