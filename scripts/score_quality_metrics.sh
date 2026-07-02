#!/usr/bin/env bash
set -euo pipefail

tmp_dir="$(mktemp -d)"
PYTHONPATH=src uv run python -m qwen_qlora_sql_benchmark.eval.score_predictions \
  --predictions results/eval_outputs/base_model_predictions.jsonl \
  --output "$tmp_dir/baseline.csv" \
  --model-label baseline
PYTHONPATH=src uv run python -m qwen_qlora_sql_benchmark.eval.score_predictions \
  --predictions results/eval_outputs/lora_r8_predictions.jsonl \
  --output "$tmp_dir/lora_r8.csv" \
  --model-label lora_r8
PYTHONPATH=src uv run python -m qwen_qlora_sql_benchmark.eval.score_predictions \
  --predictions results/eval_outputs/lora_r16_predictions.jsonl \
  --output "$tmp_dir/lora_r16.csv" \
  --model-label lora_r16
PYTHONPATH=src uv run python -m qwen_qlora_sql_benchmark.eval.score_predictions \
  --predictions results/eval_outputs/lora_r32_predictions.jsonl \
  --output "$tmp_dir/lora_r32.csv" \
  --model-label lora_r32
uv run python - <<PY
from pathlib import Path

parts = [Path("$tmp_dir") / name for name in ["baseline.csv", "lora_r8.csv", "lora_r16.csv", "lora_r32.csv"]]
out = Path("results/tables/quality_metrics.csv")
lines = []
for index, path in enumerate(parts):
    rows = path.read_text(encoding="utf-8").splitlines()
    lines.extend(rows if index == 0 else rows[1:])
out.write_text("\\n".join(lines) + "\\n", encoding="utf-8")
print(out.read_text(encoding="utf-8"))
PY
