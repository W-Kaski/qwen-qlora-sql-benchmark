#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src uv run python -m qwen_qlora_sql_benchmark.eval.summarize_errors \
  --prediction baseline results/eval_outputs/base_model_predictions.jsonl \
  --prediction lora_r8 results/eval_outputs/lora_r8_predictions.jsonl \
  --prediction lora_r16 results/eval_outputs/lora_r16_predictions.jsonl \
  --prediction lora_r32 results/eval_outputs/lora_r32_predictions.jsonl \
  --output results/tables/error_analysis.csv
