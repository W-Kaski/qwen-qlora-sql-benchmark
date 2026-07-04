#!/usr/bin/env bash
set -euo pipefail

uv run --extra kaggle python -m qwen_qlora_sql_benchmark.eval.execution_eval \
  --cases data/execution_eval/cases.jsonl \
  --output results/tables/execution_eval.csv \
  --summary-output results/tables/execution_eval_summary.csv \
  --adapter-path outputs/adapters/lora_r32
