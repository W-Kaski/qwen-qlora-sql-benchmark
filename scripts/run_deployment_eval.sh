#!/usr/bin/env bash
set -euo pipefail

uv run --extra kaggle python -m qwen_qlora_sql_benchmark.eval.deployment_eval \
  --cases data/deployment_eval/cases.jsonl \
  --output results/tables/deployment_eval.csv \
  --summary-output results/tables/deployment_eval_summary.csv \
  --adapter-path outputs/adapters/lora_r32
