#!/usr/bin/env bash
set -euo pipefail

python -m qwen_qlora_sql_benchmark.eval.adapter_generation --config configs/eval_lora_r8.yaml
python -m qwen_qlora_sql_benchmark.eval.score_exact_match \
  --predictions results/eval_outputs/lora_r8_predictions.jsonl \
  --output results/tables/lora_r8_eval.csv
