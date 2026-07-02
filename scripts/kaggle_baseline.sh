#!/usr/bin/env bash
set -euo pipefail

python -m qwen_qlora_sql_benchmark.eval.baseline_generation --config configs/train_baseline.yaml
python -m qwen_qlora_sql_benchmark.eval.score_exact_match \
  --predictions results/eval_outputs/base_model_predictions.jsonl \
  --output results/tables/baseline_eval.csv
