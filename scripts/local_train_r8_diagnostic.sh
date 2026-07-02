#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python3 -m qwen_qlora_sql_benchmark.train.train_qlora \
  --config configs/train_lora_r8_diagnostic.yaml
