#!/usr/bin/env bash
set -euo pipefail

python -m qwen_qlora_sql_benchmark.train.train_qlora --config configs/train_lora_r8.yaml
