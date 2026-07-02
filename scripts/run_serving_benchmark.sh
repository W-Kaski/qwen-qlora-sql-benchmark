#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python3 -m qwen_qlora_sql_benchmark.benchmark.serving_benchmark \
  --config configs/benchmark_vllm.yaml
