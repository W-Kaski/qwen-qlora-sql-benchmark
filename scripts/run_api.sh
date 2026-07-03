#!/usr/bin/env bash
set -euo pipefail

uv run --extra api --extra kaggle uvicorn qwen_qlora_sql_benchmark.api.app:app \
  --host 127.0.0.1 \
  --port "${PORT:-8000}"
