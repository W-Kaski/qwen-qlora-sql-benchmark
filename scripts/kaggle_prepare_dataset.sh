#!/usr/bin/env bash
set -euo pipefail

python -m qwen_qlora_sql_benchmark.data.download_sql_create_context "$@"
