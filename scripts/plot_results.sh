#!/usr/bin/env bash
set -euo pipefail

uv run python -m qwen_qlora_sql_benchmark.visualize.plot_results
