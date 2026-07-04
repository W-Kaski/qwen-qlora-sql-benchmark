#!/usr/bin/env bash
set -euo pipefail

python -m pip install -r requirements-kaggle.txt
python - <<'PY'
from qwen_qlora_sql_benchmark.utils.gpu import collect_runtime_snapshot

print(collect_runtime_snapshot())
PY
