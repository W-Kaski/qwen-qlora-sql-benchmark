#!/usr/bin/env bash
set -euo pipefail

port="${PORT:-8000}"
api_url="http://127.0.0.1:${port}"

cleanup() {
  if [[ -n "${server_pid:-}" ]]; then
    kill "${server_pid}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

PORT="${port}" scripts/run_api.sh >/tmp/qwen_qlora_sql_api.log 2>&1 &
server_pid="$!"

for _ in $(seq 1 60); do
  if curl -fsS "${api_url}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

curl -fsS "${api_url}/health" >/dev/null
API_URL="${api_url}" scripts/demo_request.sh
