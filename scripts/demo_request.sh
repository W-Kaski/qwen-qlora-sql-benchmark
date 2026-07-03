#!/usr/bin/env bash
set -euo pipefail

api_url="${API_URL:-http://127.0.0.1:8000}"

curl -sS "${api_url}/generate-sql" \
  -H 'content-type: application/json' \
  --data-binary @examples/demo_request.json
