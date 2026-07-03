#!/usr/bin/env bash
set -euo pipefail

repo_id="${HF_REPO_ID:-}"
adapter_dir="${ADAPTER_DIR:-outputs/adapters/lora_r32}"
model_card_dir="${MODEL_CARD_DIR:-model_cards/qwen25-15b-text2sql-lora-r32}"

if [[ -z "${repo_id}" ]]; then
  echo "HF_REPO_ID is required, for example: HF_REPO_ID=W-Kaski/qwen25-15b-text2sql-lora-r32"
  exit 2
fi

if ! hf auth whoami >/dev/null 2>&1; then
  echo "Hugging Face CLI is not logged in. Run 'hf auth login' locally, then rerun this script."
  exit 2
fi

if [[ ! -d "${adapter_dir}" ]]; then
  echo "Adapter directory not found: ${adapter_dir}"
  exit 2
fi

work_dir="$(mktemp -d)"
cleanup() {
  rm -rf "${work_dir}"
}
trap cleanup EXIT

cp -R "${adapter_dir}/." "${work_dir}/"
cp "${model_card_dir}/README.md" "${work_dir}/README.md"

hf upload "${repo_id}" "${work_dir}" . \
  --repo-type model \
  --commit-message "Upload Qwen2.5 Text-to-SQL LoRA adapter"
