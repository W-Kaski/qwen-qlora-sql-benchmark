# Hugging Face Packaging

## Status

Packaging files are ready. The local Hugging Face CLI is logged in as `W-Kaski`, but adapter upload returned `403 Forbidden` when creating `W-Kaski/qwen25-15b-text2sql-lora-r32`.

Current blocker:

- the active Hugging Face token does not have permission to create a model repo under the `W-Kaski` namespace, or
- the target namespace/repo id needs to be changed.

Check auth without exposing tokens:

```bash
hf auth whoami
```

## Model Card

The model card is stored at:

```text
model_cards/qwen25-15b-text2sql-lora-r32/README.md
```

## Upload

After logging in locally with a token that has write access:

```bash
HF_REPO_ID=W-Kaski/qwen25-15b-text2sql-lora-r32 scripts/hf_upload_adapter.sh
```

The script copies:

- `outputs/adapters/lora_r32`
- `model_cards/qwen25-15b-text2sql-lora-r32/README.md`

into a temporary directory and uploads that directory with `hf upload`.

The script does not print or inspect any token value.
