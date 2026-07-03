# Hugging Face Packaging

## Status

Adapter packaging is uploaded to:

```text
https://huggingface.co/W-Kaski/qwen25-15b-text2sql-lora-r32
```

The uploaded repo contains the final adapter, tokenizer files, and model card. Training checkpoints and optimizer state are intentionally excluded.

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

To re-upload after retraining:

```bash
HF_REPO_ID=W-Kaski/qwen25-15b-text2sql-lora-r32 scripts/hf_upload_adapter.sh
```

The script copies:

- `outputs/adapters/lora_r32`
- `model_cards/qwen25-15b-text2sql-lora-r32/README.md`

into a temporary directory and uploads that directory with `hf upload`.

The script does not print or inspect any token value.
