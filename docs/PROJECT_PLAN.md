# Project Plan

## Goal

Build a reproducible Qwen2.5-1.5B-Instruct QLoRA Text-to-SQL experiment that can run on Kaggle T4 and be reviewed from GitHub.

## Research Questions

1. Can QLoRA fine-tuning run reliably on Kaggle T4 for a 1.5B instruct model?
2. How do LoRA ranks 8, 16, and 32 affect Text-to-SQL quality, memory, and training time?
3. How does vLLM base-model serving compare with Transformers for latency and throughput?

## First Release Boundary

Included:

- Baseline generation with Qwen2.5-1.5B-Instruct.
- QLoRA rank ablation for ranks 8, 16, and 32.
- Exact Match and SQL validity.
- Execution Accuracy only after database files and schema execution are verified.
- vLLM benchmark for the base model.

Excluded from first release:

- 7B training.
- DeepSpeed.
- Frontend.
- Docker deployment.
- Required vLLM LoRA serving.
- Hugging Face upload.

## Milestones

1. Scaffold repo, configs, docs, tests, and local validation.
2. Select dataset and inspect exact field names before writing conversion commands.
3. Convert data to prompt-completion JSONL.
4. Run base model baseline and save prediction artifacts.
5. Run a small Kaggle QLoRA diagnostic for rank 8.
6. Run rank 8, rank 16, and rank 32 experiments.
7. Score outputs and create result tables and figures.
8. Run vLLM base model benchmark.
9. Write final README and technical report.

## Decision Gates

- Do not run full training until the diagnostic confirms format, memory, loss, and generation behavior.
- Do not report quality improvement without baseline metrics.
- Do not make remote GitHub changes without explicit approval.
