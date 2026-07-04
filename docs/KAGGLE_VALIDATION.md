# Kaggle Validation

This document records the Kaggle environment validation path for the repository.
It is not evidence of a full Kaggle QLoRA rank ablation run.

## Runtime

Recommended Kaggle runtime:

- GPU: single NVIDIA T4 16 GB
- Python GPU notebook
- Internet enabled for Hugging Face model and dataset downloads

The first release targets `Qwen/Qwen2.5-1.5B-Instruct`. Do not start with a 7B
model for this project version.

## Completed Validation

Kaggle validation has been run through:

- kernel: `ericwang7717/qwen-qlora-sql-validation`
- version: 3
- status: complete
- URL: `https://www.kaggle.com/code/ericwang7717/qwen-qlora-sql-validation`

Validated steps:

- GitHub repo clone
- GPU runtime availability through `nvidia-smi`
- `uv run --extra dev pytest`
- `uv run --extra dev ruff check .`
- dataset preparation with `qwen_qlora_sql_benchmark.data.download_sql_create_context`
- runtime snapshot command

Observed validation result:

- tests: 55 passed
- lint: all checks passed
- status file: `KAGGLE_VALIDATION_OK`

The validation run does not execute full QLoRA training.

## Setup Commands

Install Kaggle dependencies:

```bash
pip install -r requirements-kaggle.txt
```

Run setup check:

```bash
scripts/kaggle_validate_environment.sh
```

Expected setup output includes the Python version, CUDA availability, GPU name,
and memory summary.

Prepare data:

```bash
scripts/kaggle_prepare_dataset.sh
```

Expected outputs:

- `data/splits/train.jsonl`
- `data/splits/eval.jsonl`
- `data/splits/benchmark_prompts.jsonl`

The source dataset fields are `answer`, `question`, and `context`.

## Full Training Path

The reported full rank-ablation results were produced on a local WSL GPU, not
by the completed Kaggle validation run. The same training and evaluation
entrypoints can be used on Kaggle after environment validation:

```bash
scripts/train_r8.sh
scripts/eval_r8.sh
scripts/train_r16.sh
scripts/eval_r16.sh
scripts/train_r32.sh
scripts/eval_r32.sh
```

Expected adapter outputs:

- `outputs/adapters/lora_r8`
- `outputs/adapters/lora_r16`
- `outputs/adapters/lora_r32`

Expected prediction outputs:

- `results/eval_outputs/lora_r8_predictions.jsonl`
- `results/eval_outputs/lora_r16_predictions.jsonl`
- `results/eval_outputs/lora_r32_predictions.jsonl`

Expected table outputs:

- `results/tables/lora_r8_eval.csv`
- `results/tables/lora_r16_eval.csv`
- `results/tables/lora_r32_eval.csv`

## Result Post-processing

```bash
scripts/score_quality_metrics.sh
scripts/analyze_errors.sh
scripts/plot_results.sh
```

## Artifact Boundary

Tracked in GitHub:

- source code
- configs
- docs
- scripts
- result CSV summaries
- figures

Not tracked in GitHub:

- split JSONL payloads
- prediction JSONL payloads
- LoRA adapter weights
- checkpoints
- local logs

After a Kaggle run, download the outputs and copy them into the same paths
before regenerating result tables and figures.

## T4 Risk Notes

Rank 8, 16, and 32 QLoRA runs are configured to fit a 16 GB T4 with batch size 1
and gradient accumulation.

If a run fails with OOM:

1. reduce `training.max_seq_length`
2. reduce eval batch pressure by evaluating fewer rows first
3. run a diagnostic config before a full config
4. restart the Kaggle runtime before retrying
