# Qwen QLoRA Text-to-SQL Benchmark

This repo is a reproducible Kaggle T4 project for QLoRA fine-tuning and serving benchmarks with `Qwen/Qwen2.5-1.5B-Instruct`.

## Scope

The first release answers three questions:

1. Can QLoRA fine-tuning run reliably on Kaggle T4 for Qwen2.5-1.5B-Instruct?
2. How do LoRA ranks 8, 16, and 32 affect Text-to-SQL quality, training time, and memory?
3. How does vLLM base-model serving compare with Transformers on latency and throughput?

## Experiment Design

| Track | Tooling | Output |
| --- | --- | --- |
| baseline quality | Transformers | base model predictions and metrics |
| fine-tuned quality | Transformers + PEFT | adapter predictions and metrics |
| serving benchmark | vLLM and Transformers | latency and throughput tables |

vLLM LoRA serving is not required for the first release.

## Dataset Notes

See [docs/DATASET_ANALYSIS.md](docs/DATASET_ANALYSIS.md) for split statistics, SQL pattern distribution, and baseline failure modes.

## Repository Layout

```text
configs/      YAML experiment configs
data/         local data staging, ignored except directory markers
docs/         project specs and experiment log
notebooks/    Kaggle notebooks, one job per notebook
outputs/      adapters, checkpoints, diagnostics
results/      result tables, figures, logs, predictions
scripts/      shell entrypoints for Kaggle
src/          reusable Python package
tests/        local validation tests
```

## Local Validation

```bash
uv run pytest
uv run ruff check .
```

## Kaggle Setup

```bash
pip install -r requirements-kaggle.txt
python -m qwen_qlora_sql_benchmark.utils
```

## Current Status

Project scaffold is ready. The first dataset is `b-mc2/sql-create-context`; exact fields have been verified as `answer`, `question`, and `context`.

## Prepare Dataset

```bash
uv run --extra data python -m qwen_qlora_sql_benchmark.data.download_sql_create_context
```

## Run Baseline

On Kaggle or another Linux GPU runtime:

```bash
scripts/kaggle_baseline.sh
```

Current baseline result:

| Model | Eval rows | Exact Match | Runtime |
| --- | ---: | ---: | ---: |
| Qwen2.5-1.5B-Instruct | 500 | 0.044 | 185.80 seconds |

Current diagnostic result:

| Model | Eval rows | Exact Match | Runtime |
| --- | ---: | ---: | ---: |
| baseline on first 50 rows | 50 | 0.04 | already generated |
| LoRA rank 8 diagnostic | 50 | 0.48 | 64.98 seconds |

Current full rank 8 result:

| Model | Eval rows | Exact Match | Runtime |
| --- | ---: | ---: | ---: |
| Qwen2.5-1.5B-Instruct | 500 | 0.044 | 185.80 seconds |
| LoRA rank 8 | 500 | 0.684 | 568.25 seconds |
| LoRA rank 16 | 500 | 0.696 | 657.88 seconds |
| LoRA rank 32 | 500 | 0.712 | 530.83 seconds |

Current finding: rank 32 has the best Exact Match result, while rank 8 is close and cheaper in adapter size. This is still an Exact Match result, not a database execution result.
