# Technical Report

## Overview

This report evaluates QLoRA fine-tuning for Text-to-SQL generation under a 16 GB GPU budget using `Qwen/Qwen2.5-1.5B-Instruct`.

The experiment compares:

- base model baseline
- LoRA rank 8
- LoRA rank 16
- LoRA rank 32

It also includes a base model serving benchmark comparing Transformers and vLLM.

## Dataset

Dataset: `b-mc2/sql-create-context`

Verified source fields:

- `answer`
- `question`
- `context`

Local split:

| Split | Rows |
| --- | ---: |
| train | 5000 |
| eval | 500 |
| benchmark prompts | 100 |

The converted SFT format is prompt-completion JSONL.

## Method

The training setup uses:

- QLoRA
- 4-bit NF4 quantization
- completion-only loss
- LoRA ranks 8, 16, and 32
- 1 epoch over 5000 train rows
- 500-row eval split
- training and generation seed 42

All experiment parameters are stored in YAML configs under `configs/`.

## Parameter Scale

| Rank | Trainable Parameters | Total Parameters With Adapter | Trainable Percent |
| ---: | ---: | ---: | ---: |
| 8 | 9,232,384 | 1,786,320,384 | 0.517% |
| 16 | 18,464,768 | 1,795,552,768 | 1.028% |
| 32 | 36,929,536 | 1,814,017,536 | 2.036% |

Counts are stored in `results/tables/parameter_count.csv`. They were computed
from `Qwen/Qwen2.5-1.5B-Instruct` config using empty-weight model
initialization, then applying the same PEFT LoRA target modules as the training
configs.

## Quality Results

| Model | Exact Match | SQL Parse Valid |
| --- | ---: | ---: |
| baseline | 0.044 | 0.980 |
| LoRA rank 8 | 0.684 | 0.992 |
| LoRA rank 16 | 0.696 | 0.994 |
| LoRA rank 32 | 0.712 | 0.990 |

Rank 32 is the best Exact Match result in the current ablation. Rank 8 remains competitive and has fewer LoRA parameters.

## Error Analysis

| Model | Exact Match | Filter/Condition Mismatch | Projection Mismatch | Invalid SQL |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.044 | 0.736 | 0.182 | 0.020 |
| LoRA rank 8 | 0.684 | 0.164 | 0.132 | 0.006 |
| LoRA rank 16 | 0.696 | 0.154 | 0.136 | 0.004 |
| LoRA rank 32 | 0.712 | 0.146 | 0.126 | 0.008 |

The main baseline failure mode is filter or condition mismatch. The rank 32 adapter reduced this error type from 0.736 to 0.146.

## Training Runtime

| Rank | Train rows | Eval rows | Wall time | Final eval loss |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 5000 | 500 | 1095.10s | 0.0595 |
| 16 | 5000 | 500 | 1132.98s | 0.0537 |
| 32 | 5000 | 500 | 1077.81s | 0.0498 |

Run metadata is stored in `results/tables/run_metadata.csv`.

## Serving Benchmark

Base model serving benchmark:

| Backend | Requests | P50 latency | P95 latency | Tokens/s |
| --- | ---: | ---: | ---: | ---: |
| Transformers | 20 | 0.3144s | 0.5359s | 76.18 |
| vLLM | 20 | 0.2156s | 0.4098s | 74.61 |

This benchmark is a local single-request sanity check. It does not measure high-concurrency serving throughput.

## Key Findings

1. QLoRA increased Exact Match from 0.044 to 0.712 on this split.
2. The base model already produces SQL-shaped output, so the main improvement is exact dataset-style query matching.
3. Rank 32 performs best in Exact Match, but rank 8 is close and cheaper.
4. vLLM lowers single-request latency in this local setup, while total tokens/s is similar for this sequential benchmark.

## Limitations

- Exact Match rejects semantically equivalent SQL.
- SQL parse validity is not Execution Accuracy.
- The dataset is mostly single-table SQL.
- No database execution files are included in the current split.
- vLLM LoRA serving is not part of the first release.
- Kaggle validation has setup scripts and runtime checks, but the reported full runs were local WSL GPU runs.

## Reproduction

```bash
uv run --extra data python -m qwen_qlora_sql_benchmark.data.download_sql_create_context
scripts/local_train_r8_diagnostic.sh
scripts/local_eval_r8_diagnostic.sh
PYTHONPATH=src python3 -m qwen_qlora_sql_benchmark.train.train_qlora --config configs/train_lora_r8.yaml
scripts/local_eval_r8.sh
scripts/score_quality_metrics.sh
scripts/run_serving_benchmark.sh
scripts/plot_results.sh
```

Run validation:

```bash
uv run --extra dev pytest
uv run --extra dev ruff check .
```
