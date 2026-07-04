# Reproduction

This document describes the path used to reproduce the tracked result tables and
figures in this repository.

## Environment

The project is managed with `uv` and expects Python 3.10 or newer.

Install the development, data, and GPU experiment dependencies:

```bash
uv sync --extra dev --extra data --extra kaggle
```

The reported full-rank QLoRA runs were produced on a local WSL GPU environment.
Kaggle validation covers environment setup, tests, lint, dataset preparation,
and GPU runtime availability. It does not run the full rank ablation.

## Data

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

Prepare the split:

```bash
uv run --extra data python -m qwen_qlora_sql_benchmark.data.download_sql_create_context
```

Expected outputs:

- `data/splits/train.jsonl`
- `data/splits/eval.jsonl`
- `data/splits/benchmark_prompts.jsonl`

The split JSONL files are not tracked in Git.

## Training

The training entrypoint is:

```bash
python -m qwen_qlora_sql_benchmark.train.train_qlora --config <config>
```

Convenience scripts:

```bash
scripts/train_r8.sh
scripts/train_r16.sh
scripts/train_r32.sh
```

Expected adapter directories:

- `outputs/adapters/lora_r8`
- `outputs/adapters/lora_r16`
- `outputs/adapters/lora_r32`

Adapter weights and checkpoints are not tracked in Git.

Training configs set `training.seed: 42`. The training entrypoint passes this
value to Transformers `set_seed`, `TrainingArguments.seed`, and
`TrainingArguments.data_seed`.

## Evaluation

Run baseline generation:

```bash
scripts/baseline_generate.sh
```

Run adapter evaluation:

```bash
scripts/eval_r8.sh
scripts/eval_r16.sh
scripts/eval_r32.sh
```

Post-process prediction files:

```bash
scripts/score_quality_metrics.sh
scripts/analyze_errors.sh
scripts/plot_results.sh
```

Tracked summary outputs:

- `results/tables/eval_summary.csv`
- `results/tables/quality_metrics.csv`
- `results/tables/error_analysis.csv`
- `results/tables/run_metadata.csv`
- `results/tables/parameter_count.csv`
- `results/figures/exact_match_by_rank.png`
- `results/figures/sql_validity_by_model.png`
- `results/figures/training_time_by_rank.png`

Raw prediction JSONL files are not tracked in Git.

Generation configs set `generation.seed: 42`. Baseline and adapter generation
pass this value to Transformers `set_seed` before loading the model.

## Controlled SQLite Execution Sanity Check

Run:

```bash
scripts/run_execution_eval.sh
```

Inputs:

- `data/execution_eval/cases.jsonl`

Outputs:

- `results/tables/execution_eval.csv`
- `results/tables/execution_eval_summary.csv`

This check uses 30 manually curated SQLite-backed cases and in-memory databases.
It is a small execution sanity check, not a held-out benchmark or broad
deployment benchmark.

The execution helper restricts setup SQL to single `CREATE TABLE` or `INSERT`
statements and restricts generated SQL to one read-only `SELECT`.

## Single-Request Serving Sanity Check

Run:

```bash
scripts/run_serving_benchmark.sh
```

Tracked outputs:

- `results/tables/serving_benchmark.csv`
- `results/tables/serving_benchmark_raw.csv`

The benchmark is sequential and single-request. It does not measure
high-concurrency serving throughput.

## Validation

Run:

```bash
uv run --extra dev pytest
uv run --extra dev ruff check .
```

Current validation status:

- `66 passed`
- `ruff check .`: all checks passed

## Known Gaps

- Exact Match rejects semantically equivalent SQL.
- SQL parse validity is not execution correctness.
- The main dataset is mostly single-table SQL.
- The SQLite execution check is small.
- Full adapter weights are stored outside Git.
- Full rank training requires a CUDA GPU.
