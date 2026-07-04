# Qwen2.5 QLoRA Text-to-SQL Benchmark

Reproducible QLoRA fine-tuning and evaluation for
`Qwen/Qwen2.5-1.5B-Instruct` on `b-mc2/sql-create-context`.

This repository evaluates parameter-efficient adaptation for constrained
schema-conditioned SQL generation. It covers dataset conversion, YAML-managed
QLoRA training, LoRA rank ablation, exact-match evaluation, SQL parse checks,
error analysis, result plots, and a local FastAPI demo with SQL validation.

It is not a production SQL assistant.

## Result Provenance

Full rank-ablation results were produced on a local WSL GPU. Kaggle validation
covers environment setup, tests, lint, dataset preparation, and GPU runtime
availability, not the reported full training runs.

## Results

Evaluation split: 500 examples.

| Model | Exact Match | SQL Parse Valid | Main Error Type |
| --- | ---: | ---: | --- |
| baseline | 0.044 | 0.980 | filter / condition mismatch |
| LoRA rank 8 | 0.684 | 0.992 | filter / condition mismatch |
| LoRA rank 16 | 0.696 | 0.994 | filter / condition mismatch |
| LoRA rank 32 | 0.712 | 0.990 | filter / condition mismatch |

In this single-seed 500-example split, rank 32 achieved the highest Exact Match;
the incremental gain over rank 16 was modest. The base model already produced
parseable SQL most of the time, so the main gain came from better column
selection, filter construction, aggregation choice, and dataset-specific SQL
formatting.

## Method

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Dataset: `b-mc2/sql-create-context`
- Verified source fields: `answer`, `question`, `context`
- Train split: 5000 rows
- Eval split: 500 rows
- SFT format: prompt-completion JSONL
- Loss target: completion tokens only
- Quantization: 4-bit NF4
- LoRA ranks: `r=8`, `r=16`, `r=32`
- Epochs: 1
- Max sequence length: 1024
- Training and generation seed: 42

All experiment parameters are stored in `configs/`.

LoRA parameter scale:

| Rank | Trainable Parameters | Trainable Percent |
| ---: | ---: | ---: |
| 8 | 9,232,384 | 0.517% |
| 16 | 18,464,768 | 1.028% |
| 32 | 36,929,536 | 2.036% |

## Controlled SQLite Execution Sanity Check

The repository includes a small SQLite-backed execution check for the rank 32
adapter. It executes generated SQL against in-memory SQLite databases and
compares the result with reference SQL execution.

The 30 cases are manually curated and are not a held-out benchmark.

| Cases | Parse Valid | Select-only | Execution Valid | Execution Accuracy | P50 Latency | P95 Latency |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 30 | 1.000 | 1.000 | 1.000 | 0.600 | 0.456s | 1.182s |

This is a small execution sanity check, not a broad Text-to-SQL benchmark. The
main remaining failures are value normalization, wrong projection, string
predicates, `NULL`, `GROUP BY`, and `LIMIT/OFFSET`.

## Single-Request Serving Sanity Check

Base model, sequential single-request benchmark:

| Backend | Requests | P50 latency | P95 latency | Tokens/s |
| --- | ---: | ---: | ---: | ---: |
| Transformers | 20 | 0.3144s | 0.5359s | 76.18 |
| vLLM | 20 | 0.2156s | 0.4098s | 74.61 |

This check measures local single-request latency. It does not measure
high-concurrency serving throughput.

## Reproduce

Install dependencies:

```bash
uv sync --extra dev --extra data --extra kaggle
```

Prepare the dataset:

```bash
uv run --extra data python -m qwen_qlora_sql_benchmark.data.download_sql_create_context
```

Run rank training and evaluation:

```bash
scripts/train_r8.sh
scripts/eval_r8.sh
scripts/train_r16.sh
scripts/eval_r16.sh
scripts/train_r32.sh
scripts/eval_r32.sh
```

Post-process results:

```bash
scripts/score_quality_metrics.sh
scripts/analyze_errors.sh
scripts/plot_results.sh
```

Run the SQLite-backed execution check:

```bash
scripts/run_execution_eval.sh
```

Full reproduction notes are in [docs/REPRODUCTION.md](docs/REPRODUCTION.md).
Kaggle environment validation is documented in
[docs/KAGGLE_VALIDATION.md](docs/KAGGLE_VALIDATION.md).

## Local API Demo

Start the local API and browser console:

```bash
scripts/run_api.sh
```

Open:

```text
http://127.0.0.1:8000/
```

The API endpoint `/generate-sql` accepts:

- `schema`
- `question`
- `execute`
- `setup_sql`
- `max_rows`
- `timeout_ms`

It returns generated SQL, SQL parse status, read-only `SELECT` status, latency,
and optional SQLite execution metadata. The real generator expects the rank 32
adapter under `outputs/adapters/lora_r32`.

When `execute` is true, `setup_sql` is limited to single `CREATE TABLE` or
`INSERT` statements before the generated read-only `SELECT` is executed.

Send one scripted request:

```bash
scripts/demo_request.sh
```

Run the API and scripted request together:

```bash
scripts/run_local_demo.sh
```

## Repository Layout

```text
configs/       YAML experiment configs
data/          local data staging and SQLite execution cases
docs/          technical report, reproduction notes, and analyses
examples/      API request examples
model_cards/   Hugging Face adapter card
notebooks/     Kaggle-oriented setup and data notebooks
outputs/       local adapters and checkpoints, ignored by git
results/       tracked result tables and figures
scripts/       local, Kaggle, API, eval, and packaging entrypoints
src/           Python package
tests/         unit tests
```

## Hugging Face

Rank 32 adapter:
[W-Kaski/qwen25-15b-text2sql-lora-r32](https://huggingface.co/W-Kaski/qwen25-15b-text2sql-lora-r32)

The GitHub repository does not track adapter weights or checkpoints. Packaging
assets are under `model_cards/`.

## Validation

```bash
uv run --extra dev pytest
uv run --extra dev ruff check .
```

Latest local validation:

- `66 passed`
- `ruff check .`: all checks passed

## Reports

- [Technical report](docs/TECHNICAL_REPORT.md)
- [Reproduction](docs/REPRODUCTION.md)
- [Dataset analysis](docs/DATASET_ANALYSIS.md)
- [Evaluation analysis](docs/EVAL_ANALYSIS.md)
- [Error analysis](docs/ERROR_ANALYSIS.md)
- [SQLite execution evaluation](docs/EXECUTION_EVAL.md)
- [Serving benchmark analysis](docs/BENCHMARK_ANALYSIS.md)
- [Hugging Face packaging](docs/HUGGING_FACE_PACKAGING.md)

Tracked result tables include `results/tables/run_metadata.csv` and
`results/tables/parameter_count.csv`.

## Limits

- Exact Match rejects semantically equivalent SQL.
- SQL parse validity does not prove execution correctness.
- The main dataset is mostly single-table SQL.
- The SQLite execution check is small and uses in-memory databases.
- vLLM LoRA serving is not part of this version.
- Generated SQL must be validated and sandboxed before any user-facing use.
- Full rank training results were produced on a local WSL GPU. Kaggle validation
  covers setup, tests, lint, dataset preparation, and GPU runtime availability.
