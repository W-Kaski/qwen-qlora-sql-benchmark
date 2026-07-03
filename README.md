# Qwen QLoRA Text-to-SQL Benchmark

QLoRA fine-tuning and deployment checks for `Qwen/Qwen2.5-1.5B-Instruct` on a small Text-to-SQL task.

The repo covers:

- rank ablation for LoRA `r=8`, `r=16`, and `r=32`
- baseline vs adapter quality metrics
- SQL error analysis
- base-model serving benchmark with Transformers and vLLM
- guarded FastAPI inference endpoint
- SQLite execution sandbox
- deployment-style execution evaluation
- Kaggle T4 reproduction scripts

## Results

### Text-to-SQL Eval

| Model | Exact Match | SQL Parse Valid | Main Error Type |
| --- | ---: | ---: | --- |
| baseline | 0.044 | 0.980 | filter / condition mismatch |
| LoRA rank 8 | 0.684 | 0.992 | filter / condition mismatch |
| LoRA rank 16 | 0.696 | 0.994 | filter / condition mismatch |
| LoRA rank 32 | 0.712 | 0.990 | filter / condition mismatch |

Exact Match increased from `0.044` to `0.712` with the rank 32 adapter. The base model already emits parseable SQL most of the time; the adapter mainly improves column selection, filter construction, aggregation choice, and dataset-specific SQL formatting.

### Deployment Eval

The deployment evaluation executes generated SQL against in-memory SQLite databases.

| Cases | Parse Valid | Select-only | Execution Valid | Execution Accuracy | P50 Latency | P95 Latency |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 30 | 1.000 | 1.000 | 1.000 | 0.600 | 0.456s | 1.182s |

Interpretation: the adapter is usable for a scoped demo with explicit schemas and validation metadata. It is not a general SQL assistant. The main remaining failures are value normalization, wrong projection, string predicates, `NULL`, `GROUP BY`, and `LIMIT/OFFSET`.

### Serving Benchmark

Base model, sequential single-request benchmark:

| Backend | Requests | P50 latency | P95 latency | Tokens/s |
| --- | ---: | ---: | ---: | ---: |
| Transformers | 20 | 0.3144s | 0.5359s | 76.18 |
| vLLM | 20 | 0.2156s | 0.4098s | 74.61 |

This is a latency sanity check, not a high-concurrency serving study.

## Repository Layout

```text
configs/       YAML experiment configs
data/          local data staging and deployment eval cases
docs/          reports, specs, and reproduction notes
examples/      API request examples
model_cards/   Hugging Face adapter card
notebooks/     Kaggle-oriented notebooks
outputs/       local adapters and checkpoints, ignored by git
results/       tracked result tables/figures plus ignored raw artifacts
scripts/       local, Kaggle, API, eval, and packaging entrypoints
src/           Python package
tests/         unit tests
```

## Quick Validation

```bash
uv run --extra dev pytest
uv run --extra dev ruff check .
```

## Local API Demo

Start the guarded API:

```bash
scripts/run_api.sh
```

Send one request:

```bash
scripts/demo_request.sh
```

Run both in one command:

```bash
scripts/run_local_demo.sh
```

The API returns:

- `sql`
- `parse_valid`
- `is_select_only`
- `latency_ms`
- `error`
- optional SQLite `execution` metadata

The real generator expects the rank 32 adapter under `outputs/adapters/lora_r32`.

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

Prepare the split:

```bash
uv run --extra data python -m qwen_qlora_sql_benchmark.data.download_sql_create_context
```

## Training And Evaluation

Local or Kaggle GPU runtime:

```bash
scripts/kaggle_train_r8.sh
scripts/kaggle_eval_r8.sh
scripts/kaggle_train_r16.sh
scripts/kaggle_eval_r16.sh
scripts/kaggle_train_r32.sh
scripts/kaggle_eval_r32.sh
```

Post-process:

```bash
scripts/score_quality_metrics.sh
scripts/analyze_errors.sh
scripts/plot_results.sh
```

Deployment evaluation:

```bash
scripts/run_deployment_eval.sh
```

## Kaggle

Kaggle setup:

```bash
pip install -r requirements-kaggle.txt
scripts/kaggle_setup_check.sh
scripts/kaggle_prepare_dataset.sh
```

Full runbook: [docs/KAGGLE_REPRODUCTION.md](docs/KAGGLE_REPRODUCTION.md)

Kaggle validation:

- kernel: `ericwang7717/qwen-qlora-sql-validation`
- version: 3
- status: complete
- checks: `pytest`, `ruff`, dataset preparation, runtime snapshot
- GPU runtime: confirmed with `nvidia-smi`

Current note: the reported full training runs were produced on local WSL GPU. Kaggle validation covers setup, tests, lint, data preparation, and GPU runtime availability; it does not run the full QLoRA training job.

## Hugging Face

Rank 32 adapter: [W-Kaski/qwen25-15b-text2sql-lora-r32](https://huggingface.co/W-Kaski/qwen25-15b-text2sql-lora-r32)

Adapter packaging assets are under `model_cards/`. Re-upload after retraining:

```bash
HF_REPO_ID=W-Kaski/qwen25-15b-text2sql-lora-r32 scripts/hf_upload_adapter.sh
```

Packaging details: [docs/HUGGING_FACE_PACKAGING.md](docs/HUGGING_FACE_PACKAGING.md)

## Reports

- [Technical report](docs/TECHNICAL_REPORT.md)
- [Dataset analysis](docs/DATASET_ANALYSIS.md)
- [Evaluation analysis](docs/EVAL_ANALYSIS.md)
- [Error analysis](docs/ERROR_ANALYSIS.md)
- [Deployment readiness](docs/DEPLOYMENT_READINESS.md)
- [Deployment evaluation](docs/DEPLOYMENT_EVAL.md)
- [Benchmark analysis](docs/BENCHMARK_ANALYSIS.md)

## Limits

- Exact Match rejects semantically equivalent SQL.
- SQL parse validity does not prove correctness.
- The main dataset is mostly single-table SQL.
- The deployment eval is small and SQLite-only.
- vLLM LoRA serving is not part of this version.
- Generated SQL must be validated and sandboxed before user-facing use.
