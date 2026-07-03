# Deployment Readiness

## Summary

The rank 32 adapter works for scoped single-query Text-to-SQL demos with explicit schemas. It needs guardrails, validation, and execution feedback before broader user-facing use.

## Smoke Test Setup

The deployment smoke test used six hand-written SQLite-backed cases:

- simple single-table filtering
- count aggregation
- max aggregation
- two-condition filtering
- order by with limit
- two-table lookup through a subquery or join

Each generated SQL query was executed against an in-memory SQLite database and compared with the reference SQL execution result.

Runtime:

- local WSL GPU
- `Qwen/Qwen2.5-1.5B-Instruct`
- `outputs/adapters/lora_r32`
- 4-bit NF4 base model loading
- greedy decoding
- `max_new_tokens=128`

## Results

| Model | Cases | Parse Valid | Execution Match | Average Latency |
| --- | ---: | ---: | ---: | ---: |
| base model | 6 | 6 | 4 | 0.390s |
| LoRA rank 32 | 6 | 6 | 5 | 0.571s |

Detailed rows are stored in `results/tables/deployment_smoke_test.csv`.

The larger deployment evaluation uses 30 SQLite-backed cases and reports 60.0% execution accuracy. See `docs/DEPLOYMENT_EVAL.md`.

## Observed Behavior

The adapter reliably produced one SQL query and did not emit explanatory text in this smoke test. It improved the simple filtering case where the base model selected all columns instead of the requested column.

The remaining adapter failure was a value-normalization issue:

```sql
SELECT name FROM employees WHERE department = "Engineering" AND salary > 100000
```

The SQLite table contained `engineering`, so the generated query was syntactically valid but execution-incorrect under case-sensitive string comparison.

## Supported Use

Supported:

- internal demo with fixed schemas
- batch evaluation workflow
- controlled API where users pass explicit schema and natural-language question

Unsupported:

- arbitrary user-uploaded databases
- production analytics workloads
- multi-turn SQL editing
- safety-critical data work
- real customer deployment without query validation and sandboxing

## Required For Broader Use

1. Add an execution-evaluation dataset with database files.
2. Add a small API that validates generated SQL before returning it.
3. Restrict generated statements to read-only `SELECT`.
4. Add timeout and row-limit protection for execution.
5. Add case-insensitive or value-aware evaluation for realistic text fields.
6. Add input contracts for schema size, supported SQL dialect, and unsupported requests.
7. Add request logging that stores prompt length, latency, parse validity, and execution status.

## Current Boundary

This repo is a reproducible QLoRA Text-to-SQL experiment with a guarded local API. It is not a production SQL assistant.
