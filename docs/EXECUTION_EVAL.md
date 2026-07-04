# SQLite Execution Evaluation

## Scope

This evaluation measures rank 32 adapter behavior on SQLite-backed Text-to-SQL
cases. Unlike Exact Match evaluation, each prediction is executed against an
in-memory SQLite database and compared with the reference SQL execution result.

Evaluation set:

- 30 SQLite-backed cases
- single-table filters
- aggregation
- `GROUP BY`
- `ORDER BY` and `LIMIT`
- joins and subqueries
- `LIKE`, `BETWEEN`, `DISTINCT`, and `NULL`
- case-sensitive string values

Cases are stored in `data/execution_eval/cases.jsonl`.

Run:

```bash
scripts/run_execution_eval.sh
```

## Summary

| Metric | Value |
| --- | ---: |
| cases | 30 |
| parse valid rate | 1.000 |
| select-only rate | 1.000 |
| execution-valid rate | 1.000 |
| execution accuracy | 0.600 |
| P50 latency | 0.456s |
| P95 latency | 1.182s |

Detailed rows are stored in `results/tables/execution_eval.csv`.

## Smoke Test

A smaller smoke test used six hand-written SQLite-backed cases:

- simple single-table filtering
- count aggregation
- max aggregation
- two-condition filtering
- order by with limit
- two-table lookup through a subquery or join

Runtime:

- local WSL GPU
- `Qwen/Qwen2.5-1.5B-Instruct`
- `outputs/adapters/lora_r32`
- 4-bit NF4 base model loading
- greedy decoding
- `max_new_tokens=128`

| Model | Cases | Parse Valid | Execution Match | Average Latency |
| --- | ---: | ---: | ---: | ---: |
| base model | 6 | 6 | 4 | 0.390s |
| LoRA rank 32 | 6 | 6 | 5 | 0.571s |

Detailed rows are stored in `results/tables/execution_smoke_test.csv`.

## Interpretation

All 30 generations were valid single-statement `SELECT` queries and all executed
without SQLite errors.

Execution accuracy was 60%. The remaining errors were semantic rather than
syntactic:

- case-sensitive value mismatch, such as `Engineering` vs `engineering`
- selecting the wrong column, such as `item` instead of `category`
- choosing the wrong predicate, such as `country = "San"` instead of
  `name LIKE "San%"`
- returning an aggregate instead of the grouped key
- incomplete handling of `NULL`
- `LIMIT/OFFSET` mistakes

The adapter reliably produced one SQL query and did not emit explanatory text in
the six-case smoke test. It improved the simple filtering case where the base
model selected all columns instead of the requested column.

## Boundary

Supported:

- local demo with fixed schemas
- batch evaluation workflow
- controlled API where users pass explicit schema and one natural-language
  question

Unsupported:

- arbitrary user-uploaded databases
- production analytics workloads
- multi-turn SQL editing
- safety-critical data work
- real customer deployment without query validation and sandboxing

The current adapter is useful for scoped demonstrations and evaluation work. It
is not accurate enough for arbitrary user databases.

The SQLite helper only executes generated SQL after it passes the single
read-only `SELECT` check. Setup statements are restricted to single
`CREATE TABLE` or `INSERT` statements.

## Next Quality Pass

The next quality pass needs:

1. execution-backed training and evaluation data
2. value normalization
3. dialect constraints
4. more examples for joins, grouping, null handling, and string predicates
5. request logging for prompt length, latency, parse validity, and execution
   status
