# Deployment Evaluation

## Scope

This evaluation measures rank 32 adapter behavior on SQLite-backed Text-to-SQL cases. Unlike Exact Match evaluation, each prediction is executed against an in-memory SQLite database and compared with the reference SQL execution result.

Evaluation set:

- 30 SQLite-backed cases
- single-table filters
- aggregation
- `GROUP BY`
- `ORDER BY` and `LIMIT`
- joins and subqueries
- `LIKE`, `BETWEEN`, `DISTINCT`, and `NULL`
- case-sensitive string values

Cases are stored in `data/deployment_eval/cases.jsonl`.

Run:

```bash
scripts/run_deployment_eval.sh
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

Detailed rows are stored in `results/tables/deployment_eval.csv`.

## Interpretation

All 30 generations were valid single-statement `SELECT` queries and all executed without SQLite errors.

Execution accuracy was 60%. The remaining errors were semantic rather than syntactic:

- case-sensitive value mismatch, such as `Engineering` vs `engineering`
- selecting the wrong column, such as `item` instead of `category`
- choosing the wrong predicate, such as `country = "San"` instead of `name LIKE "San%"`
- returning an aggregate instead of the grouped key
- incomplete handling of `NULL`
- `LIMIT/OFFSET` mistakes

## Deployment Boundary

The current adapter is usable for a scoped local demo with explicit schemas and validation metadata. It is not accurate enough for arbitrary user databases.

The next quality pass needs execution-backed training and evaluation data, value normalization, dialect constraints, and more examples for joins, grouping, null handling, and string predicates.
