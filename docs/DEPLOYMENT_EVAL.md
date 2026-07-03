# Deployment Evaluation

## Scope

This evaluation tests whether the rank 32 adapter behaves well enough for a controlled Text-to-SQL demo. Unlike Exact Match evaluation, each prediction is executed against an in-memory SQLite database and compared with the reference SQL execution result.

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

The adapter is reliable at producing parseable read-only SQL for this deployment-style set. All 30 generations were valid single-statement `SELECT` queries and all executed without SQLite errors.

The 60% execution accuracy shows that the adapter is not ready for open-ended user deployment. The remaining errors are semantic rather than syntactic:

- case-sensitive value mismatch, such as `Engineering` vs `engineering`
- selecting the wrong column, such as `item` instead of `category`
- choosing the wrong predicate, such as `country = "San"` instead of `name LIKE "San%"`
- returning an aggregate instead of the grouped key
- incomplete handling of `NULL`
- `LIMIT/OFFSET` mistakes

## Deployment Decision

This is enough for a portfolio demo with clear scope and validation metadata. It is not enough for a general SQL assistant where users expect reliable answers across arbitrary schemas.

The next quality step is not another small LoRA rank run. The next step should use an execution-backed benchmark and add value normalization, dialect constraints, and richer training examples for joins, grouping, null handling, and string predicates.
