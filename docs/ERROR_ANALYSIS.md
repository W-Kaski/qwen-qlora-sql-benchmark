# Error Analysis

## Method

The error analysis classifies each prediction against its reference SQL using:

- exact match after whitespace, case, and trailing semicolon normalization
- non-SQL output
- invalid SQL under `sqlglot`
- projection mismatch in the `SELECT` clause
- filter or condition mismatch in the `WHERE` clause
- table or join mismatch
- other SQL mismatch

The report is generated from local prediction artifacts with:

```bash
scripts/analyze_errors.sh
```

## Summary

| Model | Error Type | Count | Rate |
| --- | --- | ---: | ---: |
| baseline | exact_match | 22 | 0.044 |
| baseline | filter_or_condition_mismatch | 368 | 0.736 |
| baseline | invalid_sql | 10 | 0.020 |
| baseline | other_sql_mismatch | 8 | 0.016 |
| baseline | projection_mismatch | 91 | 0.182 |
| baseline | table_or_join_mismatch | 1 | 0.002 |
| LoRA rank 8 | exact_match | 342 | 0.684 |
| LoRA rank 8 | filter_or_condition_mismatch | 82 | 0.164 |
| LoRA rank 8 | invalid_sql | 3 | 0.006 |
| LoRA rank 8 | other_sql_mismatch | 6 | 0.012 |
| LoRA rank 8 | projection_mismatch | 66 | 0.132 |
| LoRA rank 8 | table_or_join_mismatch | 1 | 0.002 |
| LoRA rank 16 | exact_match | 348 | 0.696 |
| LoRA rank 16 | filter_or_condition_mismatch | 77 | 0.154 |
| LoRA rank 16 | invalid_sql | 2 | 0.004 |
| LoRA rank 16 | other_sql_mismatch | 5 | 0.010 |
| LoRA rank 16 | projection_mismatch | 68 | 0.136 |
| LoRA rank 32 | exact_match | 356 | 0.712 |
| LoRA rank 32 | filter_or_condition_mismatch | 73 | 0.146 |
| LoRA rank 32 | invalid_sql | 4 | 0.008 |
| LoRA rank 32 | other_sql_mismatch | 3 | 0.006 |
| LoRA rank 32 | projection_mismatch | 63 | 0.126 |
| LoRA rank 32 | table_or_join_mismatch | 1 | 0.002 |

## Interpretation

The baseline already produces SQL-shaped output, but most non-exact rows fail because the selected filters or conditions do not match the reference. Fine-tuning sharply reduces condition mismatch from 73.6% to 14.6% at rank 32.

Projection mismatch also improves, from 18.2% for the baseline to 12.6% for rank 32. This supports the main finding that QLoRA is learning dataset-specific query structure and column selection, not merely SQL syntax.

Invalid SQL remains low for every model. The remaining gap is mostly semantic and formatting-sensitive under Exact Match. Execution-based checks are handled in the SQLite execution evaluation.

## Limitations

This analysis is heuristic. It compares parsed SQL structure, but it does not prove semantic equivalence or execution correctness.
