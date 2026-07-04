# Evaluation Analysis

## Metrics

Current evaluation uses:

- Exact Match
- SQL parse validity with `sqlglot`
- starts-with-`SELECT` rate
- empty prediction rate

Execution Accuracy is not reported because this dataset split does not include executable database files.

## Result Summary

| Model | Rows | Exact Match | SQL Parse Valid | Starts SELECT | Empty |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 500 | 0.044 | 0.980 | 1.000 | 0.000 |
| LoRA rank 8 | 500 | 0.684 | 0.992 | 1.000 | 0.000 |
| LoRA rank 16 | 500 | 0.696 | 0.994 | 1.000 | 0.000 |
| LoRA rank 32 | 500 | 0.712 | 0.990 | 1.000 | 0.000 |

## Interpretation

The base model already generates SQL-shaped output consistently: every baseline prediction starts with `SELECT`, and the parse-valid rate is 0.980. QLoRA therefore does not mainly teach the model to produce SQL syntax. The improvement is in matching this dataset's expected query structure, selected columns, aggregation choices, filters, and SQL style.

Rank 32 currently has the best Exact Match result, but the gap over rank 16 is modest. Rank 8 remains a strong lower-cost setting.

## Limitations

Exact Match rejects semantically equivalent SQL variants. SQL parse validity does not prove that the query executes against a database or returns the correct result. Database execution correctness is reported only in the SQLite execution evaluation.
