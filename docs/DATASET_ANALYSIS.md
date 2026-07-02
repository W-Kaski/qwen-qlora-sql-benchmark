# Dataset Analysis

## Dataset

Dataset: `b-mc2/sql-create-context`

Current local split:

| Split | Rows |
| --- | ---: |
| train | 5000 |
| eval | 500 |
| benchmark prompts | 100 |

## Field Contract

Verified source fields:

- `answer`
- `question`
- `context`

Converted training fields:

- `prompt`
- `completion`

## Duplicate And Overlap Checks

| Check | Result |
| --- | ---: |
| train duplicate prompts | 0 |
| train duplicate completions | 0 |
| eval duplicate prompts | 0 |
| eval duplicate completions | 0 |
| train/eval prompt overlap | 0 |
| train/eval completion overlap | 0 |

## Shape

| Metric | Train | Eval |
| --- | ---: | ---: |
| average prompt words | 39.31 | 39.67 |
| average SQL words | 10.63 | 10.95 |
| average tables per prompt | 1.03 | 1.03 |
| max tables per prompt | 5 | 3 |
| average columns per prompt | 2.35 | 2.38 |
| max columns per prompt | 10 | 7 |

## SQL Pattern Distribution

| Pattern | Train | Eval |
| --- | ---: | ---: |
| `select` | 5000 | 500 |
| `where` | 4825 | 482 |
| `count` | 728 | 74 |
| `min` | 344 | 42 |
| `max` | 288 | 29 |
| `avg` | 214 | 21 |
| `sum` | 172 | 28 |
| `join` | 111 | 14 |
| `group by` | 60 | 9 |
| `order by` | 72 | 8 |
| `limit` | 55 | 6 |
| `offset` | 36 | 7 |

## Schema Naming

Across train and eval, table names are mostly generic:

| Table name type | Mentions |
| --- | ---: |
| `table_name_N` | 3756 |
| `table_N_N` | 1432 |
| semantic name | 487 |

This means the task is mostly schema-conditioned SQL formatting and column/value selection, not domain-specific database reasoning.

## Baseline Behavior

Current baseline result:

| Model | Eval rows | Exact Match | Wall time |
| --- | ---: | ---: | ---: |
| Qwen2.5-1.5B-Instruct | 500 | 0.044 | 185.80 seconds |

Baseline shape checks:

| Check | Count |
| --- | ---: |
| predictions starting with `SELECT` | 500 |
| empty predictions | 0 |
| predictions ending with semicolon | 207 |
| unique predictions | 500 |
| duplicate predictions | 0 |

The baseline is not producing one fixed response. It generates prompt-specific SQL, usually using the supplied schema names and columns. Its main failure mode is exact query mismatch: wrong aggregation, wrong condition value, wrong operator, missing `OFFSET`, or adding unnecessary grouping.

## Expected Fine-Tuning Effect

QLoRA can plausibly improve:

- SQL output format consistency.
- Use of exact column names from schema.
- Mapping question wording to `SELECT`, `WHERE`, and aggregate patterns.
- Matching the dataset's preferred quoting and SQL style.

QLoRA will not prove:

- robust real-world Text-to-SQL generalization;
- database execution correctness without executable database files;
- strong multi-table reasoning, because joins are rare in this split;
- semantic equivalence when Exact Match rejects equivalent SQL variants.

## Risk

The dataset is useful for a first resource-constrained QLoRA experiment, but it is not enough by itself for a strong claim that the model is generally good at Text-to-SQL. The first release should describe the task as small-data schema-conditioned SQL generation, with Exact Match as the initial metric and Execution Accuracy as a later extension only after database execution is verified.
