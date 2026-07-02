# Dataset Specification

## Selected Dataset

`b-mc2/sql-create-context`

The Hugging Face dataset page describes 78,577 Text-to-SQL examples with natural language queries, SQL `CREATE TABLE` context, and SQL answers.

## Verified Fields

The first streamed row was inspected locally on 2026-07-02. The exact keys are:

```text
answer
question
context
```

Example row:

```json
{
  "answer": "SELECT COUNT(*) FROM head WHERE age > 56",
  "question": "How many heads of the departments are older than 56 ?",
  "context": "CREATE TABLE head (age INTEGER)"
}
```

## Conversion

The project converts records to:

```json
{"prompt": "...", "completion": "..."}
```

The prompt contains:

- task instruction
- `context` as schema
- `question` as user request
- `SQL:` completion boundary

The completion is the exact `answer` value after trimming surrounding whitespace.

## Initial Split Sizes

- train: 5000
- eval: 500
- benchmark prompts: 100
- seed: 42

These sizes are intentionally small for Kaggle T4 diagnostic-first training.
