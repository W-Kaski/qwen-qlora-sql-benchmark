---
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
pipeline_tag: text-generation
license: apache-2.0
datasets:
  - b-mc2/sql-create-context
language:
  - en
tags:
  - peft
  - lora
  - qlora
  - text-to-sql
  - qwen2.5
  - sqlite
  - schema-conditioned-generation
---

# Qwen2.5-1.5B Text-to-SQL QLoRA Adapter — Rank 32

A PEFT/QLoRA adapter for schema-conditioned Text-to-SQL generation, trained on top of `Qwen/Qwen2.5-1.5B-Instruct`.

> **This repository contains a LoRA adapter, not a standalone language model.**
>
> Load the base model first, then attach this adapter with `PeftModel`.

## Model Summary

This adapter generates one SQL query from an explicit SQL schema and one natural-language question. It was developed as part of a reproducible parameter-efficient fine-tuning benchmark with LoRA rank ablations, normalized Exact Match evaluation, SQL parsing checks, structured error analysis, controlled SQLite execution tests, and a local API demo.

This model is intended for controlled experiments and demonstrations. It is not a production database agent.

## Quick Start

### Installation

```bash
pip install torch transformers peft accelerate safetensors
```

A GPU is recommended.

Some hosted notebook environments may include an incompatible preinstalled version of `torchao`. If adapter loading fails with a `torchao` version error, remove the incompatible package and restart the runtime:

```bash
pip uninstall -y torchao
```

### Inference

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_ID = "W-Kaski/qwen25-15b-text2sql-lora-r32"

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_ID,
    use_fast=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
)

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_ID,
)
model.eval()


def build_prompt(schema: str, question: str) -> str:
    return (
        "You are a Text-to-SQL assistant.\n"
        "Use the SQL schema to answer the question with one SQL query.\n\n"
        f"Schema:\n{schema.strip()}\n\n"
        f"Question:\n{question.strip()}\n\n"
        "SQL:"
    )


schema = """
CREATE TABLE employees (
    id INTEGER,
    name TEXT,
    department TEXT,
    salary INTEGER
);
"""

question = (
    "What is the average salary of employees "
    "in the engineering department?"
)

prompt = build_prompt(schema, question)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.inference_mode():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
sql = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
print(sql)
```

Example output:

```sql
SELECT AVG(salary)
FROM employees
WHERE department = 'engineering';
```

Generated SQL may differ textually while remaining semantically equivalent.

## Input Format

The adapter was trained with the following prompt format:

```text
You are a Text-to-SQL assistant.
Use the SQL schema to answer the question with one SQL query.

Schema:
<SQL schema>

Question:
<natural-language question>

SQL:
```

Using a substantially different prompt format may reduce performance.

## Intended Use

Appropriate uses include:

- controlled Text-to-SQL experiments;
- evaluation of PEFT/QLoRA adaptation;
- schema-conditioned SQL demonstrations;
- sandboxed SQLite examples;
- research on SQL generation metrics and failure analysis.

## Out-of-Scope Use

This adapter is not intended for:

- unrestricted access to production databases;
- automatically executing generated SQL without validation;
- safety-critical or financial database workflows;
- arbitrary enterprise schemas without additional evaluation;
- relying on generated SQL without human or programmatic verification.

## Training Details

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Dataset: `b-mc2/sql-create-context`
- Source fields: `answer`, `question`, `context`
- Training rows: 5,000
- Evaluation rows: 500
- Fine-tuning method: QLoRA
- Quantization: 4-bit NF4
- LoRA rank: 32
- LoRA alpha: 64
- LoRA dropout: 0.05
- Epochs: 1
- Maximum sequence length: 1,024
- Loss target: completion tokens only
- Training and generation seed: 42

## Evaluation

### Main Evaluation Split

The main evaluation uses a fixed 500-example split.

| Model | Normalized Exact Match | SQL Parse Valid |
| --- | ---: | ---: |
| Base model | 0.044 | 0.980 |
| LoRA rank 8 | 0.684 | 0.992 |
| LoRA rank 16 | 0.696 | 0.994 |
| LoRA rank 32 | **0.712** | 0.990 |

Normalized Exact Match compares normalized SQL strings. It may reject semantically equivalent SQL queries.

SQL Parse Valid measures whether generated output can be parsed as SQL. It does not establish semantic or execution correctness.

### Controlled SQLite Execution Check

A separate set of 30 manually curated SQLite cases was used as an execution sanity check.

| Metric | Value |
| --- | ---: |
| Cases | 30 |
| Parse-valid rate | 1.000 |
| Read-only `SELECT` rate | 1.000 |
| Execution-valid rate | 1.000 |
| Execution accuracy | 0.600 |
| P50 latency | 0.456 s |
| P95 latency | 1.182 s |

These 30 cases are not a held-out benchmark and should not be interpreted as general Text-to-SQL accuracy.

## Known Limitations

Observed failure modes include:

- incorrect selected columns;
- value-normalization errors;
- case-sensitive string mismatches;
- incorrect string predicates;
- imperfect `NULL` handling;
- incorrect `GROUP BY` or `HAVING` clauses;
- `LIMIT` and `OFFSET` errors;
- dataset-specific formatting behavior.

The primary dataset is dominated by relatively simple, mostly single-table SQL. Performance should not be extrapolated to complex multi-table enterprise schemas.

## Safe SQL Execution

Do not directly execute generated SQL against a production database.

At minimum, use:

- SQL parsing;
- read-only statement checks;
- table and column allowlists;
- restricted setup statements;
- query timeouts;
- row limits;
- isolated database credentials;
- sandboxed execution.

The accompanying GitHub repository includes a controlled SQLite execution sandbox and a guarded FastAPI demo.

## Reproducibility

Training, evaluation, error analysis, API code, SQLite checks, and configuration files are available in the project repository:

`https://github.com/W-Kaski/qwen-qlora-sql-benchmark`

The repository includes:

- YAML experiment configurations;
- dataset preparation scripts;
- rank 8/16/32 training scripts;
- baseline and adapter generation;
- normalized Exact Match and SQL parsing evaluation;
- structured error analysis;
- SQLite execution tests;
- local serving checks;
- unit tests and lint configuration.

Full rank-ablation results were produced on a local WSL GPU. Kaggle validation covered environment setup, tests, lint, dataset preparation, and GPU runtime availability, not the reported full training runs.

## License

This adapter is released under the Apache 2.0 license. The base model and training dataset retain their respective licenses and terms.
