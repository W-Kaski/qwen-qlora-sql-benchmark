---
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
tags:
  - text-to-sql
  - qlora
  - lora
  - qwen2.5
  - sqlite
license: apache-2.0
---

# Qwen2.5-1.5B Text-to-SQL QLoRA Adapter Rank 32

This adapter was trained for a reproducible Text-to-SQL experiment using `Qwen/Qwen2.5-1.5B-Instruct` as the base model.

## Intended Use

This adapter is intended for controlled Text-to-SQL experiments and demos where the user provides an explicit SQL schema and one natural-language question.

It is not intended for production analytics, arbitrary database access, safety-critical workflows, or unsandboxed query execution.

## Training Setup

- base model: `Qwen/Qwen2.5-1.5B-Instruct`
- dataset: `b-mc2/sql-create-context`
- source fields: `answer`, `question`, `context`
- training rows: 5000
- eval rows: 500
- method: QLoRA
- quantization: 4-bit NF4
- LoRA rank: 32
- LoRA alpha: 64
- LoRA dropout: 0.05
- epochs: 1
- max sequence length: 1024
- training and generation seed: 42

## Results

Evaluation on the 500-row Text-to-SQL split:

| Metric | Value |
| --- | ---: |
| Exact Match | 0.712 |
| SQL parse valid | 0.990 |

SQLite execution evaluation:

| Metric | Value |
| --- | ---: |
| cases | 30 |
| parse valid rate | 1.000 |
| select-only rate | 1.000 |
| execution-valid rate | 1.000 |
| execution accuracy | 0.600 |

## Limitations

The adapter improves dataset-specific SQL formatting and schema-conditioned query construction over the base model, but it is not designed for arbitrary databases.

Known failure modes:

- case-sensitive value mismatch
- wrong selected column
- wrong string predicate
- imperfect `GROUP BY` and `HAVING`
- imperfect `NULL` handling
- `LIMIT/OFFSET` mistakes

Use parse validation, read-only checks, setup-statement restrictions, timeout
protection, and row limits before exposing generated SQL to users.

## Repository

Project repo: https://github.com/W-Kaski/qwen-qlora-sql-benchmark
