# Evaluation Specification

## Metrics

First release:

- Exact Match
- SQL validity
- Invalid SQL rate

Second step after database execution is verified:

- Execution Accuracy
- Execution failure rate

## Artifact Contract

Prediction files use JSONL:

```json
{"id": "...", "prompt": "...", "prediction": "...", "reference": "..."}
```

Summary tables:

- `results/tables/baseline_eval.csv`
- `results/tables/eval_summary.csv`

Raw outputs:

- `results/eval_outputs/base_model_predictions.jsonl`
- `results/eval_outputs/lora_r8_predictions.jsonl`
- `results/eval_outputs/lora_r16_predictions.jsonl`
- `results/eval_outputs/lora_r32_predictions.jsonl`

## Baseline Command

```bash
scripts/kaggle_baseline.sh
```

This command writes base model predictions first, then scores Exact Match.

## Diagnostic Adapter Command

```bash
scripts/local_eval_r8_diagnostic.sh
```

This command evaluates `outputs/adapters/lora_r8_diagnostic` on the first 50 eval rows.

## Controls

- Use the same eval prompts for every experiment.
- Keep generation parameters fixed across baseline and adapters.
- Save raw predictions before creating summaries.
