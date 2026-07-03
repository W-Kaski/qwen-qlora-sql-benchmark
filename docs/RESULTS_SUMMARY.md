# Results Summary

## Rank Ablation

| Model | Exact Match | SQL Parse Valid | Train Wall Time |
| --- | ---: | ---: | ---: |
| baseline | 0.044 | 0.980 | none |
| LoRA rank 8 | 0.684 | 0.992 | 1095.10 seconds |
| LoRA rank 16 | 0.696 | 0.994 | 1132.98 seconds |
| LoRA rank 32 | 0.712 | 0.990 | 1077.81 seconds |

## Interpretation

All LoRA ranks improved Exact Match over the base model. Rank 32 is currently best on Exact Match, while rank 8 is close and remains the lower-parameter setting.

The SQL parse-validity rate is high even for the baseline, which means this experiment is not mainly about teaching SQL syntax. The fine-tuning gain is mostly in matching the dataset's preferred query structure and exact output format.

## Figures

- `results/figures/exact_match_by_rank.png`
- `results/figures/training_time_by_rank.png`
- `results/figures/sql_validity_by_model.png`
