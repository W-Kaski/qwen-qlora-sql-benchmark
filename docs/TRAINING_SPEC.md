# Training Specification

## Base Model

`Qwen/Qwen2.5-1.5B-Instruct`

## Training Method

- Method: QLoRA
- Quantization: 4-bit NF4
- SFT data format: prompt-completion
- Required loss setting: `completion_only_loss: true`
- Epochs: 1 for the first release
- Learning rate: `0.0001`
- Dropout: `0.05`
- Training seed: `42`
- Generation seed: `42`

## Experiments

| Experiment | Config | LoRA rank | LoRA alpha |
| --- | --- | ---: | ---: |
| baseline | `configs/train_baseline.yaml` | none | none |
| rank 8 diagnostic | `configs/train_lora_r8_diagnostic.yaml` | 8 | 16 |
| rank 8 | `configs/train_lora_r8.yaml` | 8 | 16 |
| rank 16 | `configs/train_lora_r16.yaml` | 16 | 32 |
| rank 32 | `configs/train_lora_r32.yaml` | 32 | 64 |

## Parameter Scale

| Rank | Trainable Parameters | Trainable Percent |
| ---: | ---: | ---: |
| 8 | 9,232,384 | 0.517% |
| 16 | 18,464,768 | 1.028% |
| 32 | 36,929,536 | 2.036% |

The full table is stored in `results/tables/parameter_count.csv`.

## Diagnostics

Record these values before full training:

- GPU name
- CUDA availability
- package versions
- peak GPU memory
- seconds per step
- train loss trend
- eval loss trend
- grad norm
- ten-sample generation format check

## Data Contract

Training data must be JSONL with exact keys:

```json
{"prompt": "...", "completion": "..."}
```

Dataset-specific field names must be read from the dataset files before conversion.

## Diagnostic Command

```bash
scripts/local_train_r8_diagnostic.sh
```

This command uses `configs/train_lora_r8_diagnostic.yaml` and limits training to 200 train rows, 50 eval rows, and 30 optimizer steps.

## Latest Diagnostic Observation

The local rank 8 diagnostic completed 30 steps without OOM. Training loss moved from `1.499` to `0.0519`, final eval loss was `0.14725497364997864`, and the diagnostic adapter scored `0.48` Exact Match on the first 50 eval rows compared with `0.04` for the base model on the same rows.

The full local rank 8 run completed 1 epoch over 5000 train rows in `1095.10` wall-clock seconds. It scored `0.684` Exact Match on the 500-row eval split compared with `0.044` for the base model.

The full local rank 16 run completed 1 epoch over 5000 train rows in `1132.98` wall-clock seconds. It scored `0.696` Exact Match on the 500-row eval split.

The full local rank 32 run completed 1 epoch over 5000 train rows in `1077.81`
wall-clock seconds. It scored `0.712` Exact Match on the 500-row eval split,
the highest Exact Match in this single-seed rank ablation.
