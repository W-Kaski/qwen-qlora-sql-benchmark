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

## Experiments

| Experiment | Config | LoRA rank | LoRA alpha |
| --- | --- | ---: | ---: |
| baseline | `configs/train_baseline.yaml` | none | none |
| rank 8 | `configs/train_lora_r8.yaml` | 8 | 16 |
| rank 16 | `configs/train_lora_r16.yaml` | 16 | 32 |
| rank 32 | `configs/train_lora_r32.yaml` | 32 | 64 |

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
