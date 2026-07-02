# Experiment Log

## Template

### YYYY-MM-DD experiment-name

- Config:
- Kaggle notebook:
- GPU:
- Dataset snapshot:
- Command:
- Output adapter:
- Output predictions:
- Metrics:
- Observations:
- Decision:

### 2026-07-02 dataset-field-verification

- Config: `configs/dataset_sql_create_context.yaml`
- Dataset: `b-mc2/sql-create-context`
- Verified fields: `answer`, `question`, `context`
- Verification command: `uv run --with datasets python - <<'PY' ...`
- Conversion check: wrote 3 train rows, 2 eval rows, and 1 benchmark row to `/tmp/qwen_sql_splits_check`
- Observations: output JSONL uses `prompt` and `completion`; benchmark JSONL uses `prompt`
- Decision: proceed with this dataset for the first diagnostic training track

### 2026-07-02 baseline-generation-scaffold

- Config: `configs/train_baseline.yaml`
- Command: `scripts/kaggle_baseline.sh`
- Local validation: generated a two-row synthetic prediction file from `data/splits/eval.jsonl`
- Local score check: `count=2`, `exact_match=0.5`
- Observations: baseline script writes raw predictions before Exact Match scoring
- Decision: run full baseline generation on Kaggle or another Linux GPU runtime

### 2026-07-02 baseline-generation-full-local

- Config: `configs/train_baseline.yaml`
- Runtime: local WSL GPU
- Command: `env PYTHONPATH=src python3 -m qwen_qlora_sql_benchmark.eval.baseline_generation --config configs/train_baseline.yaml`
- Output predictions: `results/eval_outputs/base_model_predictions.jsonl`
- Output summary: `results/tables/baseline_eval.csv`
- Wall time: 185.80 seconds
- Metrics: `count=500`, `exact_match=0.044`
- Observations: first-pass base model predictions are often structurally close but fail exact string matching
- Decision: use this result as the base quality comparison before QLoRA diagnostic training

### 2026-07-02 lora-r8-diagnostic-local

- Config: `configs/train_lora_r8_diagnostic.yaml`
- Runtime: local WSL GPU
- Command: `scripts/local_train_r8_diagnostic.sh`
- Output adapter: `outputs/adapters/lora_r8_diagnostic`
- Output log: `results/logs/lora_r8_diagnostic_train.jsonl`
- Train rows: 200
- Eval rows: 50
- Max steps: 30
- Wall time: 117.66 seconds
- Trainer runtime: 100.9827 seconds
- First train loss: 1.499
- Last train loss: 0.0519
- Final eval loss: 0.14725497364997864
- Grad norm range observed in logs: approximately 0.3998 to 3.2584
- Decision: proceed to adapter diagnostic evaluation

### 2026-07-02 lora-r8-diagnostic-eval-local

- Config: `configs/eval_lora_r8_diagnostic.yaml`
- Runtime: local WSL GPU
- Command: `scripts/local_eval_r8_diagnostic.sh`
- Output predictions: `results/eval_outputs/lora_r8_diagnostic_predictions.jsonl`
- Output summary: `results/tables/lora_r8_diagnostic_eval.csv`
- Eval rows: 50
- Wall time: 64.98 seconds
- Metrics: `count=50`, `exact_match=0.48`
- Baseline on same first 50 rows: `exact_match=0.04`
- Shape check: 50 unique predictions, 50 predictions starting with `SELECT`, 0 empty predictions
- Observations: strong early signal, but diagnostic used repeated passes over 200 train rows
- Decision: proceed to full rank 8 run with 1 epoch over the 5000-row train split

### 2026-07-02 lora-r8-full-local

- Config: `configs/train_lora_r8.yaml`
- Runtime: local WSL GPU
- Command: `env PYTHONPATH=src python3 -m qwen_qlora_sql_benchmark.train.train_qlora --config configs/train_lora_r8.yaml`
- Output adapter: `outputs/adapters/lora_r8`
- Output log: `results/logs/lora_r8_train.jsonl`
- Train rows: 5000
- Eval rows: 500
- Epochs: 1
- Wall time: 1095.10 seconds
- Trainer runtime: 1085.7485 seconds
- First logged train loss: 1.3266
- Last logged train loss: 0.0484
- Final logged eval loss: 0.059545669704675674
- Decision: evaluate full rank 8 adapter on the 500-row eval split

### 2026-07-02 lora-r8-full-eval-local

- Config: `configs/eval_lora_r8.yaml`
- Runtime: local WSL GPU
- Command: `scripts/local_eval_r8.sh`
- Output predictions: `results/eval_outputs/lora_r8_predictions.jsonl`
- Output summary: `results/tables/lora_r8_eval.csv`
- Eval rows: 500
- Wall time: 568.25 seconds
- Metrics: `count=500`, `exact_match=0.684`
- Baseline on same 500 rows: `exact_match=0.044`
- Shape check: 500 unique predictions, 500 predictions starting with `SELECT`, 0 empty predictions
- Decision: proceed to rank 16 diagnostic after committing rank 8 artifacts

### 2026-07-02 lora-r16-diagnostic-local

- Config: `configs/train_lora_r16_diagnostic.yaml`
- Runtime: local WSL GPU
- Command: `scripts/local_train_r16_diagnostic.sh`
- Output adapter: `outputs/adapters/lora_r16_diagnostic`
- Train rows: 200
- Eval rows: 50
- Max steps: 30
- Wall time: 117.60 seconds
- Trainer runtime: 103.7101 seconds
- First train loss: 1.499
- Last train loss: 0.0275
- Final eval loss: 0.13401997089385986
- Decision: proceed to rank 16 full run

### 2026-07-02 lora-r16-diagnostic-eval-local

- Config: `configs/eval_lora_r16_diagnostic.yaml`
- Runtime: local WSL GPU
- Command: `scripts/local_eval_r16_diagnostic.sh`
- Output summary: `results/tables/lora_r16_diagnostic_eval.csv`
- Eval rows: 50
- Wall time: 66.52 seconds
- Metrics: `count=50`, `exact_match=0.44`
- Decision: proceed to full rank 16 despite diagnostic Exact Match being slightly below rank 8 diagnostic

### 2026-07-02 lora-r16-full-local

- Config: `configs/train_lora_r16.yaml`
- Runtime: local WSL GPU
- Command: `env PYTHONPATH=src python3 -m qwen_qlora_sql_benchmark.train.train_qlora --config configs/train_lora_r16.yaml`
- Output adapter: `outputs/adapters/lora_r16`
- Output log: `results/logs/lora_r16_train.jsonl`
- Train rows: 5000
- Eval rows: 500
- Epochs: 1
- Wall time: 1132.98 seconds
- Trainer runtime: 1123.2279 seconds
- First logged train loss: 1.2632
- Last logged train loss: 0.0438
- Final logged eval loss: 0.05369124561548233
- Decision: evaluate full rank 16 adapter on the 500-row eval split

### 2026-07-02 lora-r16-full-eval-local

- Config: `configs/eval_lora_r16.yaml`
- Runtime: local WSL GPU
- Command: `scripts/local_eval_r16.sh`
- Output predictions: `results/eval_outputs/lora_r16_predictions.jsonl`
- Output summary: `results/tables/lora_r16_eval.csv`
- Eval rows: 500
- Wall time: 657.88 seconds
- Metrics: `count=500`, `exact_match=0.696`
- Baseline on same 500 rows: `exact_match=0.044`
- Rank 8 on same 500 rows: `exact_match=0.684`
- Decision: proceed to rank 32 diagnostic to complete rank ablation
