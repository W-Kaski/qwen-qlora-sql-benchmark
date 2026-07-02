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
