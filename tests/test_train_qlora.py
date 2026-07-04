import json
from pathlib import Path

import pytest

from qwen_qlora_sql_benchmark.train.train_qlora import (
    PromptCompletionExample,
    build_completion_only_features,
    load_prompt_completion_examples,
    validate_training_config,
)


class TinyTokenizer:
    eos_token = "<eos>"
    pad_token = "<pad>"
    eos_token_id = 0
    pad_token_id = 1

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        del add_special_tokens
        return {"input_ids": [ord(char) for char in text]}


def test_load_prompt_completion_examples_respects_limit(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    rows = [
        {"prompt": "SQL:", "completion": "SELECT 1;"},
        {"prompt": "SQL:", "completion": "SELECT 2;"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    examples = load_prompt_completion_examples(path, limit=1)

    assert examples == [PromptCompletionExample(prompt="SQL:", completion="SELECT 1;")]


def test_build_completion_only_features_masks_prompt_tokens() -> None:
    features = build_completion_only_features(
        TinyTokenizer(),
        PromptCompletionExample(prompt="SQL:", completion="SELECT 1;"),
        max_seq_length=64,
    )

    prompt_length = len("SQL:")
    assert features["labels"][:prompt_length] == [-100] * prompt_length
    assert features["labels"][prompt_length:] == features["input_ids"][prompt_length:]
    assert features["attention_mask"] == [1] * len(features["input_ids"])


def test_build_completion_only_features_truncates_labels_with_input_ids() -> None:
    features = build_completion_only_features(
        TinyTokenizer(),
        PromptCompletionExample(prompt="SQL:", completion="SELECT 1;"),
        max_seq_length=5,
    )

    assert len(features["input_ids"]) == 5
    assert len(features["labels"]) == 5
    assert len(features["attention_mask"]) == 5


def test_validate_training_config_requires_completion_only_loss() -> None:
    config = {
        "model": {"name": "Qwen/Qwen2.5-1.5B-Instruct"},
        "data": {"train_path": "data/splits/train.jsonl", "eval_path": "data/splits/eval.jsonl"},
        "training": {
            "completion_only_loss": False,
            "seed": 42,
            "max_seq_length": 1024,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "learning_rate": 0.0001,
        },
        "lora": {"r": 8, "alpha": 16, "dropout": 0.05, "target_modules": ["q_proj"]},
        "outputs": {"adapter_dir": "outputs/adapters/lora_r8", "log_path": "results/logs/x.jsonl"},
    }

    with pytest.raises(ValueError, match="completion_only_loss"):
        validate_training_config(config)


def test_validate_training_config_rejects_negative_seed() -> None:
    config = {
        "model": {"name": "Qwen/Qwen2.5-1.5B-Instruct"},
        "data": {"train_path": "data/splits/train.jsonl", "eval_path": "data/splits/eval.jsonl"},
        "training": {
            "completion_only_loss": True,
            "seed": -1,
            "max_seq_length": 1024,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "learning_rate": 0.0001,
        },
        "lora": {"r": 8, "alpha": 16, "dropout": 0.05, "target_modules": ["q_proj"]},
        "outputs": {"adapter_dir": "outputs/adapters/lora_r8", "log_path": "results/logs/x.jsonl"},
    }

    with pytest.raises(ValueError, match="training.seed"):
        validate_training_config(config)
