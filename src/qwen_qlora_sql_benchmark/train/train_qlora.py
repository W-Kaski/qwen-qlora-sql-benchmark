from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qwen_qlora_sql_benchmark.utils.config import load_yaml_config, require_keys


@dataclass(frozen=True)
class PromptCompletionExample:
    prompt: str
    completion: str


def load_prompt_completion_examples(
    path: Path, limit: int | None = None
) -> list[PromptCompletionExample]:
    examples: list[PromptCompletionExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if limit is not None and len(examples) >= limit:
                break
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            for key in ("prompt", "completion"):
                if key not in payload:
                    raise KeyError(f"{path}:{line_number}.{key}")
                if not isinstance(payload[key], str):
                    raise TypeError(f"{path}:{line_number}.{key} must be a string")
                if not payload[key].strip():
                    raise ValueError(f"{path}:{line_number}.{key} must not be blank")
            examples.append(
                PromptCompletionExample(
                    prompt=payload["prompt"].strip(),
                    completion=payload["completion"].strip(),
                )
            )
    return examples


def build_completion_only_features(
    tokenizer: Any,
    example: PromptCompletionExample,
    max_seq_length: int,
) -> dict[str, list[int]]:
    eos_token = tokenizer.eos_token or ""
    prompt_ids = tokenizer(example.prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(
        example.completion + eos_token,
        add_special_tokens=False,
    )["input_ids"]
    input_ids = (prompt_ids + completion_ids)[:max_seq_length]
    labels = ([-100] * len(prompt_ids) + completion_ids)[:max_seq_length]
    attention_mask = [1] * len(input_ids)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def validate_training_config(config: dict[str, Any]) -> None:
    require_keys(
        config,
        [
            "model.name",
            "data.train_path",
            "data.eval_path",
            "training.completion_only_loss",
            "training.max_seq_length",
            "training.per_device_train_batch_size",
            "training.gradient_accumulation_steps",
            "training.learning_rate",
            "lora.r",
            "lora.alpha",
            "lora.dropout",
            "lora.target_modules",
            "outputs.adapter_dir",
            "outputs.log_path",
        ],
    )
    if config["training"]["completion_only_loss"] is not True:
        raise ValueError("training.completion_only_loss must be true")
    if int(config["training"]["max_seq_length"]) <= 0:
        raise ValueError("training.max_seq_length must be positive")
    if int(config["lora"]["r"]) <= 0:
        raise ValueError("lora.r must be positive")


class CompletionOnlyDataCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        max_length = max(len(feature["input_ids"]) for feature in features)
        pad_token_id = self.tokenizer.pad_token_id
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            pad_length = max_length - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [pad_token_id] * pad_length)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad_length)
            batch["labels"].append(feature["labels"] + [-100] * pad_length)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def write_training_event(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def run_training(config: dict[str, Any]) -> None:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )

    validate_training_config(config)
    train_limit = config.get("diagnostic", {}).get("max_train_samples")
    eval_limit = config.get("diagnostic", {}).get("max_eval_samples")
    train_examples = load_prompt_completion_examples(
        Path(config["data"]["train_path"]), train_limit
    )
    eval_examples = load_prompt_completion_examples(Path(config["data"]["eval_path"]), eval_limit)

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_seq_length = int(config["training"]["max_seq_length"])
    train_dataset = [
        build_completion_only_features(tokenizer, example, max_seq_length)
        for example in train_examples
    ]
    eval_dataset = [
        build_completion_only_features(tokenizer, example, max_seq_length)
        for example in eval_examples
    ]

    quantization = config.get("quantization", {})
    compute_dtype_name = quantization.get("bnb_4bit_compute_dtype", "bfloat16")
    compute_dtype = getattr(torch, compute_dtype_name)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=bool(quantization.get("load_in_4bit", True)),
        bnb_4bit_quant_type=quantization.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=int(config["lora"]["r"]),
        lora_alpha=int(config["lora"]["alpha"]),
        lora_dropout=float(config["lora"]["dropout"]),
        target_modules=list(config["lora"]["target_modules"]),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    output_dir = Path(config["outputs"]["adapter_dir"])
    max_steps = config.get("diagnostic", {}).get("max_steps", -1)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(config["training"]["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(config["training"]["gradient_accumulation_steps"]),
        num_train_epochs=float(config["training"].get("num_train_epochs", 1)),
        max_steps=int(max_steps),
        learning_rate=float(config["training"]["learning_rate"]),
        warmup_ratio=float(config["training"].get("warmup_ratio", 0.1)),
        logging_steps=int(config["training"].get("logging_steps", 10)),
        save_steps=int(config["training"].get("save_steps", 100)),
        eval_steps=int(config["training"].get("eval_steps", 100)),
        eval_strategy="steps",
        save_strategy="steps",
        report_to=[],
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CompletionOnlyDataCollator(tokenizer),
    )

    log_path = Path(config["outputs"]["log_path"])
    write_training_event(
        log_path,
        {
            "event": "start",
            "train_examples": len(train_dataset),
            "eval_examples": len(eval_dataset),
            "max_steps": int(max_steps),
            "adapter_dir": str(output_dir),
        },
    )
    train_result = trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    metrics = dict(train_result.metrics)
    write_training_event(log_path, {"event": "finished", "metrics": metrics})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QLoRA training from a YAML config.")
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    run_training(config)


if __name__ == "__main__":
    main()
