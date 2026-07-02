from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from qwen_qlora_sql_benchmark.eval.baseline_generation import (
    EvalRecord,
    build_prediction_record,
    load_eval_records,
    strip_generated_sql,
    write_prediction_records,
)
from qwen_qlora_sql_benchmark.utils.config import load_yaml_config, require_keys


def resolve_eval_limit(config: dict[str, Any]) -> int | None:
    limit = config.get("diagnostic", {}).get("max_eval_samples")
    if limit is None:
        return None
    return int(limit)


def validate_adapter_eval_config(config: dict[str, Any]) -> None:
    require_keys(
        config,
        [
            "model.name",
            "adapter.path",
            "data.eval_path",
            "generation.max_new_tokens",
            "generation.temperature",
            "generation.top_p",
            "outputs.predictions_path",
        ],
    )
    adapter_path = Path(config["adapter"]["path"])
    if not adapter_path.exists():
        raise FileNotFoundError(adapter_path)


def generate_with_peft_adapter(
    records: list[EvalRecord],
    model_name: str,
    adapter_path: Path,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[dict[str, str]]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    do_sample = temperature > 0.0
    predictions: list[dict[str, str]] = []
    for record in records:
        inputs = tokenizer(record.prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        prediction = strip_generated_sql(record.prompt, generated_text)
        predictions.append(build_prediction_record(record, prediction))
    return predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run generation with a PEFT adapter.")
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    validate_adapter_eval_config(config)
    records = load_eval_records(Path(config["data"]["eval_path"]))
    limit = resolve_eval_limit(config)
    if limit is not None:
        records = records[:limit]
    predictions = generate_with_peft_adapter(
        records=records,
        model_name=config["model"]["name"],
        adapter_path=Path(config["adapter"]["path"]),
        max_new_tokens=int(config["generation"]["max_new_tokens"]),
        temperature=float(config["generation"]["temperature"]),
        top_p=float(config["generation"]["top_p"]),
    )
    write_prediction_records(Path(config["outputs"]["predictions_path"]), predictions)


if __name__ == "__main__":
    main()
