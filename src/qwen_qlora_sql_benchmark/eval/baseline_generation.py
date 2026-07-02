from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qwen_qlora_sql_benchmark.utils.config import load_yaml_config, require_keys


@dataclass(frozen=True)
class EvalRecord:
    id: str
    prompt: str
    reference: str


def load_eval_records(path: Path) -> list[EvalRecord]:
    records: list[EvalRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number + 1} must contain a JSON object")
            for key in ("prompt", "completion"):
                if key not in payload:
                    raise KeyError(f"{path}:{line_number + 1}.{key}")
                if not isinstance(payload[key], str):
                    raise TypeError(f"{path}:{line_number + 1}.{key} must be a string")
            records.append(
                EvalRecord(
                    id=str(line_number),
                    prompt=payload["prompt"].strip(),
                    reference=payload["completion"].strip(),
                )
            )
    return records


def strip_generated_sql(prompt: str, generated_text: str) -> str:
    stripped = generated_text.strip()
    if stripped.startswith(prompt):
        stripped = stripped[len(prompt) :]
    return stripped.strip()


def build_prediction_record(record: EvalRecord, prediction: str) -> dict[str, str]:
    return {
        "id": record.id,
        "prompt": record.prompt,
        "prediction": prediction.strip(),
        "reference": record.reference,
    }


def write_prediction_records(path: Path, records: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def generate_with_transformers(
    records: list[EvalRecord],
    model_name: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[dict[str, str]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
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
    parser = argparse.ArgumentParser(description="Run baseline generation with Transformers.")
    parser.add_argument("--config", type=Path, default=Path("configs/train_baseline.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config: dict[str, Any] = load_yaml_config(args.config)
    require_keys(
        config,
        [
            "model.name",
            "data.eval_path",
            "generation.max_new_tokens",
            "generation.temperature",
            "generation.top_p",
            "outputs.predictions_path",
        ],
    )

    records = load_eval_records(Path(config["data"]["eval_path"]))
    predictions = generate_with_transformers(
        records=records,
        model_name=config["model"]["name"],
        max_new_tokens=int(config["generation"]["max_new_tokens"]),
        temperature=float(config["generation"]["temperature"]),
        top_p=float(config["generation"]["top_p"]),
    )
    write_prediction_records(Path(config["outputs"]["predictions_path"]), predictions)


if __name__ == "__main__":
    main()
