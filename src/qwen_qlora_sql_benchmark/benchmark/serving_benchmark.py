from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

from qwen_qlora_sql_benchmark.utils.config import load_yaml_config, require_keys


def load_prompts(path: Path, limit: int) -> list[str]:
    prompts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if len(prompts) >= limit:
                break
            payload = json.loads(line)
            prompts.append(payload["prompt"])
    return prompts


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    sorted_values = sorted(values)
    index = max(0, math.ceil(len(sorted_values) * quantile) - 1)
    return sorted_values[index]


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, float | int | str]:
    latencies = [float(row["latency_seconds"]) for row in rows]
    tokens = [int(row["generated_tokens"]) for row in rows]
    total_latency = sum(latencies)
    total_tokens = sum(tokens)
    return {
        "backend": str(rows[0]["backend"]),
        "requests": len(rows),
        "p50_latency_seconds": percentile(latencies, 0.50),
        "p95_latency_seconds": percentile(latencies, 0.95),
        "total_latency_seconds": total_latency,
        "generated_tokens": total_tokens,
        "tokens_per_second": total_tokens / total_latency if total_latency > 0 else 0.0,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_transformers(
    prompts: list[str], model_name: str, max_new_tokens: int
) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    model.eval()
    rows: list[dict[str, Any]] = []
    for index, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        start = time.perf_counter()
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        latency = time.perf_counter() - start
        generated_tokens = int(output_ids.shape[-1] - inputs["input_ids"].shape[-1])
        rows.append(
            {
                "backend": "transformers",
                "request_index": index,
                "latency_seconds": latency,
                "generated_tokens": generated_tokens,
                "tokens_per_second": generated_tokens / latency if latency > 0 else 0.0,
            }
        )
    return rows


def run_vllm(prompts: list[str], model_name: str, max_new_tokens: int) -> list[dict[str, Any]]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_name,
        dtype="auto",
        gpu_memory_utilization=0.85,
        max_model_len=1024,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)
    rows: list[dict[str, Any]] = []
    for index, prompt in enumerate(prompts):
        start = time.perf_counter()
        outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
        latency = time.perf_counter() - start
        generated_tokens = len(outputs[0].outputs[0].token_ids)
        rows.append(
            {
                "backend": "vllm",
                "request_index": index,
                "latency_seconds": latency,
                "generated_tokens": generated_tokens,
                "tokens_per_second": generated_tokens / latency if latency > 0 else 0.0,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run base model serving benchmark.")
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark_vllm.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    require_keys(
        config,
        [
            "model.name",
            "benchmark.prompts_path",
            "benchmark.max_new_tokens",
            "benchmark.iterations",
            "outputs.raw_path",
            "outputs.summary_path",
        ],
    )
    prompts = load_prompts(
        Path(config["benchmark"]["prompts_path"]), int(config["benchmark"]["iterations"])
    )
    max_new_tokens = int(config["benchmark"]["max_new_tokens"])
    model_name = config["model"]["name"]
    raw_rows = run_transformers(prompts, model_name, max_new_tokens)
    raw_rows.extend(run_vllm(prompts, model_name, max_new_tokens))
    write_csv(Path(config["outputs"]["raw_path"]), raw_rows)
    summary_rows = [
        summarize_rows([row for row in raw_rows if row["backend"] == "transformers"]),
        summarize_rows([row for row in raw_rows if row["backend"] == "vllm"]),
    ]
    write_csv(Path(config["outputs"]["summary_path"]), summary_rows)


if __name__ == "__main__":
    main()
