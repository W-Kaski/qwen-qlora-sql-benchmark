from __future__ import annotations

from pathlib import Path
from typing import Protocol

from qwen_qlora_sql_benchmark.eval.baseline_generation import strip_generated_sql


class SqlGenerator(Protocol):
    def generate_sql(self, schema: str, question: str) -> str:
        """Generate one SQL query from a schema and natural-language question."""


def build_text_to_sql_prompt(schema: str, question: str) -> str:
    return (
        "You are a Text-to-SQL assistant.\n"
        "Use the SQL schema to answer the question with one SQL query.\n\n"
        f"Schema:\n{schema.strip()}\n\n"
        f"Question:\n{question.strip()}\n\n"
        "SQL:"
    )


class LazyPeftSqlGenerator:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        adapter_path: Path = Path("outputs/adapters/lora_r32"),
        max_new_tokens: int = 128,
    ) -> None:
        self.model_name = model_name
        self.adapter_path = adapter_path
        self.max_new_tokens = max_new_tokens
        self._tokenizer = None
        self._model = None

    def generate_sql(self, schema: str, question: str) -> str:
        import torch

        tokenizer, model = self._load()
        prompt = build_text_to_sql_prompt(schema, question)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return strip_generated_sql(prompt, generated_text)

    def _load(self):
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, self.adapter_path)
        model.eval()
        self._tokenizer = tokenizer
        self._model = model
        return tokenizer, model
