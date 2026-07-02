from pathlib import Path

from qwen_qlora_sql_benchmark.eval.adapter_generation import (
    resolve_eval_limit,
    validate_adapter_eval_config,
)


def test_resolve_eval_limit_returns_none_when_absent() -> None:
    assert resolve_eval_limit({}) is None


def test_resolve_eval_limit_reads_diagnostic_limit() -> None:
    assert resolve_eval_limit({"diagnostic": {"max_eval_samples": 50}}) == 50


def test_validate_adapter_eval_config_accepts_required_paths(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    config = {
        "model": {"name": "Qwen/Qwen2.5-1.5B-Instruct"},
        "adapter": {"path": str(adapter_dir)},
        "data": {"eval_path": "data/splits/eval.jsonl"},
        "generation": {"max_new_tokens": 256, "temperature": 0.0, "top_p": 1.0},
        "outputs": {"predictions_path": "results/eval_outputs/x.jsonl"},
    }

    validate_adapter_eval_config(config)
