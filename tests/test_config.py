from pathlib import Path

import pytest
import yaml

from qwen_qlora_sql_benchmark.utils.config import load_yaml_config, require_keys


def test_load_yaml_config_returns_mapping(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  name: Qwen/Qwen2.5-1.5B-Instruct\n", encoding="utf-8")

    config = load_yaml_config(config_path)

    assert config == {"model": {"name": "Qwen/Qwen2.5-1.5B-Instruct"}}


def test_load_yaml_config_rejects_non_mapping(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("- model\n- train\n", encoding="utf-8")

    with pytest.raises(ValueError, match="top-level mapping"):
        load_yaml_config(config_path)


def test_require_keys_reports_missing_path() -> None:
    config = {"model": {"name": "Qwen/Qwen2.5-1.5B-Instruct"}}

    with pytest.raises(KeyError, match="training.learning_rate"):
        require_keys(config, ["model.name", "training.learning_rate"])


def test_yaml_test_fixture_is_valid() -> None:
    payload = yaml.safe_load("training:\n  completion_only_loss: true\n")

    assert payload["training"]["completion_only_loss"] is True


def test_lora_training_configs_define_seed() -> None:
    for config_path in Path("configs").glob("train_lora*.yaml"):
        config = load_yaml_config(config_path)
        seed = config["training"]["seed"]

        assert isinstance(seed, int), config_path
        assert seed >= 0, config_path


def test_generation_configs_define_seed() -> None:
    for config_path in Path("configs").glob("*.yaml"):
        config = load_yaml_config(config_path)
        if "generation" not in config:
            continue
        seed = config["generation"]["seed"]

        assert isinstance(seed, int), config_path
        assert seed >= 0, config_path
