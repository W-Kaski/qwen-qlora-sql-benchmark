from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{config_path} must contain a top-level mapping")
    return payload


def require_keys(config: Mapping[str, Any], dotted_paths: Sequence[str]) -> None:
    for dotted_path in dotted_paths:
        current: Any = config
        for part in dotted_path.split("."):
            if not isinstance(current, Mapping) or part not in current:
                raise KeyError(dotted_path)
            current = current[part]
