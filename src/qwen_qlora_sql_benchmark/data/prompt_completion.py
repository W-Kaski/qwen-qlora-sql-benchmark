from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PromptCompletionRecord:
    prompt: str
    completion: str


def _require_non_empty_string(payload: dict[str, Any], key: str) -> str:
    if key not in payload:
        raise KeyError(key)
    value = payload[key]
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{key} must not be blank")
    return stripped


def validate_prompt_completion_record(payload: dict[str, Any]) -> PromptCompletionRecord:
    return PromptCompletionRecord(
        prompt=_require_non_empty_string(payload, "prompt"),
        completion=_require_non_empty_string(payload, "completion"),
    )
