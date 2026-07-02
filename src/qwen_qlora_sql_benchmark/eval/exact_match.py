from __future__ import annotations

import re


def normalize_sql(sql: str) -> str:
    normalized = re.sub(r"\s+", " ", sql.strip())
    if normalized.endswith(";"):
        normalized = normalized[:-1]
    return normalized.strip().lower()


def exact_match(prediction: str, reference: str) -> bool:
    return normalize_sql(prediction) == normalize_sql(reference)
