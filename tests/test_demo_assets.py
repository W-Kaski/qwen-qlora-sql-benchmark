import json
from pathlib import Path


def test_demo_request_matches_api_contract() -> None:
    payload = json.loads(Path("examples/demo_request.json").read_text(encoding="utf-8"))

    assert set(payload) == {
        "schema",
        "question",
        "execute",
        "setup_sql",
        "max_rows",
        "timeout_ms",
    }
    assert payload["execute"] is True
    assert isinstance(payload["setup_sql"], list)
