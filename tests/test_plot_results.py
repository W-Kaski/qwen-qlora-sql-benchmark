import pandas as pd

from qwen_qlora_sql_benchmark.visualize.plot_results import lora_eval_rows


def test_lora_eval_rows_filters_full_lora_rows() -> None:
    frame = pd.DataFrame(
        [
            {"model": "baseline", "rank": "baseline", "eval_rows": 500, "exact_match": 0.1},
            {"model": "LoRA r8 diagnostic", "rank": "8", "eval_rows": 50, "exact_match": 0.2},
            {"model": "LoRA r8", "rank": "8", "eval_rows": 500, "exact_match": 0.3},
        ]
    )

    rows = lora_eval_rows(frame)

    assert rows[["rank", "exact_match"]].to_dict("records") == [
        {"rank": 8, "exact_match": 0.3}
    ]
