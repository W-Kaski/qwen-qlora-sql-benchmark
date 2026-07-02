from qwen_qlora_sql_benchmark.benchmark.serving_benchmark import percentile, summarize_rows


def test_percentile_returns_nearest_rank_value() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.95) == 4.0


def test_summarize_rows_computes_latency_and_throughput() -> None:
    summary = summarize_rows(
        [
            {"backend": "x", "latency_seconds": 1.0, "generated_tokens": 10},
            {"backend": "x", "latency_seconds": 2.0, "generated_tokens": 20},
        ]
    )

    assert summary["requests"] == 2
    assert summary["p50_latency_seconds"] == 1.0
    assert summary["p95_latency_seconds"] == 2.0
    assert summary["tokens_per_second"] == 10.0
