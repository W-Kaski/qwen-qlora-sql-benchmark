# Benchmark Analysis

## Scope

This benchmark compares base model serving for `Qwen/Qwen2.5-1.5B-Instruct` using:

- Transformers
- vLLM

The benchmark uses 20 single-request prompts from `data/splits/benchmark_prompts.jsonl` with `max_new_tokens=128`.

## Results

| Backend | Requests | P50 latency | P95 latency | Tokens/s |
| --- | ---: | ---: | ---: | ---: |
| Transformers | 20 | 0.3144s | 0.5359s | 76.18 |
| vLLM | 20 | 0.2156s | 0.4098s | 74.61 |

## Interpretation

vLLM has lower P50 and P95 latency in this single-request test. Total tokens per second is similar because this benchmark sends requests one at a time and does not exercise high-concurrency batching, where vLLM is usually expected to show a clearer throughput advantage.

## Limitations

- Local WSL runtime.
- `enforce_eager=True` for vLLM.
- Single-request sequential benchmark.
- No LoRA serving benchmark in the first release.
- No TTFT instrumentation yet.

This benchmark is a base-model serving sanity check, not a full production serving study.
