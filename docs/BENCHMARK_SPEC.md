# Benchmark Specification

## First Release Scope

Benchmark base model serving only:

- vLLM base model
- Transformers base model

LoRA serving through vLLM is optional and must not block the first release.

## Metrics

- time to first token
- total latency
- tokens per second
- P50 latency
- P95 latency
- peak GPU memory

## Concurrency

Run small concurrency levels first:

- 1
- 2
- 4

## Artifact Contract

- `results/tables/serving_benchmark_raw.csv`
- `results/tables/serving_benchmark.csv`
- `results/figures/latency_comparison.png`
- `results/figures/throughput_comparison.png`

## Current Local Result

| Backend | Requests | P50 latency | P95 latency | Tokens/s |
| --- | ---: | ---: | ---: | ---: |
| Transformers | 20 | 0.3144s | 0.5359s | 76.18 |
| vLLM | 20 | 0.2156s | 0.4098s | 74.61 |

This is a sequential single-request benchmark on local WSL, not a high-concurrency serving study.
