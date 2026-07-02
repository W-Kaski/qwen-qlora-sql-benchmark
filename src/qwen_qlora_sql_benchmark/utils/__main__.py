from __future__ import annotations

from qwen_qlora_sql_benchmark.utils.gpu import collect_runtime_snapshot


def main() -> None:
    print(collect_runtime_snapshot())


if __name__ == "__main__":
    main()
