import os
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/W-Kaski/qwen-qlora-sql-benchmark.git"
WORKDIR = Path("/kaggle/working/qwen-qlora-sql-benchmark")


def run(command: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    os.environ["UV_SYSTEM_PYTHON"] = "1"
    if not WORKDIR.exists():
        run(["git", "clone", "--depth", "1", REPO_URL, str(WORKDIR)])

    run(["python", "-m", "pip", "install", "-U", "uv"])
    run(["uv", "run", "--extra", "dev", "pytest"], cwd=WORKDIR)
    run(["uv", "run", "--extra", "dev", "ruff", "check", "."], cwd=WORKDIR)
    run(
        [
            "uv",
            "run",
            "--extra",
            "data",
            "python",
            "-m",
            "qwen_qlora_sql_benchmark.data.download_sql_create_context",
        ],
        cwd=WORKDIR,
    )
    run(["uv", "run", "python", "-m", "qwen_qlora_sql_benchmark.utils"], cwd=WORKDIR)
    print("KAGGLE_VALIDATION_OK", flush=True)


if __name__ == "__main__":
    main()
