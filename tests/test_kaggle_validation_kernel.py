import json
from pathlib import Path


def test_kaggle_validation_kernel_metadata_is_valid() -> None:
    metadata = json.loads(
        Path("kaggle/validation/kernel-metadata.json").read_text(encoding="utf-8")
    )

    assert metadata["id"] == "ericwang7717/qwen-qlora-sql-validation"
    assert metadata["code_file"] == "validation.py"
    assert metadata["language"] == "python"
    assert metadata["kernel_type"] == "script"
    assert metadata["enable_gpu"] == "true"


def test_kaggle_validation_script_clones_repo() -> None:
    script = Path("kaggle/validation/validation.py").read_text(encoding="utf-8")

    assert "https://github.com/W-Kaski/qwen-qlora-sql-benchmark.git" in script
    assert "KAGGLE_VALIDATION_OK" in script
