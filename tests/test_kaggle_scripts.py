from pathlib import Path


def test_kaggle_reproduction_scripts_exist() -> None:
    for path in [
        "scripts/kaggle_validate_environment.sh",
        "scripts/kaggle_prepare_dataset.sh",
        "scripts/train_r8.sh",
        "scripts/train_r16.sh",
        "scripts/train_r32.sh",
        "scripts/eval_r8.sh",
        "scripts/eval_r16.sh",
        "scripts/eval_r32.sh",
    ]:
        assert Path(path).is_file()


def test_kaggle_validation_doc_points_to_existing_scripts() -> None:
    doc = Path("docs/KAGGLE_VALIDATION.md").read_text(encoding="utf-8")

    assert "scripts/train_r32.sh" in doc
    assert "scripts/eval_r32.sh" in doc
