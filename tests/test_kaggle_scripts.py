from pathlib import Path


def test_kaggle_reproduction_scripts_exist() -> None:
    for path in [
        "scripts/kaggle_setup_check.sh",
        "scripts/kaggle_prepare_dataset.sh",
        "scripts/kaggle_train_r8.sh",
        "scripts/kaggle_train_r16.sh",
        "scripts/kaggle_train_r32.sh",
        "scripts/kaggle_eval_r8.sh",
        "scripts/kaggle_eval_r16.sh",
        "scripts/kaggle_eval_r32.sh",
    ]:
        assert Path(path).is_file()


def test_kaggle_reproduction_doc_points_to_existing_scripts() -> None:
    doc = Path("docs/KAGGLE_REPRODUCTION.md").read_text(encoding="utf-8")

    assert "scripts/kaggle_train_r32.sh" in doc
    assert "scripts/kaggle_eval_r32.sh" in doc
