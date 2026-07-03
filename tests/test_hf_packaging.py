from pathlib import Path


def test_hf_model_card_exists_and_names_base_model() -> None:
    model_card = Path("model_cards/qwen25-15b-text2sql-lora-r32/README.md")

    text = model_card.read_text(encoding="utf-8")

    assert "Qwen/Qwen2.5-1.5B-Instruct" in text
    assert "Execution Accuracy" not in text
    assert "execution accuracy" in text


def test_hf_upload_script_does_not_embed_token() -> None:
    script = Path("scripts/hf_upload_adapter.sh").read_text(encoding="utf-8")

    assert "HF_REPO_ID" in script
    assert "hf upload" in script
    assert "hf auth whoami" in script
    assert "TOKEN=" not in script
