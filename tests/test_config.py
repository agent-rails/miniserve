import json
import shutil

import pytest

from miniserve.config import load_config


def test_kv_bytes_per_token(config):
    assert config.kv_bytes_per_token == 114_688


def test_block_of_16_tokens_is_1_75_mib(config):
    assert config.kv_bytes_per_token * 16 == int(1.75 * 1024 * 1024)


def test_eos_ids_from_generation_config(config):
    assert config.eos_token_ids == (151645, 151643)


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_type", "llama"),
        ("rope_scaling", {"type": "yarn"}),
        ("use_sliding_window", True),
        ("attention_bias", True),
        ("hidden_act", "gelu"),
        ("tie_word_embeddings", False),
    ],
)
def test_unsupported_features_fail_fast(snapshot, tmp_path, field, value):
    for name in ("config.json", "generation_config.json"):
        shutil.copy(snapshot / name, tmp_path / name)
    raw = json.loads((tmp_path / "config.json").read_text())
    raw[field] = value
    (tmp_path / "config.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_config(tmp_path)
