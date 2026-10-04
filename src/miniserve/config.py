import json
from dataclasses import dataclass
from pathlib import Path

MODEL_REPO = "models--Qwen--Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    eos_token_ids: tuple[int, ...]

    @property
    def kv_bytes_per_token(self) -> int:
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * 2


def resolve_snapshot(hub_dir: Path | None = None) -> Path:
    hub = hub_dir if hub_dir is not None else Path.home() / ".cache" / "huggingface" / "hub"
    snapshot = hub / MODEL_REPO / "snapshots" / MODEL_REVISION
    if not snapshot.is_dir():
        raise FileNotFoundError(f"pinned snapshot missing: {snapshot}")
    return snapshot


def load_config(snapshot: Path) -> ModelConfig:
    raw = json.loads((snapshot / "config.json").read_text())
    if raw["model_type"] != "qwen3":
        raise ValueError(f"unsupported model_type {raw['model_type']!r}")
    if raw.get("rope_scaling") is not None:
        raise ValueError("rope_scaling unsupported")
    if raw.get("use_sliding_window"):
        raise ValueError("sliding window unsupported")
    if raw.get("attention_bias"):
        raise ValueError("attention_bias unsupported")
    if raw["hidden_act"] != "silu":
        raise ValueError(f"unsupported hidden_act {raw['hidden_act']!r}")
    if not raw["tie_word_embeddings"]:
        raise ValueError("untied embeddings unsupported")
    generation = json.loads((snapshot / "generation_config.json").read_text())
    eos = generation["eos_token_id"]
    eos_ids = tuple(eos) if isinstance(eos, list) else (eos,)
    return ModelConfig(
        vocab_size=raw["vocab_size"],
        hidden_size=raw["hidden_size"],
        intermediate_size=raw["intermediate_size"],
        num_layers=raw["num_hidden_layers"],
        num_heads=raw["num_attention_heads"],
        num_kv_heads=raw["num_key_value_heads"],
        head_dim=raw["head_dim"],
        rms_norm_eps=raw["rms_norm_eps"],
        rope_theta=float(raw["rope_theta"]),
        max_position_embeddings=raw["max_position_embeddings"],
        eos_token_ids=eos_ids,
    )
