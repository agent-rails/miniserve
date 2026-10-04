from pathlib import Path
from typing import Protocol

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from miniserve.config import ModelConfig


class KVCache(Protocol):
    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = x.dtype
    x32 = x.to(torch.float32)
    x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return weight * x32.to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class Rope:
    def __init__(self, head_dim: int, theta: float, device: torch.device):
        exponents = torch.arange(0, head_dim, 2, dtype=torch.int64, device=device).float() / head_dim
        self.inv_freq = 1.0 / (theta**exponents)

    def tables(self, positions: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = positions.to(torch.float32)[..., None] * self.inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype)[:, None], emb.sin().to(dtype)[:, None]

    @staticmethod
    def apply(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        return x * cos + rotate_half(x) * sin


def causal_mask(past: int, new: int, device: torch.device) -> torch.Tensor:
    rows = torch.arange(new, device=device)[:, None] + past
    cols = torch.arange(past + new, device=device)[None, :]
    return (cols <= rows)[None, None]


class Qwen3:
    def __init__(self, config: ModelConfig, weights: dict[str, torch.Tensor], device: torch.device):
        self.config = config
        self.weights = weights
        self.device = device
        self.dtype = weights["model.embed_tokens.weight"].dtype
        self.rope = Rope(config.head_dim, config.rope_theta, device)

    @classmethod
    def load(cls, snapshot: Path, config: ModelConfig, device: torch.device, dtype: torch.dtype) -> "Qwen3":
        weights: dict[str, torch.Tensor] = {}
        for name, tensor in load_file(str(snapshot / "model.safetensors"), device="cpu").items():
            weights[name] = tensor.to(device=device, dtype=dtype)
        expected = 3 + config.num_layers * 11
        if len(weights) != expected:
            raise ValueError(f"unexpected tensor count {len(weights)}, expected {expected}")
        return cls(config, weights, device)

    def _attention(
        self,
        layer: int,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv: KVCache,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        c, w = self.config, self.weights
        p = f"model.layers.{layer}.self_attn."
        batch, new, _ = hidden.shape
        q = F.linear(hidden, w[p + "q_proj.weight"]).view(batch, new, c.num_heads, c.head_dim)
        k = F.linear(hidden, w[p + "k_proj.weight"]).view(batch, new, c.num_kv_heads, c.head_dim)
        v = F.linear(hidden, w[p + "v_proj.weight"]).view(batch, new, c.num_kv_heads, c.head_dim)
        q = rms_norm(q, w[p + "q_norm.weight"], c.rms_norm_eps).transpose(1, 2)
        k = rms_norm(k, w[p + "k_norm.weight"], c.rms_norm_eps).transpose(1, 2)
        v = v.transpose(1, 2)
        q = Rope.apply(q, cos, sin)
        k = Rope.apply(k, cos, sin)
        k_all, v_all = kv.update(layer, k, v)
        group = c.num_heads // c.num_kv_heads
        k_all = k_all.repeat_interleave(group, dim=1)
        v_all = v_all.repeat_interleave(group, dim=1)
        out = F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=mask)
        out = out.transpose(1, 2).reshape(batch, new, c.num_heads * c.head_dim)
        return F.linear(out, w[p + "o_proj.weight"])

    def _mlp(self, layer: int, hidden: torch.Tensor) -> torch.Tensor:
        w = self.weights
        p = f"model.layers.{layer}.mlp."
        gate = F.linear(hidden, w[p + "gate_proj.weight"])
        up = F.linear(hidden, w[p + "up_proj.weight"])
        return F.linear(F.silu(gate) * up, w[p + "down_proj.weight"])

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv: KVCache,
        mask: torch.Tensor,
        logit_index: torch.Tensor,
    ) -> torch.Tensor:
        c, w = self.config, self.weights
        hidden = F.embedding(input_ids, w["model.embed_tokens.weight"])
        cos, sin = self.rope.tables(positions, hidden.dtype)
        for layer in range(c.num_layers):
            p = f"model.layers.{layer}."
            normed = rms_norm(hidden, w[p + "input_layernorm.weight"], c.rms_norm_eps)
            hidden = hidden + self._attention(layer, normed, cos, sin, kv, mask)
            normed = rms_norm(hidden, w[p + "post_attention_layernorm.weight"], c.rms_norm_eps)
            hidden = hidden + self._mlp(layer, normed)
        hidden = rms_norm(hidden, w["model.norm.weight"], c.rms_norm_eps)
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        last = hidden[rows, logit_index]
        return F.linear(last, w["lm_head.weight"])
