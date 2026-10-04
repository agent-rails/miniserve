import torch

from miniserve.config import ModelConfig


class ContiguousKV:
    def __init__(self, config: ModelConfig, batch: int, max_len: int, dtype: torch.dtype, device: torch.device):
        shape = (config.num_layers, batch, config.num_kv_heads, max_len, config.head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.max_len = max_len
        self.length = 0

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        new = k.shape[2]
        end = self.length + new
        if end > self.max_len:
            raise ValueError(f"cache overflow: {end} > {self.max_len}")
        self.k[layer, :, :, self.length : end] = k
        self.v[layer, :, :, self.length : end] = v
        return self.k[layer, :, :, :end], self.v[layer, :, :, :end]

    def advance(self, new: int) -> None:
        self.length += new
