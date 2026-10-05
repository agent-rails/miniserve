from collections.abc import Sequence

import torch

from miniserve.config import ModelConfig


class PagedKV:
    def __init__(
        self,
        config: ModelConfig,
        num_blocks: int,
        block_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        shape = (config.num_layers, num_blocks, block_size, config.num_kv_heads, config.head_dim)
        self.k_pool = torch.zeros(shape, dtype=dtype, device=device)
        self.v_pool = torch.zeros(shape, dtype=dtype, device=device)
        self.config = config
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.device = device
        self._rows = 0
        self._new = 0
        self._blocks_per_row = 0
        self._gather_index: torch.Tensor | None = None
        self._write_slots: torch.Tensor | None = None

    def bind(self, tables: Sequence[Sequence[int]], pasts: Sequence[int], new: int) -> torch.Tensor:
        if len(tables) != len(pasts) or not tables:
            raise ValueError("tables and pasts must be equal length and non-empty")
        if new < 1:
            raise ValueError("new must be >= 1")
        bs = self.block_size
        needs: list[int] = []
        for table, past in zip(tables, pasts, strict=True):
            if past < 0:
                raise ValueError("past must be >= 0")
            if any(block < 0 or block >= self.num_blocks for block in table):
                raise ValueError("block id outside cache pool")
            need = -(-(past + new) // bs)
            if len(table) < need:
                raise ValueError(f"table has {len(table)} blocks, needs {need}")
            needs.append(need)
        blocks_per_row = max(needs)
        gather: list[int] = []
        slots: list[int] = []
        for table, past, need in zip(tables, pasts, needs, strict=True):
            padded = list(table[:need]) + [table[0]] * (blocks_per_row - need)
            gather.extend(padded)
            for pos in range(past, past + new):
                slots.append(table[pos // bs] * bs + pos % bs)
        self._rows = len(tables)
        self._new = new
        self._blocks_per_row = blocks_per_row
        self._gather_index = torch.tensor(gather, dtype=torch.long, device=self.device)
        self._write_slots = torch.tensor(slots, dtype=torch.long, device=self.device)
        span = blocks_per_row * bs
        past_t = torch.tensor(list(pasts), dtype=torch.long, device=self.device)[:, None, None]
        rows = torch.arange(new, device=self.device)[None, :, None]
        cols = torch.arange(span, device=self.device)[None, None, :]
        return (cols <= past_t + rows)[:, None]

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._gather_index is None or self._write_slots is None:
            raise RuntimeError("bind must be called before update")
        c = self.config
        if k.shape[0] != self._rows or k.shape[2] != self._new:
            raise ValueError(f"k shape {tuple(k.shape)} does not match bound batch")
        flat = self.num_blocks * self.block_size
        for pool, value in ((self.k_pool[layer], k), (self.v_pool[layer], v)):
            rows = value.transpose(1, 2).reshape(self._rows * self._new, c.num_kv_heads, c.head_dim)
            pool.view(flat, c.num_kv_heads, c.head_dim).index_copy_(0, self._write_slots, rows)
        span = self._blocks_per_row * self.block_size
        k_all = torch.index_select(self.k_pool[layer], 0, self._gather_index)
        v_all = torch.index_select(self.v_pool[layer], 0, self._gather_index)
        k_all = k_all.view(self._rows, span, c.num_kv_heads, c.head_dim).transpose(1, 2)
        v_all = v_all.view(self._rows, span, c.num_kv_heads, c.head_dim).transpose(1, 2)
        return k_all, v_all
