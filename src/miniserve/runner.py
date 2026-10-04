from collections.abc import Sequence

import torch

from miniserve.allocator import BlockAllocator
from miniserve.kv_paged import PagedKV
from miniserve.model import Qwen3


class PagedRunner:
    def __init__(self, model: Qwen3, paged: PagedKV, allocator: BlockAllocator):
        if allocator.num_blocks != paged.num_blocks:
            raise ValueError("allocator and cache pool sizes differ")
        self.model = model
        self.paged = paged
        self.allocator = allocator

    def prefill(self, seq_id: str, token_ids: Sequence[int], past: int) -> torch.Tensor:
        if not token_ids:
            raise ValueError("empty chunk")
        device = self.model.device
        count = len(token_ids)
        mask = self.paged.bind([self.allocator.table(seq_id)], [past], count)
        ids = torch.tensor([list(token_ids)], dtype=torch.long, device=device)
        positions = torch.arange(past, past + count, device=device)[None]
        index = torch.tensor([count - 1], dtype=torch.long, device=device)
        return self.model.forward(ids, positions, self.paged, mask, index)[0]

    def decode(self, seq_ids: Sequence[str], tokens: Sequence[int], pasts: Sequence[int]) -> torch.Tensor:
        if not (len(seq_ids) == len(tokens) == len(pasts)) or not seq_ids:
            raise ValueError("seq_ids, tokens and pasts must be equal length and non-empty")
        device = self.model.device
        tables = [self.allocator.table(s) for s in seq_ids]
        mask = self.paged.bind(tables, pasts, 1)
        ids = torch.tensor(list(tokens), dtype=torch.long, device=device)[:, None]
        positions = torch.tensor(list(pasts), dtype=torch.long, device=device)[:, None]
        index = torch.zeros(len(seq_ids), dtype=torch.long, device=device)
        return self.model.forward(ids, positions, self.paged, mask, index)
