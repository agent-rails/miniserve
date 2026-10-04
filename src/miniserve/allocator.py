import random
from collections import deque


class OutOfBlocks(Exception):
    pass


class InvariantViolation(Exception):
    pass


class BlockAllocator:
    def __init__(self, num_blocks: int, shuffle_seed: int | None = None):
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        self.num_blocks = num_blocks
        order = list(range(num_blocks))
        if shuffle_seed is not None:
            random.Random(shuffle_seed).shuffle(order)
        self._free: deque[int] = deque(order)
        self._owned: dict[str, list[int]] = {}

    @property
    def free_count(self) -> int:
        return len(self._free)

    @property
    def used_count(self) -> int:
        return self.num_blocks - len(self._free)

    def can_allocate(self, count: int) -> bool:
        return 0 < count <= len(self._free)

    def allocate(self, seq_id: str, count: int) -> list[int]:
        if count < 1:
            raise ValueError("count must be >= 1")
        if seq_id in self._owned:
            raise ValueError(f"sequence {seq_id!r} already owns blocks")
        if count > len(self._free):
            raise OutOfBlocks(f"need {count}, free {len(self._free)}")
        blocks = [self._free.popleft() for _ in range(count)]
        self._owned[seq_id] = blocks
        return list(blocks)

    def free(self, seq_id: str) -> None:
        if seq_id not in self._owned:
            raise ValueError(f"sequence {seq_id!r} owns no blocks")
        self._free.extend(self._owned.pop(seq_id))

    def table(self, seq_id: str) -> list[int]:
        if seq_id not in self._owned:
            raise ValueError(f"sequence {seq_id!r} owns no blocks")
        return list(self._owned[seq_id])

    def check(self, tables: dict[str, list[int]] | None = None) -> None:
        owned_flat = [b for blocks in self._owned.values() for b in blocks]
        everything = owned_flat + list(self._free)
        if len(everything) != self.num_blocks:
            raise InvariantViolation(f"block count {len(everything)} != pool {self.num_blocks}")
        if set(everything) != set(range(self.num_blocks)):
            raise InvariantViolation("blocks missing or out of range")
        if len(set(owned_flat)) != len(owned_flat):
            raise InvariantViolation("a block is owned by two sequences")
        if set(owned_flat) & set(self._free):
            raise InvariantViolation("a block is both owned and free")
        if tables is not None and tables != self._owned:
            raise InvariantViolation("external block tables differ from allocator records")
