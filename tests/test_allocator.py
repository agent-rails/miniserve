import pytest

from miniserve.allocator import BlockAllocator, InvariantViolation, OutOfBlocks


def test_allocate_and_free_conserve_blocks():
    alloc = BlockAllocator(8)
    a = alloc.allocate("a", 3)
    b = alloc.allocate("b", 2)
    assert set(a).isdisjoint(b)
    assert alloc.free_count == 3
    alloc.check()
    alloc.free("a")
    assert alloc.free_count == 6
    alloc.check()
    alloc.free("b")
    assert alloc.free_count == 8
    alloc.check()


def test_out_of_blocks_is_atomic():
    alloc = BlockAllocator(4)
    alloc.allocate("a", 3)
    with pytest.raises(OutOfBlocks):
        alloc.allocate("b", 2)
    assert alloc.free_count == 1
    alloc.check()
    with pytest.raises(ValueError):
        alloc.table("b")


def test_duplicate_sequence_rejected():
    alloc = BlockAllocator(4)
    alloc.allocate("a", 1)
    with pytest.raises(ValueError):
        alloc.allocate("a", 1)
    alloc.check()


def test_double_free_rejected():
    alloc = BlockAllocator(4)
    alloc.allocate("a", 1)
    alloc.free("a")
    with pytest.raises(ValueError):
        alloc.free("a")


def test_zero_count_rejected():
    with pytest.raises(ValueError):
        BlockAllocator(4).allocate("a", 0)


def test_invalid_pool_rejected():
    with pytest.raises(ValueError):
        BlockAllocator(0)


def test_check_detects_shared_block():
    alloc = BlockAllocator(4)
    alloc.allocate("a", 2)
    alloc.allocate("b", 2)
    alloc._owned["b"][0] = alloc._owned["a"][0]
    with pytest.raises(InvariantViolation):
        alloc.check()


def test_check_detects_block_both_owned_and_free():
    alloc = BlockAllocator(4)
    alloc.allocate("a", 2)
    alloc._free.append(alloc._owned["a"][0])
    with pytest.raises(InvariantViolation):
        alloc.check()


def test_check_detects_table_divergence():
    alloc = BlockAllocator(4)
    alloc.allocate("a", 2)
    with pytest.raises(InvariantViolation):
        alloc.check({"a": [0, 1]} if alloc.table("a") != [0, 1] else {"a": [3, 2]})


def test_check_accepts_matching_tables():
    alloc = BlockAllocator(4)
    alloc.allocate("a", 2)
    alloc.check({"a": alloc.table("a")})


def test_shuffle_is_deterministic_and_scrambled():
    first = BlockAllocator(16, shuffle_seed=7).allocate("a", 16)
    second = BlockAllocator(16, shuffle_seed=7).allocate("a", 16)
    assert first == second
    assert first != list(range(16))
    assert sorted(first) == list(range(16))
