import pytest
import torch

from miniserve.allocator import BlockAllocator
from miniserve.generate import greedy_generate
from miniserve.kv_paged import PagedKV
from miniserve.runner import PagedRunner

BLOCK_SIZE = 4
POOL_BLOCKS = 96
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    ",
    "Explain why the sky is blue in one sentence:",
    "1 2 3 4 5 6 7 8 9",
]
NEW_TOKENS = 20


def blocks_needed(prompt_len: int, new_tokens: int) -> int:
    return -(-(prompt_len + new_tokens) // BLOCK_SIZE)


@pytest.fixture
def stack(cpu_model, config):
    paged = PagedKV(config, POOL_BLOCKS, BLOCK_SIZE, cpu_model.dtype, cpu_model.device)
    alloc = BlockAllocator(POOL_BLOCKS, shuffle_seed=3)
    return PagedRunner(cpu_model, paged, alloc), alloc


def paged_single(runner, alloc, seq, prompt_ids, new_tokens, eos):
    alloc.allocate(seq, blocks_needed(len(prompt_ids), new_tokens))
    logits = runner.prefill(seq, prompt_ids, 0)
    out = []
    past = len(prompt_ids)
    for _ in range(new_tokens):
        token = int(logits.argmax().item())
        out.append(token)
        if token in eos or len(out) == new_tokens:
            break
        logits = runner.decode([seq], [token], [past])[0]
        past += 1
    return out


@pytest.mark.parametrize("prompt", PROMPTS)
def test_paged_matches_contiguous(prompt, stack, cpu_model, tokenizer, config):
    runner, alloc = stack
    ids = tokenizer.encode(prompt).ids
    expected = greedy_generate(cpu_model, ids, NEW_TOKENS)
    got = paged_single(runner, alloc, "s", ids, NEW_TOKENS, set(config.eos_token_ids))
    assert got == expected
    alloc.check()
    alloc.free("s")
    alloc.check()


def test_chunked_prefill_matches_single_pass(stack, cpu_model, tokenizer):
    runner, alloc = stack
    ids = tokenizer.encode("The quick brown fox jumps over the lazy dog and keeps running far away").ids
    alloc.allocate("whole", blocks_needed(len(ids), 1))
    whole = runner.prefill("whole", ids, 0)
    alloc.allocate("chunks", blocks_needed(len(ids), 1))
    past = 0
    for start in range(0, len(ids), 5):
        chunk = ids[start : start + 5]
        logits = runner.prefill("chunks", chunk, past)
        past += len(chunk)
    assert torch.allclose(whole, logits, atol=1e-4)
    assert int(whole.argmax()) == int(logits.argmax())


def test_freed_blocks_with_stale_data_do_not_leak(stack, cpu_model, tokenizer, config):
    runner, alloc = stack
    eos = set(config.eos_token_ids)
    first = tokenizer.encode("Stale content that should be overwritten completely by the next owner").ids
    paged_single(runner, alloc, "a", first, NEW_TOKENS, eos)
    alloc.free("a")
    ids = tokenizer.encode("The capital of France is").ids
    expected = greedy_generate(cpu_model, ids, NEW_TOKENS)
    got = paged_single(runner, alloc, "b", ids, NEW_TOKENS, eos)
    assert got == expected


def test_batched_decode_matches_individual(stack, cpu_model, tokenizer, config):
    runner, alloc = stack
    eos = set(config.eos_token_ids)
    prompts = [tokenizer.encode(p).ids for p in PROMPTS]
    expected = [greedy_generate(cpu_model, p, NEW_TOKENS) for p in prompts]
    seqs = [f"s{i}" for i in range(len(prompts))]
    state = {}
    for seq, p in zip(seqs, prompts, strict=True):
        alloc.allocate(seq, blocks_needed(len(p), NEW_TOKENS))
        logits = runner.prefill(seq, p, 0)
        state[seq] = {"out": [int(logits.argmax())], "past": len(p)}
    done = {s for s in seqs if state[s]["out"][-1] in eos or len(state[s]["out"]) == NEW_TOKENS}
    while len(done) < len(seqs):
        active = [s for s in seqs if s not in done]
        logits = runner.decode(active, [state[s]["out"][-1] for s in active], [state[s]["past"] for s in active])
        for seq, row in zip(active, logits, strict=True):
            state[seq]["past"] += 1
            state[seq]["out"].append(int(row.argmax()))
            if state[seq]["out"][-1] in eos or len(state[seq]["out"]) == NEW_TOKENS:
                done.add(seq)
        alloc.check()
    for seq, exp in zip(seqs, expected, strict=True):
        assert state[seq]["out"] == exp


def test_bind_rejects_short_table(config, cpu_model):
    paged = PagedKV(config, 8, BLOCK_SIZE, cpu_model.dtype, cpu_model.device)
    with pytest.raises(ValueError):
        paged.bind([[0]], [0], 9)


def test_bind_rejects_negative_past(config, cpu_model):
    paged = PagedKV(config, 8, BLOCK_SIZE, cpu_model.dtype, cpu_model.device)
    with pytest.raises(ValueError):
        paged.bind([[0]], [-1], 1)


@pytest.mark.parametrize("block", [-1, 8])
def test_bind_rejects_out_of_range_block(config, cpu_model, block):
    paged = PagedKV(config, 8, BLOCK_SIZE, cpu_model.dtype, cpu_model.device)
    with pytest.raises(ValueError):
        paged.bind([[block]], [0], 1)


def test_update_before_bind_fails(config, cpu_model):
    paged = PagedKV(config, 8, BLOCK_SIZE, cpu_model.dtype, cpu_model.device)
    k = torch.zeros(1, config.num_kv_heads, 1, config.head_dim)
    with pytest.raises(RuntimeError):
        paged.update(0, k, k)


def test_runner_rejects_mismatched_pool(config, cpu_model):
    paged = PagedKV(config, 8, BLOCK_SIZE, cpu_model.dtype, cpu_model.device)
    with pytest.raises(ValueError):
        PagedRunner(cpu_model, paged, BlockAllocator(9))
