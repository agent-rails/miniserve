import itertools

import pytest

from miniserve.allocator import BlockAllocator
from miniserve.engine import Engine, Request, TerminalEvent, TokenEvent
from miniserve.generate import greedy_generate
from miniserve.kv_paged import PagedKV
from miniserve.runner import PagedRunner

BLOCK_SIZE = 4
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    ",
    "Explain why the sky is blue in one sentence:",
    "1 2 3 4 5 6 7 8 9",
]
NEW_TOKENS = 16


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def make_engine(cpu_model, config):
    def build(pool_blocks=96, **overrides):
        paged = PagedKV(config, pool_blocks, BLOCK_SIZE, cpu_model.dtype, cpu_model.device)
        runner = PagedRunner(cpu_model, paged, BlockAllocator(pool_blocks, shuffle_seed=5))
        params = {
            "max_model_len": 256,
            "queue_capacity": 16,
            "prefill_chunk": 64,
            "max_running": 8,
            "check_invariants": True,
        }
        params.update(overrides)
        return Engine(runner, config.eos_token_ids, **params)

    return build


def synthetic(request_id, prompt_len, max_new, deadline_s=None):
    prompt = tuple(1000 + i for i in range(prompt_len))
    return Request(request_id, prompt, max_new, deadline_s)


def run_all(engine, limit=500):
    events = []
    for _ in range(limit):
        if not engine.has_work:
            return events
        events.extend(engine.step())
    raise AssertionError("engine did not drain")


def tokens_of(events, request_id):
    return [e.token_id for e in events if isinstance(e, TokenEvent) and e.request_id == request_id]


def terminal_of(events, request_id):
    found = [e for e in events if isinstance(e, TerminalEvent) and e.request_id == request_id]
    assert len(found) == 1
    return found[0]


def test_single_request_matches_greedy(make_engine, cpu_model, tokenizer):
    engine = make_engine()
    ids = tokenizer.encode(PROMPTS[0]).ids
    engine.submit(Request("r", tuple(ids), NEW_TOKENS))
    events = run_all(engine)
    assert tokens_of(events, "r") == greedy_generate(cpu_model, ids, NEW_TOKENS)
    assert terminal_of(events, "r").reason == "finished"
    assert engine.runner.allocator.free_count == engine.runner.allocator.num_blocks
    record = engine.records["r"]
    assert record.first_token is not None and record.finished is not None
    assert record.output_tokens == len(tokens_of(events, "r"))


def test_staggered_arrivals_are_batch_independent(make_engine, cpu_model, tokenizer):
    engine = make_engine()
    prompts = [tokenizer.encode(p).ids for p in PROMPTS]
    expected = [greedy_generate(cpu_model, p, NEW_TOKENS) for p in prompts]
    events = []
    engine.submit(Request("r0", tuple(prompts[0]), NEW_TOKENS))
    engine.submit(Request("r1", tuple(prompts[1]), NEW_TOKENS))
    for _ in range(3):
        events.extend(engine.step())
    engine.submit(Request("r2", tuple(prompts[2]), NEW_TOKENS))
    events.extend(engine.step())
    engine.submit(Request("r3", tuple(prompts[3]), NEW_TOKENS))
    events.extend(run_all(engine))
    for i, exp in enumerate(expected):
        assert tokens_of(events, f"r{i}") == exp
        assert terminal_of(events, f"r{i}").reason == "finished"


def test_chunked_prefill_spans_steps_and_matches_greedy(make_engine, cpu_model, tokenizer):
    engine = make_engine(prefill_chunk=8)
    ids = tokenizer.encode(
        "The quick brown fox jumps over the lazy dog and then keeps running far away today. " * 3
    ).ids
    assert len(ids) > 40
    engine.submit(Request("r", tuple(ids), 8))
    events = run_all(engine)
    first = next(e for e in events if isinstance(e, TokenEvent))
    assert first.step >= len(ids) // 8
    assert tokens_of(events, "r") == greedy_generate(cpu_model, ids, 8)


def test_fifo_head_of_line_blocks_later_small_request(make_engine):
    tick = itertools.count()
    engine = make_engine(pool_blocks=10, clock=lambda: float(next(tick)))
    engine.submit(synthetic("a", 8, 24))
    engine.submit(synthetic("b", 8, 16))
    engine.submit(synthetic("c", 4, 4))
    engine.step()
    assert engine.records["a"].admitted is not None
    assert engine.records["b"].admitted is None
    assert engine.records["c"].admitted is None
    assert engine.runner.allocator.free_count == 2
    events = run_all(engine)
    assert {terminal_of(events, r).reason for r in "abc"} == {"finished"}
    assert engine.records["b"].admitted < engine.records["c"].admitted


@pytest.mark.parametrize(
    "request_obj,detail",
    [
        (Request("x", (), 4), "empty prompt"),
        (Request("x", (1, 2, 3), 0), "max_new_tokens"),
        (synthetic("x", 200, 100), "max_model_len"),
        (synthetic("x", 8, 100), "reservation exceeds pool"),
    ],
)
def test_invalid_requests_are_rejected(make_engine, request_obj, detail):
    engine = make_engine(pool_blocks=8)
    engine.submit(request_obj)
    events = engine.drain()
    assert len(events) == 1
    assert isinstance(events[0], TerminalEvent)
    assert events[0].reason == "rejected"
    assert detail in events[0].detail
    assert not engine.has_work


def test_queue_full_rejects(make_engine):
    engine = make_engine(queue_capacity=1)
    engine.submit(synthetic("a", 4, 4))
    engine.submit(synthetic("b", 4, 4))
    events = engine.drain()
    assert [(e.request_id, e.reason) for e in events if isinstance(e, TerminalEvent)] == [("b", "rejected")]
    assert engine.waiting_count == 1


def test_duplicate_id_raises_and_original_unaffected(make_engine):
    engine = make_engine()
    engine.submit(synthetic("a", 4, 4))
    with pytest.raises(ValueError):
        engine.submit(synthetic("a", 4, 4))
    events = run_all(engine)
    assert terminal_of(events, "a").reason == "finished"


def test_cancel_unknown_raises(make_engine):
    with pytest.raises(ValueError):
        make_engine().cancel("nope")


def test_cancel_waiting_request(make_engine):
    engine = make_engine(pool_blocks=10)
    engine.submit(synthetic("a", 8, 24))
    engine.submit(synthetic("b", 8, 16))
    engine.step()
    engine.cancel("b")
    events = engine.step()
    assert terminal_of(events, "b").reason == "cancelled"
    assert engine.waiting_count == 0
    assert "a" in {s.request.id for s in engine._running}


def test_cancel_during_prefill_frees_blocks(make_engine):
    engine = make_engine(prefill_chunk=4)
    engine.submit(synthetic("a", 20, 8))
    engine.step()
    assert engine.runner.allocator.used_count > 0
    engine.cancel("a")
    events = engine.step()
    assert terminal_of(events, "a").reason == "cancelled"
    assert not any(isinstance(e, TokenEvent) for e in events)
    assert engine.runner.allocator.free_count == engine.runner.allocator.num_blocks


def test_cancel_during_decode_delivers_no_token_after_cancel(make_engine):
    engine = make_engine()
    engine.submit(synthetic("a", 6, 40))
    first = engine.step()
    assert tokens_of(first, "a")
    engine.cancel("a")
    events = engine.step()
    assert tokens_of(events, "a") == []
    assert terminal_of(events, "a").reason == "cancelled"
    assert engine.runner.allocator.free_count == engine.runner.allocator.num_blocks


def test_deadline_counts_queue_wait(make_engine):
    clock = FakeClock()
    engine = make_engine(pool_blocks=10, clock=clock)
    engine.submit(synthetic("a", 8, 24))
    engine.submit(synthetic("b", 8, 16, deadline_s=5.0))
    engine.step()
    clock.now = 10.0
    events = engine.step()
    assert terminal_of(events, "b").reason == "deadline_exceeded"
    assert engine.records["b"].admitted is None
    assert engine.records["a"].reason is None


def test_runner_failure_errors_running_and_frees_blocks(make_engine, monkeypatch):
    engine = make_engine(max_running=2)
    for name in ("a", "b", "c"):
        engine.submit(synthetic(name, 6, 12))
    engine.step()

    def boom(*args, **kwargs):
        raise RuntimeError("device fault")

    monkeypatch.setattr(engine.runner, "decode", boom)
    with pytest.raises(RuntimeError, match="device fault"):
        engine.step()
    events = engine.drain()
    assert {terminal_of(events, r).reason for r in "ab"} == {"error"}
    assert "device fault" in terminal_of(events, "a").detail
    assert engine.runner.allocator.free_count == engine.runner.allocator.num_blocks
    assert engine.waiting_count == 1
    monkeypatch.undo()
    rest = run_all(engine)
    assert terminal_of(rest, "c").reason == "finished"


def test_step_log_reports_reserved_and_used_blocks(make_engine):
    engine = make_engine()
    engine.submit(synthetic("a", 6, 10))
    run_all(engine)
    log = engine.step_log
    assert log[0].reserved_blocks == -(-16 // BLOCK_SIZE)
    assert all(r.reserved_blocks >= r.used_blocks for r in log)
    assert any(r.used_blocks < r.reserved_blocks for r in log)
    assert log[-1].reserved_blocks == 0 and log[-1].free_blocks == engine.runner.allocator.num_blocks


@pytest.mark.parametrize("field", ["max_model_len", "queue_capacity", "prefill_chunk", "max_running"])
def test_engine_limits_validated(make_engine, field):
    with pytest.raises(ValueError):
        make_engine(**{field: 0})
