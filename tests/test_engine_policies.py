import pytest
from test_engine import FakeClock, run_all, synthetic, terminal_of, tokens_of

from miniserve.allocator import BlockAllocator
from miniserve.engine import Engine, Request
from miniserve.generate import greedy_generate
from miniserve.kv_paged import PagedKV
from miniserve.runner import PagedRunner

BLOCK_SIZE = 4


@pytest.fixture
def make_engine(cpu_model, config):
    def build(pool_blocks=96, **overrides):
        paged = PagedKV(config, pool_blocks, BLOCK_SIZE, cpu_model.dtype, cpu_model.device)
        runner = PagedRunner(cpu_model, paged, BlockAllocator(pool_blocks, shuffle_seed=5))
        params = {
            "max_model_len": 64,
            "queue_capacity": 16,
            "prefill_chunk": 64,
            "max_running": 2,
            "check_invariants": True,
        }
        params.update(overrides)
        return Engine(runner, config.eos_token_ids, **params)

    return build


def test_static_does_not_admit_mid_batch_and_holds_finished_rows(make_engine):
    engine = make_engine(policy="static")
    engine.submit(synthetic("a", 6, 30))
    engine.submit(synthetic("b", 6, 4))
    engine.step()
    engine.submit(synthetic("c", 6, 4))
    events = []
    while engine.records["b"].reason is None:
        events.extend(engine.step())
    assert engine.records["a"].reason is None
    for _ in range(3):
        events.extend(engine.step())
        assert engine.records["c"].admitted is None
        assert engine.runner.allocator.used_count > 0
    assert any(r.held_rows >= 1 for r in engine.step_log)
    events.extend(run_all(engine))
    assert engine.records["c"].admitted is not None
    assert terminal_of(events, "c").reason == "finished"
    assert engine.runner.allocator.free_count == engine.runner.allocator.num_blocks


def test_static_outputs_match_greedy_under_padding(make_engine, cpu_model, tokenizer):
    engine = make_engine(policy="static", max_model_len=128)
    short = tokenizer.encode("1 2 3 4").ids
    long = tokenizer.encode("Explain why the sky is blue in one sentence:").ids
    engine.submit(Request("short", tuple(short), 5))
    engine.submit(Request("long", tuple(long), 20))
    events = run_all(engine)
    assert tokens_of(events, "short") == greedy_generate(cpu_model, short, 5)
    assert tokens_of(events, "long") == greedy_generate(cpu_model, long, 20)


def test_static_waits_for_timeout_unless_batch_is_full(make_engine):
    clock = FakeClock()
    engine = make_engine(policy="static", max_running=3, static_wait_s=5.0, clock=clock)
    engine.submit(synthetic("a", 4, 4))
    engine.step()
    assert engine.records["a"].admitted is None
    clock.now = 6.0
    engine.step()
    assert engine.records["a"].admitted is not None
    run_all(engine)

    full = make_engine(policy="static", max_running=2, static_wait_s=5.0, clock=FakeClock())
    full.submit(synthetic("x", 4, 4))
    full.submit(synthetic("y", 4, 4))
    full.step()
    assert full.records["x"].admitted is not None and full.records["y"].admitted is not None


def test_max_len_reservation_limits_concurrency(make_engine):
    exact = make_engine(pool_blocks=40, max_running=8)
    slotted = make_engine(pool_blocks=40, max_running=8, reservation="max_len")
    for engine in (exact, slotted):
        for name in ("a", "b", "c"):
            engine.submit(synthetic(name, 6, 6))
        engine.step()
    assert exact.running_count == 3
    assert slotted.running_count == 2
    assert slotted.records["a"].reserved_blocks == 64 // BLOCK_SIZE
    assert exact.records["a"].reserved_blocks == 3


def test_static_runner_failure_with_held_rows_has_single_terminal_per_request(make_engine, monkeypatch):
    engine = make_engine(policy="static")
    engine.submit(synthetic("a", 6, 3))
    engine.submit(synthetic("b", 6, 30))
    events = []
    while engine.records["a"].reason is None:
        events.extend(engine.step())

    def boom(*args, **kwargs):
        raise RuntimeError("device fault")

    monkeypatch.setattr(engine.runner, "decode", boom)
    with pytest.raises(RuntimeError):
        engine.step()
    events.extend(engine.drain())
    assert terminal_of(events, "a").reason == "finished"
    assert terminal_of(events, "b").reason == "error"
    assert engine.runner.allocator.free_count == engine.runner.allocator.num_blocks


def test_negative_static_wait_rejected(make_engine):
    with pytest.raises(ValueError):
        make_engine(static_wait_s=-1.0)
