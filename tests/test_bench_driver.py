import time

import pytest

from miniserve.allocator import BlockAllocator
from miniserve.bench.driver import run_workload
from miniserve.bench.metrics import summarize
from miniserve.bench.workload import WorkItem
from miniserve.engine import Engine
from miniserve.kv_paged import PagedKV
from miniserve.runner import PagedRunner


@pytest.fixture
def engine(cpu_model, config):
    paged = PagedKV(config, 64, 4, cpu_model.dtype, cpu_model.device)
    runner = PagedRunner(cpu_model, paged, BlockAllocator(64))
    return Engine(runner, config.eos_token_ids, 64, 16, 32, 4, check_invariants=True)


def items(arrivals):
    return [WorkItem(f"r{i}", tuple(range(1000 + i, 1006 + i)), 4, t) for i, t in enumerate(arrivals)]


def test_open_loop_run_records_scheduled_arrivals_and_token_times(engine):
    work = items([0.0, 0.05, 0.4])
    result = run_workload(engine, work)
    for item in work:
        record = result.records[item.request_id]
        assert record.arrival == pytest.approx(result.started + item.arrival_s)
        assert record.reason == "finished"
        assert len(result.token_times[item.request_id]) == record.output_tokens
    assert result.ended - result.started >= 0.4
    summary = summarize(result)
    assert summary["finished"] == 3 and summary["ttft_ms"]["p50"] > 0


def test_driver_waits_for_future_arrivals(engine):
    start = time.monotonic()
    result = run_workload(engine, items([0.3]))
    assert time.monotonic() - start >= 0.3
    assert result.records["r0"].admitted is not None
    assert result.records["r0"].admitted >= result.started + 0.3


def test_clock_mismatch_rejected(engine):
    with pytest.raises(ValueError):
        run_workload(engine, items([0.0]), clock=lambda: 0.0)
