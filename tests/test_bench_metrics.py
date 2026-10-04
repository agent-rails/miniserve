import pytest

from miniserve.bench.metrics import RunResult, intertoken_ms, percentile, summarize
from miniserve.engine import RequestRecord, StepRecord


def test_percentile_interpolates():
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert percentile([5.0], 0.99) == 5.0
    assert percentile([1.0, 2.0], 0.0) == 1.0
    assert percentile([1.0, 2.0], 1.0) == 2.0


def test_percentile_validates():
    with pytest.raises(ValueError):
        percentile([], 0.5)
    with pytest.raises(ValueError):
        percentile([1.0], 1.5)


def test_intertoken_ms():
    assert intertoken_ms([0.0, 0.010, 0.030]) == pytest.approx([10.0, 20.0])
    assert intertoken_ms([1.0]) == []


def step(n, running, held, waiting, reserved, used):
    return StepRecord(n, 10.0, 0, running, running, held, waiting, 100 - reserved, reserved, used)


def test_summarize_computes_latency_and_utilization():
    records = {
        "a": RequestRecord(
            arrival=0.0, admitted=0.1, first_token=0.2, finished=1.0, reason="finished", output_tokens=3
        ),
        "b": RequestRecord(
            arrival=0.5, admitted=0.5, first_token=0.9, finished=2.0, reason="finished", output_tokens=5
        ),
        "c": RequestRecord(arrival=0.6, finished=0.6, reason="rejected"),
    }
    times = {"a": [0.2, 0.4, 1.0], "b": [0.9, 1.0, 1.2, 1.5, 2.0]}
    log = [step(0, 2, 0, 0, 10, 4), step(1, 1, 1, 0, 6, 6)]
    summary = summarize(RunResult(records, times, log, 0.0, 2.0))
    assert summary["requests"] == 3 and summary["finished"] == 2
    assert summary["rejection_rate"] == pytest.approx(1 / 3)
    assert summary["output_tokens"] == 8
    assert summary["tokens_per_s"] == pytest.approx(4.0)
    assert summary["ttft_ms"]["p50"] == pytest.approx(300.0)
    assert summary["queue_wait_ms"]["p50"] == pytest.approx(50.0)
    assert summary["mean_reserved_blocks"] == pytest.approx(8.0)
    assert summary["mean_used_blocks"] == pytest.approx(5.0)
    assert summary["peak_reserved_blocks"] == 10
    assert summary["mean_held_rows"] == pytest.approx(0.5)


def test_summarize_with_no_finished_requests():
    records = {"a": RequestRecord(arrival=0.0, finished=0.0, reason="rejected")}
    summary = summarize(RunResult(records, {}, [], 0.0, 1.0))
    assert summary["ttft_ms"] is None and summary["finished"] == 0 and summary["steps"] == 0
