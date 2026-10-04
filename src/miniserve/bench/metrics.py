import statistics
from dataclasses import dataclass

from miniserve.engine import RequestRecord, StepRecord


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("no values")
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be in [0, 1]")
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


@dataclass(frozen=True)
class RunResult:
    records: dict[str, RequestRecord]
    token_times: dict[str, list[float]]
    step_log: list[StepRecord]
    started: float
    ended: float


def _dist(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {"p50": percentile(values, 0.5), "p99": percentile(values, 0.99), "mean": statistics.fmean(values)}


def intertoken_ms(times: list[float]) -> list[float]:
    return [(b - a) * 1000 for a, b in zip(times, times[1:], strict=False)]


def summarize(result: RunResult) -> dict[str, object]:
    records = result.records
    finished = {rid: r for rid, r in records.items() if r.reason == "finished"}
    ttft = [(r.first_token - r.arrival) * 1000 for r in finished.values() if r.first_token is not None]
    queue = [(r.admitted - r.arrival) * 1000 for r in finished.values() if r.admitted is not None]
    e2e = [(r.finished - r.arrival) * 1000 for r in finished.values() if r.finished is not None]
    itl = [x for rid in finished for x in intertoken_ms(result.token_times.get(rid, []))]
    tokens = sum(r.output_tokens for r in finished.values())
    makespan = result.ended - result.started
    log = result.step_log
    rejected = sum(1 for r in records.values() if r.reason == "rejected")
    return {
        "requests": len(records),
        "finished": len(finished),
        "rejection_rate": rejected / len(records),
        "output_tokens": tokens,
        "makespan_s": makespan,
        "tokens_per_s": tokens / makespan,
        "ttft_ms": _dist(ttft),
        "itl_ms": _dist(itl),
        "queue_wait_ms": _dist(queue),
        "e2e_ms": _dist(e2e),
        "steps": len(log),
        "mean_step_ms": statistics.fmean(s.wall_ms for s in log) if log else None,
        "mean_running": statistics.fmean(s.running for s in log) if log else None,
        "mean_held_rows": statistics.fmean(s.held_rows for s in log) if log else None,
        "mean_reserved_blocks": statistics.fmean(s.reserved_blocks for s in log) if log else None,
        "mean_used_blocks": statistics.fmean(s.used_blocks for s in log) if log else None,
        "peak_reserved_blocks": max((s.reserved_blocks for s in log), default=0),
        "mean_waiting": statistics.fmean(s.waiting for s in log) if log else None,
        "requests_per_s": len(finished) / makespan,
    }
