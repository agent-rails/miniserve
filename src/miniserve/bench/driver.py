import time
from collections import defaultdict
from collections.abc import Callable

from miniserve.bench.metrics import RunResult
from miniserve.bench.workload import WorkItem
from miniserve.engine import Engine, Request, TokenEvent


def run_workload(
    engine: Engine,
    items: list[WorkItem],
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> RunResult:
    if engine.clock is not clock:
        raise ValueError("engine and driver must share one clock")
    pending = sorted(items, key=lambda item: item.arrival_s)
    token_times: dict[str, list[float]] = defaultdict(list)
    started = clock()
    index = 0
    while index < len(pending) or engine.has_work:
        elapsed = clock() - started
        while index < len(pending) and pending[index].arrival_s <= elapsed:
            item = pending[index]
            engine.submit(Request(item.request_id, item.prompt_token_ids, item.max_new_tokens))
            engine.records[item.request_id].arrival = started + item.arrival_s
            index += 1
        if not engine.has_work:
            sleep(max(0.0, pending[index].arrival_s - (clock() - started)))
            continue
        events = engine.step()
        now = clock()
        for event in events:
            if isinstance(event, TokenEvent):
                token_times[event.request_id].append(now)
    return RunResult(dict(engine.records), dict(token_times), list(engine.step_log), started, clock())
