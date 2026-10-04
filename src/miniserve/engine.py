import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import torch

from miniserve.runner import PagedRunner

TerminalReason = Literal["finished", "cancelled", "deadline_exceeded", "rejected", "error"]


@dataclass(frozen=True)
class Request:
    id: str
    prompt_token_ids: tuple[int, ...]
    max_new_tokens: int
    deadline_s: float | None = None


@dataclass(frozen=True)
class TokenEvent:
    request_id: str
    token_id: int
    step: int


@dataclass(frozen=True)
class TerminalEvent:
    request_id: str
    reason: TerminalReason
    step: int
    detail: str = ""


Event = TokenEvent | TerminalEvent


@dataclass
class RequestRecord:
    arrival: float
    admitted: float | None = None
    first_token: float | None = None
    finished: float | None = None
    reason: TerminalReason | None = None
    output_tokens: int = 0
    reserved_blocks: int = 0


@dataclass(frozen=True)
class StepRecord:
    step: int
    wall_ms: float
    prefill_tokens: int
    decode_tokens: int
    running: int
    waiting: int
    free_blocks: int
    reserved_blocks: int
    used_blocks: int


class _State(Enum):
    WAITING = "waiting"
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass
class _Sequence:
    request: Request
    state: _State = _State.WAITING
    prefill_pos: int = 0
    generated: list[int] = field(default_factory=list)

    @property
    def cached_tokens(self) -> int:
        if self.state is _State.PREFILL:
            return self.prefill_pos
        if self.state is _State.DECODE:
            return len(self.request.prompt_token_ids) + len(self.generated) - 1
        return 0


class Engine:
    def __init__(
        self,
        runner: PagedRunner,
        eos_token_ids: tuple[int, ...],
        max_model_len: int,
        queue_capacity: int,
        prefill_chunk: int,
        max_running: int,
        clock: Callable[[], float] = time.monotonic,
        check_invariants: bool = False,
    ):
        if min(max_model_len, queue_capacity, prefill_chunk, max_running) < 1:
            raise ValueError("engine limits must be >= 1")
        self.runner = runner
        self.eos = frozenset(eos_token_ids)
        self.max_model_len = max_model_len
        self.queue_capacity = queue_capacity
        self.prefill_chunk = prefill_chunk
        self.max_running = max_running
        self.clock = clock
        self.check_invariants = check_invariants
        self.records: dict[str, RequestRecord] = {}
        self.step_log: list[StepRecord] = []
        self._waiting: deque[_Sequence] = deque()
        self._running: list[_Sequence] = []
        self._cancelled: set[str] = set()
        self._events: list[Event] = []
        self._step = 0

    @property
    def block_size(self) -> int:
        return self.runner.paged.block_size

    @property
    def has_work(self) -> bool:
        return bool(self._waiting or self._running)

    @property
    def waiting_count(self) -> int:
        return len(self._waiting)

    @property
    def running_count(self) -> int:
        return len(self._running)

    def blocks_needed(self, request: Request) -> int:
        return -(-(len(request.prompt_token_ids) + request.max_new_tokens) // self.block_size)

    def submit(self, request: Request) -> None:
        if request.id in self.records:
            raise ValueError(f"duplicate request id {request.id!r}")
        self.records[request.id] = RequestRecord(arrival=self.clock())
        detail = self._validation_error(request)
        if detail is not None:
            self._terminate_record(request.id, "rejected", detail)
            return
        self.records[request.id].reserved_blocks = self.blocks_needed(request)
        self._waiting.append(_Sequence(request))

    def cancel(self, request_id: str) -> None:
        record = self.records.get(request_id)
        if record is None:
            raise ValueError(f"unknown request id {request_id!r}")
        if record.reason is None:
            self._cancelled.add(request_id)

    def drain(self) -> list[Event]:
        events, self._events = self._events, []
        return events

    def step(self) -> list[Event]:
        started = time.perf_counter()
        self._expire_and_cancel()
        self._admit()
        decoding = [s for s in self._running if s.state is _State.DECODE]
        try:
            prefill_tokens = self._run_prefill()
            decode_tokens = self._run_decode(decoding)
        except Exception as err:
            self._fail_running(f"{type(err).__name__}: {err}")
            self._step += 1
            raise
        self._log_step(started, prefill_tokens, decode_tokens)
        self._step += 1
        if self.check_invariants:
            self._check()
        return self.drain()

    def _validation_error(self, request: Request) -> str | None:
        if not request.prompt_token_ids:
            return "empty prompt"
        if request.max_new_tokens < 1:
            return "max_new_tokens must be >= 1"
        if len(request.prompt_token_ids) + request.max_new_tokens > self.max_model_len:
            return "prompt plus max_new_tokens exceeds max_model_len"
        if self.blocks_needed(request) > self.runner.allocator.num_blocks:
            return "reservation exceeds pool"
        if len(self._waiting) >= self.queue_capacity:
            return "queue full"
        return None

    def _expire_and_cancel(self) -> None:
        now = self.clock()
        for seq in [*self._waiting, *self._running]:
            request = seq.request
            if request.id in self._cancelled:
                self._finish(seq, "cancelled")
            elif request.deadline_s is not None and now - self.records[request.id].arrival > request.deadline_s:
                self._finish(seq, "deadline_exceeded")

    def _admit(self) -> None:
        allocator = self.runner.allocator
        while self._waiting and len(self._running) < self.max_running:
            head = self._waiting[0]
            need = self.blocks_needed(head.request)
            if not allocator.can_allocate(need):
                return
            allocator.allocate(head.request.id, need)
            self._waiting.popleft()
            head.state = _State.PREFILL
            self._running.append(head)
            self.records[head.request.id].admitted = self.clock()

    def _run_prefill(self) -> int:
        seq = next((s for s in self._running if s.state is _State.PREFILL), None)
        if seq is None:
            return 0
        prompt = seq.request.prompt_token_ids
        chunk = prompt[seq.prefill_pos : seq.prefill_pos + self.prefill_chunk]
        logits = self.runner.prefill(seq.request.id, chunk, seq.prefill_pos)
        seq.prefill_pos += len(chunk)
        if seq.prefill_pos == len(prompt):
            seq.state = _State.DECODE
            self._accept_token(seq, int(torch.argmax(logits).item()))
        return len(chunk)

    def _run_decode(self, batch: list[_Sequence]) -> int:
        if not batch:
            return 0
        ids = [s.request.id for s in batch]
        tokens = [s.generated[-1] for s in batch]
        pasts = [s.cached_tokens for s in batch]
        logits = self.runner.decode(ids, tokens, pasts)
        chosen = [int(t) for t in torch.argmax(logits, dim=-1).to("cpu")]
        for seq, token in zip(batch, chosen, strict=True):
            self._accept_token(seq, token)
        return len(batch)

    def _accept_token(self, seq: _Sequence, token: int) -> None:
        record = self.records[seq.request.id]
        seq.generated.append(token)
        record.output_tokens += 1
        if record.first_token is None:
            record.first_token = self.clock()
        self._events.append(TokenEvent(seq.request.id, token, self._step))
        if token in self.eos or len(seq.generated) == seq.request.max_new_tokens:
            self._finish(seq, "finished")

    def _finish(self, seq: _Sequence, reason: TerminalReason, detail: str = "") -> None:
        request_id = seq.request.id
        if seq in self._running:
            self._running.remove(seq)
            self.runner.allocator.free(request_id)
        elif seq in self._waiting:
            self._waiting.remove(seq)
        self._cancelled.discard(request_id)
        self._terminate_record(request_id, reason, detail)

    def _terminate_record(self, request_id: str, reason: TerminalReason, detail: str) -> None:
        record = self.records[request_id]
        record.reason = reason
        record.finished = self.clock()
        self._events.append(TerminalEvent(request_id, reason, self._step, detail))

    def _fail_running(self, detail: str) -> None:
        for seq in list(self._running):
            self._finish(seq, "error", detail)

    def _log_step(self, started: float, prefill_tokens: int, decode_tokens: int) -> None:
        allocator = self.runner.allocator
        used = sum(-(-s.cached_tokens // self.block_size) for s in self._running)
        self.step_log.append(
            StepRecord(
                step=self._step,
                wall_ms=(time.perf_counter() - started) * 1000,
                prefill_tokens=prefill_tokens,
                decode_tokens=decode_tokens,
                running=len(self._running),
                waiting=len(self._waiting),
                free_blocks=allocator.free_count,
                reserved_blocks=allocator.used_count,
                used_blocks=used,
            )
        )

    def _check(self) -> None:
        allocator = self.runner.allocator
        allocator.check({s.request.id: allocator.table(s.request.id) for s in self._running})
