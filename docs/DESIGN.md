# miniserve design

Status: revision 4. Phases 1 to 4 implemented and tested. The benchmark has one complete repetition, not the three planned; see `BENCHMARKS.md`. Phase 5 (HTTP endpoint) is not built. Measured results go in `MICROBENCH.md` and `BENCHMARKS.md`. Where this document and a measurement disagree, the measurement wins.

## Problem and mechanism

A serving engine turns a prompt into tokens in two phases.

- Prefill: process all prompt tokens at once. Writes keys and values (the KV cache) for every prompt token.
- Decode: produce one token per step. Reads the whole KV cache of the sequence on every step.

Two mechanisms decide server efficiency.

- Continuous batching: the scheduler adds and removes sequences between decode steps. A finished sequence frees its slot immediately. Static batching holds the whole batch until the longest sequence ends.
- Paged KV cache: KV memory is allocated in fixed-size blocks, mapped per sequence by a block table. This removes the need to reserve contiguous memory for the maximum length per sequence.

Goal: implement both from first principles, prove they do not change model output, and measure what each one changes separately.

Sources: [PagedAttention](https://arxiv.org/abs/2309.06180), [Orca (iteration-level scheduling)](https://www.usenix.org/conference/osdi22/presentation/yu). Both URLs resolved on 2026-10-04. The papers' text has not been checked against this document line by line.

Open questions, resolved by Phase 1.5 measurement before any claim is written:

- Decode is usually described as memory-bandwidth-bound. For a 1.2 GB model on M1 Max the bandwidth floor is a few milliseconds per step, and Python plus MPS dispatch across 28 layers may exceed it. If so, batching helps mainly by amortizing dispatch, and the documentation must say that.
- Cost of gather-based attention against a contiguous cache.
- Whether MPS supports the GQA attention path with acceptable speed and numerics.

## Scope

In scope:

- Qwen3-0.6B forward pass written directly in PyTorch (no `transformers` model code at runtime).
- Paged KV cache with a block allocator and per-sequence block tables.
- Iteration-level scheduler with chunked prefill, bounded queue, deadlines, and cancellation.
- Greedy decoding.
- Benchmark harness with open-loop arrivals.
- Streaming HTTP endpoint.

Out of scope, by decision:

- Fused paged-attention kernel. Attention gathers blocks into a contiguous tensor and calls PyTorch SDPA. Speed claims for the paged layout against contiguous are not made.
- Tensor or data parallelism, quantization, speculative decoding, prefix caching, sampling beyond greedy.
- Preemption and swapping. Trigger to revisit: in the pool sweep, if requests wait while measured blocks in use stay below 60% of the pool, worst-case reservation is the limiter and on-demand growth with preemption is justified.
- Multi-tenant isolation. There is one trust domain.

## Inputs, outputs, ownership

- Model identity: Hugging Face repo id plus the exact snapshot revision. The loader refuses a snapshot whose `config.json` has an unsupported feature.
- Request: `{id, prompt_token_ids, max_new_tokens, deadline_s}`. Validation rejects a request (terminal event `rejected`) when the prompt is empty, `max_new_tokens < 1`, `len(prompt) + max_new_tokens > max_model_len`, the block reservation exceeds the whole pool, or the queue is full. A duplicate id raises `ValueError` at submit instead of emitting an event, because an event for a reused id would look like the end of the original request.
- Output: a stream of `(request_id, token_id, step_index)` events and exactly one terminal event: `finished`, `cancelled`, `deadline_exceeded`, `rejected`, or `error`.
- Ownership: the block allocator is the only owner of the KV pool. Only the scheduler mutates block tables. The model runner only reads them.
- Asserted versus observed: the caller asserts prompt tokens. The engine observes lengths, free blocks, and elapsed time.

## Resource model

Qwen3-0.6B, from `config.json` of snapshot `c1899de289a04d12100db370d81485cdf75e47ca`. The review re-derived these numbers independently.

```text
L = 28 layers, Hkv = 8 KV heads, D = 128 head dim, Bkv = 2 bytes (bf16/fp16)
KV bytes per token = 2 x L x Hkv x D x Bkv = 2 x 28 x 8 x 128 x 2 = 114,688 B = 112 KiB
block of 16 tokens = 1.75 MiB
1,024-token sequence = 112 MiB
```

Reservation policies compared in the benchmark, for one sequence with prompt `p` and actual output `o` (unknown at admission) and cap `m = max_new_tokens`:

- `max_len` slot: reserves `max_model_len` tokens. At 2048 tokens that is 224 MiB per admitted sequence.
- `exact_cap`: reserves `p + m` tokens, rounded up to blocks.
- `on_demand`: grows by one block when needed. Not implemented in v1.

v1 admission uses `exact_cap`. This captures only the first benefit of paging, avoiding the `max_model_len` slot. It does not capture the second benefit, not paying for output length that EOS makes unnecessary. The benchmark must report reserved blocks and used blocks separately so the gap is visible, and documentation must state this limit plainly.

KV memory does not bind on this machine at normal sizes. A 4 GiB pool holds about 36,000 tokens, roughly 17 sequences at 2048 tokens. The benchmark therefore sweeps the pool size down to the point where admission binds, and reports the pool size at which each policy starts queuing.

Budgets are separate and explicit: weights (about 1.2 GB in bf16, measured at load), KV pool (configured block count), step token budget, queue capacity.

## Scheduling policy

- One limit, `max_running`, bounds admitted sequences and the decode batch. A separate decode cap could starve an admitted sequence.
- Admission is strict FIFO. If the head request does not fit, later requests wait behind it. This is simple and starvation-free. Skip-ahead is rejected for v1 because it can starve large requests.
- Each scheduler step runs one forward pass for the decode batch and, separately, one forward pass for at most one prefill chunk. Mixed prefill and decode in one pass is deferred, because it needs variable-length masking.
- Decode batch rows with different lengths are padded to the longest row and masked.
- Static policy: a batch forms when `max_running` requests are queued or the oldest has waited `static_wait_s`. No request is admitted until every row of the batch has finished. A finished row keeps its blocks and keeps running in the decode batch on a dummy token until the last row ends. Tokens and the terminal event are still delivered when the row finishes.
- A deadline starts at request arrival, so it includes queue wait. Time to first token is measured separately and is not a deadline.
- Cancellation takes effect at the next step boundary. A token produced in the step where the cancel is observed is not delivered.

## State and failures

Sequence states:

```text
WAITING -> RUNNING_PREFILL -> RUNNING_DECODE -> FINISHED
   |             |                  |
   +-------------+------------------+--> CANCELLED | DEADLINE_EXCEEDED | ERROR
WAITING --> REJECTED (queue full or request invalid)
```

- Every terminal transition frees the sequence's blocks in the same scheduler step.
- Block invariants, checked after every step in tests: free and allocated sets are disjoint; their union is the whole pool; each sequence's block table equals the blocks the allocator records for it; no block appears in two tables.
- A model-runner exception fails the step: all running sequences get an `error` event and release blocks. The engine does not retry a failed step, because a retry could duplicate output tokens.
- No silent fallback. A missing device raises. A shape mismatch raises.

## Alternatives and decision

| Option | For | Against | Decision |
| --- | --- | --- | --- |
| Wrap `transformers` generate | Fast | Teaches nothing; cannot control KV | Rejected, used only as correctness oracle |
| Contiguous KV, slot per sequence | Simple | Fragmentation, the problem to study | Kept as benchmark baseline |
| Paged KV, gather then SDPA | Real allocator and block tables; portable to MPS and CPU | Gather copy cost; no fused kernel | Chosen |
| Fused Metal paged-attention kernel | Realistic speed | Large scope, unrelated to the scheduling question | Deferred |
| Admission at `exact_cap` | No mid-decode failure; beats `max_len` slots | Does not capture EOS savings | Chosen for v1 |
| On-demand growth with preemption | Highest utilization | Needs recompute or swap paths | Deferred, trigger in Scope |

## Verification

Correctness invariants:

1. Oracle equality. For a fixed prompt set, greedy tokens equal Hugging Face `generate` tokens with the same weights. Required exact on CPU float32. On MPS bf16, report match rate and first divergence position; do not claim exactness.
2. Batch independence. A request's logits are the same within a stated tolerance whether it runs alone or batched in any arrival order. Token equality is required only where the top-1 and top-2 logit margin exceeds that tolerance. Report every divergence position. Batched reductions change float summation order, so exact equality across batch shapes is not assumed.
3. Block invariants (listed above) after every step and at drain.
4. Cancellation frees blocks within one step.

Negative tests: over-length request, reservation larger than the pool, duplicate id, queue full, deadline during queue wait, cancel during prefill, cancel during decode, pool exhaustion keeps requests waiting in FIFO order, runner exception.

Engine configurations compared (a 2x2 plus a baseline), so each effect is isolated. All five run through the same paged attention code. The second axis is the reservation policy, not a separate cache implementation, so gather cost is identical across arms and cannot be mistaken for a memory-policy effect.

| Configuration | Scheduling | Reservation per sequence |
| --- | --- | --- |
| `sequential` | one request at a time | `exact_cap` |
| `static-maxlen` | fixed batches | `max_model_len` slot |
| `static-exact` | fixed batches | `exact_cap` blocks |
| `continuous-maxlen` | iteration-level | `max_model_len` slot |
| `continuous-exact` | iteration-level | `exact_cap` blocks |

The cost of the paged layout is measured separately at batch 1: the same requests through contiguous-cache greedy decoding and through the paged `sequential` engine. Gather cost at higher batch sizes is in `MICROBENCH.md`.

Static batching policy: form a batch from queued requests when the batch is full or a wait timeout expires. Finished rows stay in the batch and keep consuming padded compute until the longest row ends. Without this, static would be a strawman.

Workload: open-loop Poisson arrivals. A request's arrival time is its scheduled time, not the time the driver submitted it, so a long step cannot hide queue wait (coordinated omission). Arrival rates are set as multiples of the measured `sequential` capacity. The final run used 1, 2, 4 and 8. The 0.5 rate was dropped after the first attempt because it is arrival-bound and adds little beyond cost. Prompts use the Qwen3 chat template with thinking disabled and vary in length; answers end on EOS, so output lengths vary and reserved and used blocks differ. The pool sweep shrinks the pool to 32, 48, 96 and 192 blocks at the highest arrival rate to find where each reservation policy starts to queue. Prompt and output length distributions, seeds, warmup runs, and repetition counts are recorded with every result. Cold and warm runs are recorded as separate rows in the raw results, but are not analyzed in `BENCHMARKS.md`. Saturation results are sanity-checked against the arrival-rate and service-rate relation.

Metrics: time to first token, inter-token latency (p50, p99), output tokens per second, queue wait, rejection rate, reserved against used blocks. Process memory was planned and not measured.

Claims policy: report what was measured on this machine only. No claim about vLLM or any server not run on the same machine and workload. MPS results are labeled MPS. Decode is not described as bandwidth-bound unless Phase 1.5 shows it.

## Observability and operation

Structured event log per working step: step index, prefill tokens, decode tokens, running, held rows (static policy), waiting, free blocks, reserved blocks, used blocks, step wall time. Idle steps, where a static batch is still forming, are not logged. The benchmark reads this log, not wall-clock estimates.

## Phases

1. Model runner with contiguous cache. Exit: oracle equality on CPU float32.
1.5. Microbenchmark on MPS: decode step time against batch size, and gather against contiguous attention cost. Exit: a measured note that fixes the framing of later claims.
2. Paged cache and allocator. Exit: logits equal to phase 1 within tolerance; block invariants pass.
3. Scheduler: continuous batching, chunked prefill, admission, cancel, deadlines. Exit: batch independence and failure tests pass.
4. Benchmark harness and the five configurations. Exit: `BENCHMARKS.md` with sample counts and distributions, plus a check of MPS bf16 token agreement against the CPU float32 reference.
5. Streaming HTTP endpoint. Optional: a single-file HTML explainer of block tables and batching.
