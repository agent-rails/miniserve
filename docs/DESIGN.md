# miniserve design

Status: proposed. Nothing here is measured yet. Measured results go in `BENCHMARKS.md`.

## Problem and mechanism

An LLM server turns a prompt into tokens in two phases.

- Prefill: process all prompt tokens at once. Compute-bound. Writes keys and values (the KV cache) for every prompt token.
- Decode: produce one token per step. Memory-bandwidth-bound. Reads the whole KV cache of the sequence on every step.

Two mechanisms decide server efficiency.

- Continuous batching: the scheduler adds and removes sequences between decode steps. A finished sequence frees its slot immediately. Static batching holds the whole batch until the longest sequence ends.
- Paged KV cache: KV memory is allocated in fixed-size blocks, mapped per sequence by a block table. This removes the need to reserve `max_len` of contiguous memory per sequence.

Goal: implement both from first principles, prove they do not change model output, and measure what they change.

Sources: [PagedAttention](https://arxiv.org/abs/2309.06180), [Orca (iteration-level scheduling)](https://www.usenix.org/conference/osdi22/presentation/yu).

Not yet understood and to be resolved by experiment: how much the gather-based attention on PyTorch/MPS costs against a contiguous cache, and whether MPS bf16 greedy decoding is stable enough for token-exact comparison.

## Scope

In scope:

- Qwen3-0.6B forward pass written directly in PyTorch (no `transformers` model code at runtime).
- Paged KV cache with a block allocator and per-sequence block tables.
- Iteration-level scheduler with chunked prefill, bounded queue, deadlines, and cancellation.
- Greedy decoding.
- Benchmark harness with open-loop arrivals and three engine modes.
- Streaming HTTP endpoint (last phase).

Out of scope, by decision:

- Fused paged-attention kernel. Attention gathers blocks into a contiguous tensor and calls PyTorch SDPA. Speed claims for the paged layout versus contiguous are therefore not made.
- Tensor or data parallelism, quantization, speculative decoding, prefix caching, sampling beyond greedy.
- Preemption and swapping. A sequence that cannot get blocks waits in the queue.
- Multi-tenant isolation. There is one trust domain.

## Inputs, outputs, ownership

- Model identity: Hugging Face repo id plus the exact snapshot revision. The loader refuses a snapshot whose `config.json` differs from the supported architecture.
- Request: `{id, prompt_token_ids, max_new_tokens, deadline_s}`. The engine validates `len(prompt) + max_new_tokens <= max_model_len` and rejects otherwise.
- Output: a stream of `(request_id, token_id, step_index)` events and one terminal event: `finished`, `cancelled`, `deadline_exceeded`, or `rejected`.
- Ownership: the block allocator is the only owner of the KV pool. Only the scheduler mutates block tables. The model runner only reads them.
- Asserted versus observed: the caller asserts prompt tokens. The engine observes lengths, free blocks, and elapsed time.

## Resource model

Qwen3-0.6B, from `config.json` of snapshot `c1899de289a04d12100db370d81485cdf75e47ca`:

```text
L = 28 layers, Hkv = 8 KV heads, D = 128 head dim, Bkv = 2 bytes (bf16/fp16)
KV bytes per token = 2 x L x Hkv x D x Bkv = 2 x 28 x 8 x 128 x 2 = 114,688 B = 112 KiB
block of 16 tokens = 1.75 MiB
1,024-token sequence = 112 MiB
```

Contiguous reservation at `max_model_len = 2048` costs 224 MiB per admitted sequence whatever its real length. Paged allocation costs `ceil(len/16)` blocks. For a 200-token sequence that is 13 blocks, 22.75 MiB. The ratio is the fragmentation the benchmark must reproduce.

Admission rule: admit a request only if free blocks cover `ceil((prompt + max_new_tokens) / block_size)`. This reserves worst case, so decode never fails for lack of blocks. The cost is lower utilization than on-demand growth. Falsification: if measured peak blocks in use stay far below reserved blocks, the rule is too conservative and on-demand growth plus preemption becomes justified.

Budgets are separate and explicit: weights (about 1.2 GB bf16, measured at load), KV pool (configured block count), step token budget (prefill chunk plus decode tokens per step), queue capacity.

## State and failures

Sequence states:

```text
WAITING -> RUNNING_PREFILL -> RUNNING_DECODE -> FINISHED
   |             |                  |
   +-------------+------------------+--> CANCELLED | DEADLINE_EXCEEDED
WAITING --> REJECTED (queue full or request invalid)
```

- Every terminal transition frees the sequence's blocks in the same scheduler step. The invariant `free + sum(allocated) == pool_size` is checked after every step in tests.
- Cancellation arrives between steps and takes effect at the next step boundary. It never leaves a half-written block table.
- Deadlines are checked each step against a monotonic clock.
- A model-runner exception fails the whole step: all running sequences get a terminal `error` event and release blocks. The engine does not retry a failed step, because a retry could duplicate output tokens.
- Idempotency: request ids are unique. A duplicate id is rejected.
- No silent fallback. A missing device raises. A shape mismatch raises.

## Alternatives and decision

| Option | For | Against | Decision |
| --- | --- | --- | --- |
| Wrap `transformers` generate | Fast | Teaches nothing; cannot control KV | Rejected, used only as correctness oracle |
| Contiguous KV, slot per sequence | Simple | Fragmentation, the problem to study | Kept as benchmark baseline |
| Paged KV, gather then SDPA | Real allocator and block tables; portable to MPS and CPU | Gather copy cost; no fused kernel | Chosen |
| Fused Metal paged-attention kernel | Realistic speed | Large scope, unrelated to the scheduling question | Deferred |
| Admission with worst-case reservation | No mid-decode failure | Lower utilization | Chosen for v1 |
| On-demand growth with preemption | Higher utilization | Needs recompute or swap paths | Deferred, revisit trigger above |

## Verification

Correctness invariants:

1. Oracle equality. For a fixed prompt set, greedy tokens from miniserve equal greedy tokens from Hugging Face `generate` with the same weights. Required to be exact on CPU float32. On MPS bf16, report the match rate and the first divergence position. Do not claim exactness there.
2. Batch independence. A request's output tokens are identical whether it runs alone or batched with others in any arrival order.
3. Block conservation after every step and at drain.
4. Cancellation frees blocks within one step.

Negative tests: over-length request, duplicate id, queue full, deadline exceeded, cancel during prefill, cancel during decode, pool exhaustion keeps requests waiting, exception in runner.

Qualification workload: open-loop Poisson arrivals, prompt and output length distributions recorded with results, cold and warm runs reported separately.

Metrics: time to first token, inter-token latency (p50, p99), output tokens per second, queue wait, rejection rate, peak blocks in use against blocks reserved, process memory.

Engine modes compared:

- `sequential`: one request at a time.
- `static`: fixed-size batches, each held until the longest member finishes.
- `continuous`: iteration-level scheduling with the paged cache.

An external server such as vLLM or llama.cpp is not part of the comparison unless it is run on the same machine and workload. No claim about vLLM is made without that run.

## Observability and operation

Structured event log per step: step index, tokens in step, running, waiting, free blocks, step wall time. The benchmark reads this log, not wall-clock estimates.

## Phases

1. Model runner with contiguous cache. Exit: oracle equality on CPU float32.
2. Paged cache and allocator. Exit: output identical to phase 1; conservation tests pass.
3. Scheduler: continuous batching, chunked prefill, admission, cancel, deadlines. Exit: batch independence and failure tests pass.
4. Benchmark harness and three modes. Exit: `BENCHMARKS.md` with sample counts and distributions.
5. Streaming HTTP endpoint and interactive HTML explainer.
