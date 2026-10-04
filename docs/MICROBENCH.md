# Phase 1.5 microbenchmark: MPS decode step and block gather

Measured, one session. Reproduce: `uv run python scripts/microbench_mps.py --device mps --dtype bfloat16`. Raw output: `bench_results/microbench_mps_bf16.json`.

## Setup

- Apple M1 Max, 32 GB, torch 2.14.1, MPS, bfloat16, Qwen3-0.6B (`c1899de`).
- Context 256 tokens, 5 warmup runs, 30 timed repetitions, median reported. Synchronization after each timed unit.
- Random token inputs. The numbers measure step cost, not output quality.
- One machine, one session. Background load was not controlled. No confidence intervals.

## Results (median milliseconds)

| Batch | Decode step | Gather, advanced indexing, 56 per step | Gather, `index_select`, 56 per step | Clone, 56 per step | Trivial op, 56 per step |
| --- | --- | --- | --- | --- | --- |
| 1 | 22.16 | 2.24 | 0.81 | 0.58 | 0.55 |
| 2 | 21.02 | 3.72 | 0.92 | 0.70 | 0.53 |
| 4 | 22.57 | 6.49 | 1.11 | 0.86 | 0.52 |
| 8 | 26.78 | 12.13 | 1.46 | 1.27 | 0.55 |
| 16 | 38.24 | 23.24 | 2.03 | 1.86 | 0.53 |

Fifty-six gathers equal one decode step: keys and values for 28 layers. The gather and clone columns time that many operations in isolation, not inside the model.

## Findings

1. Decode step time is 22 ms at batch 1 and stays within 2% of that through batch 4. The weights are about 1.19 GB. At the 400 GB/s memory bandwidth listed in Apple's specification (not measured here), reading them takes about 3 ms. The gap and the flat scaling are consistent with a step dominated by per-operation overhead rather than memory bandwidth. This is an inference. No profiler run confirms it.
2. Batching reduces cost per sequence: 22.2 ms per sequence at batch 1, 2.4 ms at batch 16. On this machine the gain comes mostly from amortizing a fixed per-step cost.
3. Gathering KV blocks with advanced indexing (`pool[table]`) costs 23 ms per step at batch 16, which is 61% of the measured decode step. `torch.index_select` on the flattened block ids costs 2.0 ms, about the same as a plain clone. The paged cache uses `index_select`.
4. Gather cost with `index_select` is a copy cost. It is real but small at these sizes. It does not support or refute any claim about a fused paged kernel.

## Consequences for claims

- Do not describe decode as memory-bandwidth-bound for this setup.
- Report continuous-batching gains as measured here, with the per-step overhead explanation labeled as an inference.
- Gather overhead belongs in the benchmark's paged configurations. Cost of paged against contiguous attention is measured end to end in Phase 4, not extrapolated from this table.

## Limits

- Context 256 only. Gather cost grows with context.
- Isolated gather timings omit overlap with other GPU work.
- MPS only. CPU and other hardware may differ.
- A first version of this script timed a no-op copy and a per-call synchronization floor. Both were replaced before the numbers above were recorded.
