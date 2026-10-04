# miniserve

A minimal LLM serving engine written from scratch in PyTorch to study two mechanisms: continuous batching and a paged KV cache. It runs Qwen3-0.6B on Apple Silicon (MPS) or CPU.

Status: engine, tests and benchmark harness are done. Benchmark results are in `docs/BENCHMARKS.md` once the run finishes. Design: `docs/DESIGN.md`.

## What it is

- Qwen3-0.6B forward pass (GQA, qk-norm, RoPE, SwiGLU) loaded from a pinned Hugging Face snapshot. No `transformers` model code at runtime.
- Paged KV cache: a block allocator with ownership invariants, per-sequence block tables, `index_copy_` writes and `index_select` gathers.
- Iteration-level scheduler: strict FIFO admission, chunked prefill, bounded queue, deadlines that include queue wait, cancellation, one terminal event per request.
- Benchmark arms: sequential, static batching, continuous batching, each with `max_model_len` slot or exact-length block reservation.

## What it is not

- Not fast. Attention gathers blocks then calls PyTorch SDPA. There is no fused paged-attention kernel.
- Not a vLLM replacement or comparison. vLLM was not run on the same machine and workload, so no claim is made about it.
- Greedy decoding only. One model. One trust domain.

## Verification

- Greedy output equals Hugging Face `generate` token for token on CPU float32.
- Paged output equals contiguous-cache output with scrambled block tables, reused blocks and batched padded decode.
- Block ownership is checked after every step in tests. Injected defects in the cache and the scheduler make the tests fail.

## Run

Requires Python 3.13.7 or newer, `uv`, and the Qwen3-0.6B snapshot `c1899de289a04d12100db370d81485cdf75e47ca` in the Hugging Face cache.

```bash
uv sync
uv run pytest -q
uv run ruff check . && uv run ruff format --check . && uv run pyright
uv run python scripts/microbench_mps.py --device mps --dtype bfloat16
uv run python scripts/run_matrix.py --device mps --dtype bfloat16 --n 48 --reps 3 --out bench_results/matrix_mps_bf16.jsonl
```

## Layout

- `src/miniserve/model.py`, `config.py`: model and pinned config checks.
- `src/miniserve/allocator.py`, `kv_paged.py`, `kv_contiguous.py`, `runner.py`: cache and model runner.
- `src/miniserve/engine.py`: scheduler.
- `src/miniserve/bench/`: workload, driver, metrics, configurations.
- `scripts/`: microbenchmark and matrix runner.
- `docs/`: design, microbenchmark note, benchmark report.
