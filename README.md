# miniserve

A small LLM serving engine, built from scratch, to show how two key ideas in modern LLM serving work: **continuous batching** and a **paged KV cache**.

It runs a real model (Qwen3-0.6B) on a Mac or a CPU. It is small enough to read in an afternoon.

Full results: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md). Design decisions: [`docs/DESIGN.md`](docs/DESIGN.md).

## The problem in two minutes

A language model writes its answer one token at a time. Two things make this hard to do for many users at once.

**1. The model must remember everything it has read so far.**
It keeps this memory in the *KV cache*. The cache grows with every token, and each user has their own. Memory runs out fast if you reserve too much per user.

**2. Users finish at different times.**
If you group users into a batch and wait for the slowest one, fast users sit idle.

miniserve fixes both:

| Idea | Plain meaning | What it fixes |
| --- | --- | --- |
| Paged KV cache | Hand out memory in small fixed blocks, not one big slab per user. | Wasted memory. |
| Continuous batching | Let new users join, and finished users leave, after every single token step. | Idle waiting. |

```text
Slab (old way)                       Paged (miniserve)
user A  [#####.................]      pool of small blocks:
user B  [##...................]      [A][A][B][A][C][B][ ][ ][C][ ]
user C  [#######...............]      each user keeps a short list of its blocks
        '.' = reserved, unused       A: 0,1,3   B: 2,5   C: 4,8
```

## Results at a glance

Measured on one Apple M1 Max, one complete run, 48 requests per cell. Read the limits in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) before quoting any number.

| Question | Answer |
| --- | --- |
| How much more work per second with continuous batching? | 5.8 times sequential, 1.45 times static batching (at 8 times the sequential load). |
| How long does a user wait for the first token? | Median 60 ms with continuous batching, 1.7 s with static batching, 28 s one at a time (at 2 times load). |
| What does batching cost? | Each user's tokens arrive slower: median gap 21 ms alone, 43 ms at the highest load. |
| When does exact-length block reservation help? | Only when memory is tight. With a small pool it gave 1.2 to 2.2 times the throughput of reserving the maximum for everyone. With a big pool it made no difference. |
| What does paging cost? | About 2.3 to 2.8% slower at one user at a time. |

## What is in this repo

- **The model.** Qwen3-0.6B written directly in PyTorch. Its output matches Hugging Face token for token (tested on CPU).
- **The cache.** A block allocator that never gives one block to two users, plus a paged KV cache.
- **The scheduler.** It decides who runs each step. Requests are served in arrival order. Long prompts are split into chunks. Requests can be cancelled or given a deadline. Every request ends with exactly one result: finished, cancelled, deadline exceeded, rejected, or error.
- **A benchmark.** It compares five setups on the same random workload (see below).

## The five setups we compare

| Setup | Who runs together | Memory reserved per user |
| --- | --- | --- |
| `sequential` | one user at a time | exactly what the request needs |
| `static-maxlen` | fixed groups; the group waits for its slowest user | the maximum possible length |
| `static-exact` | fixed groups | exactly what the request needs |
| `continuous-maxlen` | users join and leave every step | the maximum possible length |
| `continuous-exact` | users join and leave every step | exactly what the request needs |

Changing one thing at a time shows which idea causes which effect.

## Honest limits

- **It is not fast.** There is no custom GPU kernel. Attention copies the blocks it needs, then calls the standard PyTorch routine.
- **No vLLM comparison.** vLLM was not run on this machine, so this project makes no claim about it.
- **One model, greedy decoding, one machine** (Apple M1 Max, MPS). Numbers will differ on other hardware.
- **Reservation is simple.** A request reserves its worst case up front. This captures only part of what paging can give. The design doc explains what is missing and when to add it.

## How we know it works

- Output equals Hugging Face output token for token (CPU, float32).
- Paged output equals non-paged output, even when blocks are shuffled, reused, or batched with padding.
- After every scheduler step, tests check that each block has exactly one owner.
- We injected bugs on purpose (a wrong write slot, a wrong mask, blocks never freed, deadlines off, queue-jumping). The tests failed each time, so they can catch real mistakes.

## Run it

You need Python 3.13.7 or newer, [`uv`](https://docs.astral.sh/uv/), and the model files for `Qwen/Qwen3-0.6B` (revision `c1899de289a04d12100db370d81485cdf75e47ca`) in your Hugging Face cache.

```bash
uv sync                                   # install
uv run pytest -q                          # run all tests
uv run ruff check . && uv run pyright     # lint and type-check

uv run python scripts/microbench_mps.py --device mps --dtype bfloat16
uv run python scripts/run_matrix.py --device mps --dtype bfloat16 \
    --n 48 --reps 3 --out bench_results/matrix_mps_bf16.jsonl
```

The full benchmark takes a couple of hours. Do not run other GPU work at the same time.

## Where to look in the code

| File | What it does |
| --- | --- |
| `src/miniserve/model.py` | The model forward pass. |
| `src/miniserve/allocator.py` | Hands out cache blocks and checks ownership. |
| `src/miniserve/kv_paged.py` | The paged cache: write and read blocks. |
| `src/miniserve/runner.py` | Runs one prefill chunk or one batched decode step. |
| `src/miniserve/engine.py` | The scheduler. Start here to see how requests flow. |
| `src/miniserve/bench/` | Workload generator, metrics, and the benchmark driver. |
| `scripts/` | Microbenchmark and full benchmark runner. |

## Words used here

- **Token**: a piece of text, roughly a word or part of a word.
- **Prefill**: the model reads your prompt. Done once, in chunks.
- **Decode**: the model writes its answer, one token per step.
- **KV cache**: the model's memory of what it has read so far.
- **TTFT**: time to first token. How long a user waits to see anything.
- **Inter-token latency**: the gap between tokens while an answer streams.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
