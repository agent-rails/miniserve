# Benchmark results

What was measured: five ways of running the same small model on one Mac, under the same random stream of requests. The aim is to show what continuous batching and exact-length block reservation each change, one at a time.

Hardware and software: Apple M1 Max (32 GB), MPS, bfloat16, torch 2.14.1, Qwen3-0.6B. Sequential capacity on this machine is 0.703 requests per second (45.6 output tokens per second). Arrival rates below are multiples of that figure.

**Sample size: one complete repetition.** Three were planned. Repetition 0 finished with all 36 cells valid. The run was stopped during repetition 1, which has 2 valid cells. Almost every number below is a single run. Treat differences under about 10% as noise. The large effects reported here are far larger than that.

## What we found

1. **Continuous batching raises throughput and cuts waiting by a lot.** At 8 times the sequential capacity, continuous batching produced 267 to 269 tokens per second. Sequential produced 46 (5.8 times less). Static batching produced 186 (continuous is 1.45 times higher).
2. **Users wait far less to see the first token.** At 2 times capacity the median time to first token was 28 s for sequential, 1.7 s for static batching, and 60 ms for continuous batching. At 4 times: 33 s, 2.3 to 2.6 s, and 77 to 80 ms. Past that point (8 times) all configurations are overloaded and queues build.
3. **Batching costs speed per user.** The median gap between tokens rose from 21 ms (sequential) to 43 ms at 8 times capacity, and the p99 gap from 24 ms to about 115 ms. Throughput and per-user speed trade against each other.
4. **Static batching wastes work.** At 8 times capacity, 3.67 of about 12 rows in each static step were requests that had already finished but still held their place. Continuous batching has none. At light load (1 times) static and continuous give similar throughput (61 against 63 tokens per second), but static users wait about 0.9 to 1.0 s for the first token against 58 to 71 ms.
5. **Exact-length reservation matters only when memory is tight.** With a large pool (512 blocks) the two reservation policies were within 1% of each other. With a small pool, reserving exactly what a request needs beat reserving the maximum length for every request:

   | Pool (blocks) | Continuous, exact vs max-length throughput | Blocks holding tokens / reserved (max-length, exact) |
   | --- | --- | --- |
   | 32 | 2.2 times | 25%, 62% |
   | 48 | 1.8 times | 25%, 62% |
   | 96 | 1.5 times | 25%, 62% |
   | 192 | 1.24 times | 25%, 62% |

   At 32 blocks a max-length slot is 24 blocks, so only one request fits. That configuration ran at the sequential speed (46 tokens per second) with 0.99 requests running.
6. **Paging overhead at batch 1 was small: 2.3% to 2.8% slower** than a contiguous cache, in two separate 8-request runs (2.8% in the final run; 2.3% in the discarded attempt, measured before its slowdown began). No repetitions of either.
7. **Exact-length reservation still leaves slack.** Even with exact reservation only 62% of reserved blocks held tokens. The rest is space reserved for tokens that were never generated because answers ended early. This design does not recover it. Doing so needs growth on demand and preemption, which the design document defers.

## What this does not show

- Nothing about vLLM or any other server. None was run on this machine.
- Nothing about other hardware, other models, or a fused attention kernel.
- No claim that paging is faster than a contiguous cache. At batch 1 it was slightly slower.
- No claim about large-batch paging cost beyond the microbenchmark in `MICROBENCH.md`.
- Static-batching numbers depend on two untuned settings: batch size 16 and a 0.25 s formation wait.

## How the test worked

- Workload: 48 requests per cell, Poisson arrivals, chat-template prompts of varying length (mean 68 tokens, at most 220) with answers up to 96 tokens that end on the model's stop token. The same seed gives the same requests in every configuration.
- Arrival time is the scheduled time, not the time the driver submitted the request. A slow step cannot hide queueing delay.
- Pool sweep: pool size set to 32, 48, 96 and 192 blocks of 16 tokens at 8 times capacity.
- Raw results: `bench_results/matrix_mps_bf16.jsonl`. Tables below come from `scripts/analyze_matrix.py`.

## Validity and limits

- **A first attempt was discarded.** Its later cells ran up to four times slower than the same cells earlier (single-sequence step time rose from 21 ms to 82 ms), while swap was heavily used and other apps were open. Those results are kept as `bench_results/attempt1_contaminated.jsonl` and are not used.
- **Machine check.** Before and after every cell, the run measured a fixed single-sequence decode step (the canary, baseline 18.5 ms). A cell was rerun if either reading exceeded 1.15 times the baseline, up to 6 attempts.
- **The check was too strict.** Readings taken just after a cell are about 14% higher than before it, which sits on the limit. This caused 40 reruns and 2 cells (static, 1 times, repetition 1) to be excluded after 6 attempts, with readings 27% to 31% above baseline. The largest accepted after-reading was 21.290 ms against a 21.292 ms limit.
- **What the check covers.** It measures one single-sequence decode step before and after a cell. It cannot see interference in the middle of a cell or at larger batch sizes. After 40 reruns, kept cells may be the more favorable repeats.
- **Fixed cell order.** Every repetition ran configurations in the same order, so slow drift in the machine is confounded with configuration.
- **Timing.** Step times use a monotonic clock read after each step, with no explicit MPS synchronize call. The read-back of chosen tokens to the CPU synchronizes implicitly.
- **Not isolated.** Chrome, Telegram and a VPN client stayed open, and macOS background services ran. OrbStack and Cursor were closed shortly after the run started.
- **Numerics.** On the benchmark's own prompts, bfloat16 on MPS matched the CPU float32 reference exactly for 7 of 16 sequences over 48 tokens. The other 9 diverged, first at tokens 12 to 45 (`bench_results/bf16_agreement.json`). Outputs, and so answer lengths, differ slightly between configurations because batch composition changes rounding. At 8 times capacity total output tokens ranged from 3,316 to 3,358 across configurations (about 1.3%).
- **Percentiles.** 48 requests per cell means the p99 is close to the maximum.
- **The 1 times column is volatile.** Near full capacity, queue delay depends heavily on the arrival sequence. For example, sequential median time to first token was 1.5 s in one repetition and 16.9 s in another. The 1 times means mix two different arrival sequences.

## Full results

Cells show the mean across repetitions that have valid data, with the range in parentheses when there is more than one.

### Run identity

- Code: `95c6c21`, chip Apple M1 Max, torch 2.14.1, device mps, dtype bfloat16
- 48 requests per cell, base seed 100, 3 repetitions planned
- Valid cells per repetition: rep 0: 36, rep 1: 2
- Block size 16, max_model_len 384, max_new_tokens 96, static wait 0.25 s
- Sequential capacity: 0.703 requests/s, 45.6 output tokens/s, mean prompt 68 tokens
- Machine canary baseline: 18.5 ms per single-sequence decode step; cells outside 1.15x before or after were rerun up to 6 times
- Cells excluded as invalid after all attempts: 2; reruns triggered: 40

### Paging overhead at batch 1

8 requests: contiguous cache 46.8 tokens/s, paged engine 45.5 tokens/s (+2.8% slower for paged). Single run, no repetitions.

### Arrival-rate matrix

Cells show the mean across repetitions, with the range in parentheses.

#### Output tokens per second

| Config | 1x | 2x | 4x | 8x |
| --- | --- | --- | --- | --- |
| sequential | 40.4 (34.4-46.3) | 46.3 | 46.0 | 46.2 |
| static-maxlen | 61.2 | 118.3 | 175.5 | 185.6 |
| static-exact | 61.2 | 118.3 | 175.8 | 185.9 |
| continuous-maxlen | 48.7 (34.4-63.1) | 121.1 | 228.4 | 266.9 |
| continuous-exact | 63.1 | 121.1 | 230.0 | 269.1 |

#### Time to first token, median (ms)

| Config | 1x | 2x | 4x | 8x |
| --- | --- | --- | --- | --- |
| sequential | 9184 (1508-16859) | 28074 | 33338 | 35719 |
| static-maxlen | 970 | 1676 | 2308 | 5096 |
| static-exact | 905 | 1696 | 2555 | 5099 |
| continuous-maxlen | 71 (57-85) | 59 | 77 | 2247 |
| continuous-exact | 58 | 60 | 80 | 2237 |

#### Time to first token, p99 (ms)

| Config | 1x | 2x | 4x | 8x |
| --- | --- | --- | --- | --- |
| sequential | 14247 (7839-20655) | 44467 | 57640 | 63768 |
| static-maxlen | 3036 | 4132 | 5467 | 8705 |
| static-exact | 2975 | 4158 | 5762 | 8702 |
| continuous-maxlen | 187 (169-206) | 144 | 859 | 3692 |
| continuous-exact | 165 | 142 | 826 | 3678 |

#### Inter-token latency, median (ms)

| Config | 1x | 2x | 4x | 8x |
| --- | --- | --- | --- | --- |
| sequential | 21.2 (20.9-21.5) | 21.0 | 21.2 | 21.1 |
| static-maxlen | 22.9 | 26.6 | 40.9 | 42.1 |
| static-exact | 23.0 | 26.7 | 42.0 | 42.0 |
| continuous-maxlen | 22.5 (21.8-23.1) | 25.3 | 35.9 | 43.0 |
| continuous-exact | 23.4 | 25.5 | 37.9 | 42.9 |

#### Inter-token latency, p99 (ms)

| Config | 1x | 2x | 4x | 8x |
| --- | --- | --- | --- | --- |
| sequential | 27.0 (23.9-30.2) | 24.7 | 24.7 | 24.1 |
| static-maxlen | 59.0 | 81.7 | 109.6 | 93.6 |
| static-exact | 60.4 | 80.2 | 114.6 | 93.9 |
| continuous-maxlen | 68.0 (58.0-78.0) | 92.4 | 103.5 | 114.2 |
| continuous-exact | 79.2 | 92.5 | 102.5 | 116.0 |

#### Queue wait, median (ms)

| Config | 1x | 2x | 4x | 8x |
| --- | --- | --- | --- | --- |
| sequential | 9141 (1459-16823) | 28037 | 33301 | 35681 |
| static-maxlen | 881 | 1253 | 1987 | 4350 |
| static-exact | 857 | 1262 | 2184 | 4355 |
| continuous-maxlen | 11 (10-11) | 13 | 22 | 2167 |
| continuous-exact | 15 | 11 | 19 | 2157 |

#### Mean sequences running per step

| Config | 1x | 2x | 4x | 8x |
| --- | --- | --- | --- | --- |
| sequential | 0.98 (0.98-0.99) | 0.99 | 0.99 | 0.99 |
| static-maxlen | 2.11 | 4.53 | 9.83 | 12.14 |
| static-exact | 2.11 | 4.53 | 9.82 | 12.14 |
| continuous-maxlen | 1.67 (1.36-1.98) | 3.41 | 8.39 | 11.59 |
| continuous-exact | 2.00 | 3.43 | 8.55 | 11.81 |

#### Mean finished-but-held rows per step

| Config | 1x | 2x | 4x | 8x |
| --- | --- | --- | --- | --- |
| sequential | 0.00 (0.00-0.00) | 0.00 | 0.00 | 0.00 |
| static-maxlen | 0.34 | 1.19 | 2.91 | 3.67 |
| static-exact | 0.34 | 1.19 | 2.78 | 3.67 |
| continuous-maxlen | 0.00 (0.00-0.00) | 0.00 | 0.00 | 0.00 |
| continuous-exact | 0.00 | 0.00 | 0.00 | 0.00 |

#### Rejection rate

| Config | 1x | 2x | 4x | 8x |
| --- | --- | --- | --- | --- |
| sequential | 0.000 (0.000-0.000) | 0.000 | 0.000 | 0.000 |
| static-maxlen | 0.000 | 0.000 | 0.000 | 0.000 |
| static-exact | 0.000 | 0.000 | 0.000 | 0.000 |
| continuous-maxlen | 0.000 (0.000-0.000) | 0.000 | 0.000 | 0.000 |
| continuous-exact | 0.000 | 0.000 | 0.000 | 0.000 |

### Pool-size sweep at the highest arrival rate

#### Output tokens per second

| Config | 32 blocks | 48 blocks | 96 blocks | 192 blocks |
| --- | --- | --- | --- | --- |
| static-maxlen | 46.0 | 67.8 | 109.2 | 149.6 |
| static-exact | 85.3 | 113.6 | 160.2 | 185.3 |
| continuous-maxlen | 46.1 | 83.5 | 146.4 | 214.3 |
| continuous-exact | 102.4 | 150.8 | 225.6 | 265.1 |

#### Queue wait, median (ms)

| Config | 32 blocks | 48 blocks | 96 blocks | 192 blocks |
| --- | --- | --- | --- | --- |
| static-maxlen | 35823 | 22215 | 12487 | 7522 |
| static-exact | 17229 | 12405 | 7072 | 4320 |
| continuous-maxlen | 35255 | 18015 | 8799 | 4503 |
| continuous-exact | 14259 | 8925 | 3806 | 2209 |

#### Time to first token, p99 (ms)

| Config | 32 blocks | 48 blocks | 96 blocks | 192 blocks |
| --- | --- | --- | --- | --- |
| static-maxlen | 64077 | 40950 | 22003 | 13922 |
| static-exact | 31123 | 20798 | 12041 | 8693 |
| continuous-maxlen | 63854 | 31798 | 14781 | 7140 |
| continuous-exact | 24265 | 13921 | 6120 | 3787 |

#### Mean sequences running per step

| Config | 32 blocks | 48 blocks | 96 blocks | 192 blocks |
| --- | --- | --- | --- | --- |
| static-maxlen | 0.99 | 1.98 | 3.74 | 6.82 |
| static-exact | 2.66 | 4.02 | 7.95 | 12.14 |
| continuous-maxlen | 0.99 | 1.97 | 3.89 | 7.07 |
| continuous-exact | 2.56 | 4.18 | 7.96 | 11.81 |

#### Mean blocks reserved

| Config | 32 blocks | 48 blocks | 96 blocks | 192 blocks |
| --- | --- | --- | --- | --- |
| static-maxlen | 23.7 | 47.4 | 89.7 | 163.7 |
| static-exact | 26.5 | 39.6 | 80.0 | 122.0 |
| continuous-maxlen | 23.7 | 47.2 | 93.3 | 169.6 |
| continuous-exact | 24.9 | 40.7 | 77.4 | 114.8 |

#### Mean blocks holding tokens

| Config | 32 blocks | 48 blocks | 96 blocks | 192 blocks |
| --- | --- | --- | --- | --- |
| static-maxlen | 5.9 | 12.5 | 23.9 | 43.2 |
| static-exact | 16.7 | 25.0 | 50.0 | 75.5 |
| continuous-maxlen | 5.9 | 11.8 | 23.4 | 42.6 |
| continuous-exact | 15.4 | 25.2 | 47.9 | 70.8 |

#### Block utilization (holding tokens / reserved)

| Config | 32 blocks | 48 blocks | 96 blocks | 192 blocks |
| --- | --- | --- | --- | --- |
| static-maxlen | 0.25 | 0.26 | 0.27 | 0.26 |
| static-exact | 0.63 | 0.63 | 0.63 | 0.62 |
| continuous-maxlen | 0.25 | 0.25 | 0.25 | 0.25 |
| continuous-exact | 0.62 | 0.62 | 0.62 | 0.62 |

