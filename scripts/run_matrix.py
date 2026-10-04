import argparse
import gc
import json
import platform
import statistics
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

from miniserve.allocator import BlockAllocator
from miniserve.bench.canary import CANARY_TOLERANCE, measure_canary_ms, within_tolerance
from miniserve.bench.configs import (
    BLOCK_SIZE,
    CONFIGS,
    MAX_MODEL_LEN,
    MAX_NEW_TOKENS,
    MAX_PROMPT_TOKENS,
    PREFILL_CHUNK,
    STATIC_WAIT_S,
    EngineConfig,
)
from miniserve.bench.driver import run_workload
from miniserve.bench.metrics import summarize
from miniserve.bench.workload import WorkItem, generate_workload
from miniserve.config import load_config, resolve_snapshot
from miniserve.engine import Engine
from miniserve.generate import greedy_generate
from miniserve.kv_paged import PagedKV
from miniserve.model import Qwen3
from miniserve.runner import PagedRunner

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
AMPLE_POOL_BLOCKS = 512
MAX_ATTEMPTS = 6
RETRY_SLEEP_S = 60.0
BASELINE_CANARIES = 5
SWEEP_POOLS = [32, 48, 96, 192]


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def build_engine(model: Qwen3, spec: EngineConfig, pool_blocks: int, queue_capacity: int) -> Engine:
    config = model.config
    paged = PagedKV(config, pool_blocks, BLOCK_SIZE, model.dtype, model.device)
    runner = PagedRunner(model, paged, BlockAllocator(pool_blocks))
    return Engine(
        runner,
        config.eos_token_ids,
        max_model_len=MAX_MODEL_LEN,
        queue_capacity=queue_capacity,
        prefill_chunk=PREFILL_CHUNK,
        max_running=spec.max_running,
        check_invariants=True,
        policy=spec.policy,
        reservation=spec.reservation,
        static_wait_s=STATIC_WAIT_S,
    )


def run_cell(model: Qwen3, spec: EngineConfig, pool_blocks: int, items: list[WorkItem]) -> dict[str, Any]:
    engine = build_engine(model, spec, pool_blocks, queue_capacity=len(items))
    result = run_workload(engine, items)
    summary = summarize(result)
    del engine
    gc.collect()
    if model.device.type == "mps":
        torch.mps.empty_cache()
    return summary


def gated_cell(
    model: Qwen3,
    spec: EngineConfig,
    pool_blocks: int,
    items: list[WorkItem],
    baseline_ms: float,
    emit: Callable[..., None],
) -> dict[str, Any]:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        before = measure_canary_ms(model)
        summary = run_cell(model, spec, pool_blocks, items)
        after = measure_canary_ms(model)
        valid = within_tolerance(baseline_ms, before) and within_tolerance(baseline_ms, after)
        gate = {"canary_before_ms": before, "canary_after_ms": after, "attempt": attempt, "valid": valid}
        if valid:
            return {"summary": summary, **gate}
        emit("canary_retry", config=spec.name, pool_blocks=pool_blocks, baseline_ms=baseline_ms, **gate)
        time.sleep(RETRY_SLEEP_S)
    return {"summary": summary, **gate}


def workload_stats(items: list[WorkItem]) -> dict[str, float]:
    lengths = [len(i.prompt_token_ids) for i in items]
    return {"mean_prompt_tokens": sum(lengths) / len(lengths), "max_prompt_tokens": max(lengths)}


def paging_overhead(model: Qwen3, items: list[WorkItem]) -> dict[str, float]:
    sync(model.device)
    start = time.perf_counter()
    contiguous_tokens = 0
    for item in items:
        contiguous_tokens += len(greedy_generate(model, list(item.prompt_token_ids), item.max_new_tokens))
    sync(model.device)
    contiguous_s = time.perf_counter() - start

    engine = build_engine(model, CONFIGS["sequential"], AMPLE_POOL_BLOCKS, queue_capacity=len(items))
    closed = [WorkItem(i.request_id, i.prompt_token_ids, i.max_new_tokens, 0.0) for i in items]
    result = run_workload(engine, closed)
    paged_tokens = sum(r.output_tokens for r in result.records.values())
    paged_s = result.ended - result.started
    return {
        "requests": len(items),
        "contiguous_tokens": contiguous_tokens,
        "contiguous_s": contiguous_s,
        "contiguous_tokens_per_s": contiguous_tokens / contiguous_s,
        "paged_tokens": paged_tokens,
        "paged_s": paged_s,
        "paged_tokens_per_s": paged_tokens / paged_s,
    }


def environment(args: argparse.Namespace) -> dict[str, object]:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", ".", ":!bench_results"], capture_output=True, text=True, check=True
    ).stdout.strip()
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    return {
        "git_sha": sha,
        "git_dirty": bool(dirty),
        "chip": chip,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": args.device,
        "dtype": args.dtype,
        "requests_per_run": args.n,
        "reps": args.reps,
        "base_seed": args.seed,
        "rate_multipliers": args.mults,
        "block_size": BLOCK_SIZE,
        "max_model_len": MAX_MODEL_LEN,
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "prefill_chunk": PREFILL_CHUNK,
        "static_wait_s": STATIC_WAIT_S,
        "canary_tolerance": CANARY_TOLERANCE,
        "max_attempts": MAX_ATTEMPTS,
        "ample_pool_blocks": AMPLE_POOL_BLOCKS,
        "sweep_pools": SWEEP_POOLS,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True, choices=["mps", "cpu"])
    parser.add_argument("--dtype", required=True, choices=list(DTYPES))
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--mults", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0])
    parser.add_argument("--sweep-mult", type=float, default=8.0)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("mps unavailable")
    snapshot = resolve_snapshot()
    model = Qwen3.load(snapshot, load_config(snapshot), device, DTYPES[args.dtype])
    tokenizer = Tokenizer.from_file(str(snapshot / "tokenizer.json"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as sink:

        def emit(kind: str, **fields: object) -> None:
            sink.write(json.dumps({"kind": kind, **fields}) + "\n")
            sink.flush()
            print(kind, {k: v for k, v in fields.items() if k != "summary"}, flush=True)

        emit("environment", **environment(args))

        def make_items(seed: int, rate: float | None, count: int) -> list[WorkItem]:
            return generate_workload(tokenizer, count, seed, rate, MAX_NEW_TOKENS, MAX_PROMPT_TOKENS)

        warm = make_items(args.seed - 1, None, 6)
        emit("cold_run", config="sequential", summary=run_cell(model, CONFIGS["sequential"], AMPLE_POOL_BLOCKS, warm))
        emit("warm_run", config="sequential", summary=run_cell(model, CONFIGS["sequential"], AMPLE_POOL_BLOCKS, warm))

        baseline = statistics.median([measure_canary_ms(model) for _ in range(BASELINE_CANARIES)])
        emit("canary_baseline", baseline_ms=baseline)

        calibration = make_items(args.seed, None, args.n)
        capacity = run_cell(model, CONFIGS["sequential"], AMPLE_POOL_BLOCKS, calibration)
        capacity_rps = capacity["requests_per_s"]
        emit(
            "calibration",
            config="sequential",
            capacity_requests_per_s=capacity_rps,
            workload=workload_stats(calibration),
            summary=capacity,
        )
        emit("paging_overhead", **paging_overhead(model, calibration[:8]))

        for rep in range(args.reps):
            seed = args.seed + 1 + rep
            for mult in args.mults:
                items = make_items(seed, mult * capacity_rps, args.n)
                for spec in CONFIGS.values():
                    gated = gated_cell(model, spec, AMPLE_POOL_BLOCKS, items, baseline, emit)
                    emit(
                        "matrix",
                        config=spec.name,
                        rep=rep,
                        seed=seed,
                        mult=mult,
                        rate_per_s=mult * capacity_rps,
                        pool_blocks=AMPLE_POOL_BLOCKS,
                        workload=workload_stats(items),
                        **gated,
                    )

            items = make_items(seed, args.sweep_mult * capacity_rps, args.n)
            for pool in SWEEP_POOLS:
                for name in ("continuous-maxlen", "continuous-exact", "static-maxlen", "static-exact"):
                    gated = gated_cell(model, CONFIGS[name], pool, items, baseline, emit)
                    emit(
                        "sweep",
                        config=name,
                        rep=rep,
                        seed=seed,
                        mult=args.sweep_mult,
                        rate_per_s=args.sweep_mult * capacity_rps,
                        pool_blocks=pool,
                        **gated,
                    )


if __name__ == "__main__":
    main()
