import argparse
import json
import statistics
import time

import torch

from miniserve.config import load_config, resolve_snapshot
from miniserve.kv_contiguous import ContiguousKV
from miniserve.model import Qwen3, causal_mask

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def timed(fn, device: torch.device, warmup: int, reps: int) -> list[float]:
    for _ in range(warmup):
        fn()
    sync(device)
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        sync(device)
        samples.append((time.perf_counter() - start) * 1000)
    return samples


def summarize(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "median_ms": statistics.median(ordered),
        "p90_ms": ordered[int(0.9 * (len(ordered) - 1))],
        "n": len(ordered),
    }


def decode_step_bench(model: Qwen3, batch: int, context: int, warmup: int, reps: int) -> dict:
    device = model.device
    kv = ContiguousKV(model.config, batch, context + warmup + reps + 1, model.dtype, device)
    prefill_ids = torch.randint(0, model.config.vocab_size, (batch, context), device=device)
    positions = torch.arange(context, device=device)[None].expand(batch, -1)
    index = torch.full((batch,), context - 1, dtype=torch.long, device=device)
    model.forward(prefill_ids, positions, kv, causal_mask(0, context, device), index)
    kv.advance(context)
    zero = torch.zeros(batch, dtype=torch.long, device=device)

    def step() -> None:
        ids = torch.randint(0, model.config.vocab_size, (batch, 1), device=device)
        pos = torch.full((batch, 1), kv.length, dtype=torch.long, device=device)
        model.forward(ids, pos, kv, causal_mask(kv.length, 1, device), zero)
        kv.advance(1)

    return summarize(timed(step, device, warmup, reps))


def gather_bench(
    device: torch.device, dtype: torch.dtype, batch: int, context: int, block: int, warmup: int, reps: int
) -> dict:
    blocks_per_seq = context // block
    pool_blocks = batch * blocks_per_seq * 2
    pool = torch.randn(pool_blocks, block, 8, 128, device=device, dtype=dtype)
    table = torch.randperm(pool_blocks, device=device)[: batch * blocks_per_seq].view(batch, blocks_per_seq)
    contiguous = torch.randn(batch, context, 8, 128, device=device, dtype=dtype)

    per_step = 2 * 28

    flat = table.flatten()

    def gather_advanced() -> None:
        for _ in range(per_step):
            pool[table].view(batch, context, 8, 128)

    def gather() -> None:
        for _ in range(per_step):
            torch.index_select(pool, 0, flat).view(batch, context, 8, 128)

    def copy() -> None:
        for _ in range(per_step):
            contiguous.clone()

    def noop() -> None:
        for _ in range(per_step):
            table + 1

    return {
        "gather_advanced_index_per_step": summarize(timed(gather_advanced, device, warmup, reps)),
        "gather_per_step": summarize(timed(gather, device, warmup, reps)),
        "clone_per_step": summarize(timed(copy, device, warmup, reps)),
        "noop_per_step": summarize(timed(noop, device, warmup, reps)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True, choices=["mps", "cpu"])
    parser.add_argument("--dtype", required=True, choices=list(DTYPES))
    parser.add_argument("--batches", default="1,2,4,8,16")
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=30)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("mps unavailable")
    dtype = DTYPES[args.dtype]
    snapshot = resolve_snapshot()
    model = Qwen3.load(snapshot, load_config(snapshot), device, dtype)

    result = {
        "device": args.device,
        "dtype": args.dtype,
        "context": args.context,
        "torch": torch.__version__,
        "decode": {},
        "gather": {},
    }
    for batch in [int(b) for b in args.batches.split(",")]:
        result["decode"][batch] = decode_step_bench(model, batch, args.context, args.warmup, args.reps)
        result["gather"][batch] = gather_bench(device, dtype, batch, args.context, 16, args.warmup, args.reps)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
