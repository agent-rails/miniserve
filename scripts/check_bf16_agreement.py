import argparse
import json

import torch
from tokenizers import Tokenizer

from miniserve.allocator import BlockAllocator
from miniserve.bench.configs import BLOCK_SIZE, MAX_MODEL_LEN, MAX_PROMPT_TOKENS
from miniserve.bench.workload import generate_workload
from miniserve.config import load_config, resolve_snapshot
from miniserve.engine import Engine, Request, TokenEvent
from miniserve.generate import greedy_generate
from miniserve.kv_paged import PagedKV
from miniserve.model import Qwen3
from miniserve.runner import PagedRunner


def engine_tokens(model: Qwen3, items) -> dict[str, list[int]]:
    paged = PagedKV(model.config, 256, BLOCK_SIZE, model.dtype, model.device)
    runner = PagedRunner(model, paged, BlockAllocator(256))
    engine = Engine(runner, model.config.eos_token_ids, MAX_MODEL_LEN, len(items), 256, 8, check_invariants=True)
    for item in items:
        engine.submit(Request(item.request_id, item.prompt_token_ids, item.max_new_tokens))
    out: dict[str, list[int]] = {item.request_id: [] for item in items}
    while engine.has_work:
        for event in engine.step():
            if isinstance(event, TokenEvent):
                out[event.request_id].append(event.token_id)
    return out


def first_divergence(a: list[int], b: list[int]) -> int | None:
    for index, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return index
    return None if len(a) == len(b) else min(len(a), len(b))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--new-tokens", type=int, default=48)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    snapshot = resolve_snapshot()
    config = load_config(snapshot)
    tokenizer = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
    items = generate_workload(tokenizer, args.count, args.seed, None, args.new_tokens, MAX_PROMPT_TOKENS)

    reference_model = Qwen3.load(snapshot, config, torch.device("cpu"), torch.float32)
    reference = {
        i.request_id: greedy_generate(reference_model, list(i.prompt_token_ids), i.max_new_tokens) for i in items
    }
    del reference_model

    candidate_model = Qwen3.load(snapshot, config, torch.device("mps"), torch.bfloat16)
    candidate = engine_tokens(candidate_model, items)

    rows = []
    for item in items:
        rid = item.request_id
        rows.append(
            {
                "id": rid,
                "reference_tokens": len(reference[rid]),
                "candidate_tokens": len(candidate[rid]),
                "first_divergence": first_divergence(reference[rid], candidate[rid]),
            }
        )
    identical = sum(1 for r in rows if r["first_divergence"] is None)
    diverged = [r["first_divergence"] for r in rows if r["first_divergence"] is not None]
    print(
        json.dumps(
            {
                "requests": len(rows),
                "identical": identical,
                "diverged": len(diverged),
                "first_divergence_positions": sorted(diverged),
                "new_tokens_cap": args.new_tokens,
                "rows": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
