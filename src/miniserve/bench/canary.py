import statistics
import time

import torch

from miniserve.kv_contiguous import ContiguousKV
from miniserve.model import Qwen3, causal_mask

CANARY_CONTEXT = 128
CANARY_STEPS = 24
CANARY_TOLERANCE = 1.15


def within_tolerance(baseline_ms: float, value_ms: float, tolerance: float = CANARY_TOLERANCE) -> bool:
    if baseline_ms <= 0 or value_ms <= 0 or tolerance < 1:
        raise ValueError("baseline and value must be positive and tolerance >= 1")
    return value_ms <= baseline_ms * tolerance


def measure_canary_ms(model: Qwen3) -> float:
    device = model.device
    generator = torch.Generator(device="cpu").manual_seed(0)
    kv = ContiguousKV(model.config, 1, CANARY_CONTEXT + CANARY_STEPS + 1, model.dtype, device)
    prompt = torch.randint(0, model.config.vocab_size, (1, CANARY_CONTEXT), generator=generator).to(device)
    positions = torch.arange(CANARY_CONTEXT, device=device)[None]
    index = torch.tensor([CANARY_CONTEXT - 1], device=device)
    model.forward(prompt, positions, kv, causal_mask(0, CANARY_CONTEXT, device), index)
    kv.advance(CANARY_CONTEXT)
    zero = torch.zeros(1, dtype=torch.long, device=device)
    samples: list[float] = []
    for _ in range(CANARY_STEPS):
        ids = torch.randint(0, model.config.vocab_size, (1, 1), generator=generator).to(device)
        pos = torch.full((1, 1), kv.length, dtype=torch.long, device=device)
        started = time.perf_counter()
        logits = model.forward(ids, pos, kv, causal_mask(kv.length, 1, device), zero)
        logits.to("cpu")
        samples.append((time.perf_counter() - started) * 1000)
        kv.advance(1)
    return statistics.median(samples[4:])
