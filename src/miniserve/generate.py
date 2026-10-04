import torch

from miniserve.kv_contiguous import ContiguousKV
from miniserve.model import Qwen3, causal_mask


def greedy_generate(model: Qwen3, prompt_ids: list[int], max_new_tokens: int) -> list[int]:
    if not prompt_ids:
        raise ValueError("empty prompt")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")
    device = model.device
    prompt_len = len(prompt_ids)
    kv = ContiguousKV(model.config, 1, prompt_len + max_new_tokens, model.dtype, device)
    eos = set(model.config.eos_token_ids)

    ids = torch.tensor([prompt_ids], device=device)
    positions = torch.arange(prompt_len, device=device)[None]
    mask = causal_mask(0, prompt_len, device)
    logit_index = torch.tensor([prompt_len - 1], device=device)
    logits = model.forward(ids, positions, kv, mask, logit_index)
    kv.advance(prompt_len)

    out: list[int] = []
    for _ in range(max_new_tokens):
        token = int(logits.argmax(dim=-1).item())
        out.append(token)
        if token in eos or len(out) == max_new_tokens:
            break
        step_ids = torch.tensor([[token]], device=device)
        step_pos = torch.tensor([[kv.length]], device=device)
        step_mask = causal_mask(kv.length, 1, device)
        logits = model.forward(step_ids, step_pos, kv, step_mask, torch.zeros(1, dtype=torch.long, device=device))
        kv.advance(1)
    return out
