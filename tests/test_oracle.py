import pytest
import torch
from transformers import AutoModelForCausalLM

from miniserve.generate import greedy_generate

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    ",
    "Explain why the sky is blue in one sentence:",
    "1 2 3 4 5 6 7 8 9",
]
NEW_TOKENS = 24


@pytest.fixture(scope="module")
def hf_model(snapshot):
    model = AutoModelForCausalLM.from_pretrained(str(snapshot), torch_dtype=torch.float32)
    assert not model.training
    return model


@pytest.mark.parametrize("prompt", PROMPTS)
def test_greedy_matches_hugging_face(prompt, cpu_model, tokenizer, hf_model, config):
    prompt_ids = tokenizer.encode(prompt).ids
    ours = greedy_generate(cpu_model, prompt_ids, NEW_TOKENS)
    with torch.inference_mode():
        out = hf_model.generate(
            torch.tensor([prompt_ids]),
            max_new_tokens=NEW_TOKENS,
            do_sample=False,
            eos_token_id=list(config.eos_token_ids),
            pad_token_id=config.eos_token_ids[-1],
        )
    reference = out[0, len(prompt_ids) :].tolist()
    assert ours == reference


def test_rejects_empty_prompt(cpu_model):
    with pytest.raises(ValueError):
        greedy_generate(cpu_model, [], 4)


def test_rejects_zero_new_tokens(cpu_model):
    with pytest.raises(ValueError):
        greedy_generate(cpu_model, [1, 2, 3], 0)
