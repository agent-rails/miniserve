import pytest

from miniserve.bench.workload import generate_workload

IM_START = 151644


def make(tokenizer, **overrides):
    params = {"count": 40, "seed": 1, "rate_per_s": 2.0, "max_new_tokens": 32, "max_prompt_tokens": 220}
    params.update(overrides)
    return generate_workload(tokenizer, **params)


def test_same_seed_is_identical(tokenizer):
    assert make(tokenizer) == make(tokenizer)


def test_different_seed_differs(tokenizer):
    assert make(tokenizer, seed=2) != make(tokenizer)


def test_arrivals_are_increasing_for_open_loop(tokenizer):
    times = [i.arrival_s for i in make(tokenizer)]
    assert times == sorted(times) and times[0] > 0 and len(set(times)) == len(times)


def test_closed_loop_arrives_at_zero(tokenizer):
    assert {i.arrival_s for i in make(tokenizer, rate_per_s=None)} == {0.0}


def test_prompts_use_chat_template_and_respect_limit(tokenizer):
    items = make(tokenizer, max_prompt_tokens=120)
    assert all(i.prompt_token_ids[0] == IM_START for i in items)
    assert all(len(i.prompt_token_ids) <= 120 for i in items)


def test_prompt_lengths_vary(tokenizer):
    lengths = {len(i.prompt_token_ids) for i in make(tokenizer, count=80)}
    assert len(lengths) > 10 and max(lengths) > 2 * min(lengths)


def test_ids_unique(tokenizer):
    items = make(tokenizer)
    assert len({i.request_id for i in items}) == len(items)


@pytest.mark.parametrize(
    "overrides",
    [{"count": 0}, {"max_new_tokens": 0}, {"max_prompt_tokens": 0}, {"rate_per_s": 0.0}, {"rate_per_s": -1.0}],
)
def test_invalid_parameters_rejected(tokenizer, overrides):
    with pytest.raises(ValueError):
        make(tokenizer, **overrides)
