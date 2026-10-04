import random
from dataclasses import dataclass

from tokenizers import Tokenizer

TOPICS = [
    "photosynthesis",
    "the French Revolution",
    "binary search",
    "plate tectonics",
    "the water cycle",
    "supply and demand",
    "black holes",
    "the immune system",
    "public key cryptography",
    "the Roman Empire",
    "machine learning",
    "the periodic table",
    "renewable energy",
    "the printing press",
    "volcanoes",
    "the internet",
    "vaccines",
    "inflation",
    "the human heart",
    "coral reefs",
    "relational databases",
    "the Silk Road",
    "gravity",
    "the Moon",
]

TEMPLATES = [
    "Explain {t} in one sentence.",
    "Give three short facts about {t}.",
    "Write a short paragraph about {t}.",
    "What is {t}? Answer in two sentences.",
    "List five key points about {t}.",
    "Describe {t} to a ten year old.",
]

FILLER = [
    "The following notes were collected during a long afternoon of reading.",
    "Several people had asked for a clear and brief summary of the topic.",
    "Earlier discussions had focused on history, practical uses and common mistakes.",
    "The audience is a mix of students, engineers and curious readers.",
    "Please keep the answer accurate and avoid unnecessary jargon.",
    "Some background material was reviewed before this question was written.",
    "Accuracy matters more than length for this particular request.",
    "The reader has a few minutes and wants the main idea first.",
    "Examples are welcome when they make the idea easier to follow.",
    "Previous answers were criticized for being vague or too long.",
    "This request is part of a larger set of study questions.",
    "The goal is to build a short reference that can be reread quickly.",
    "Terms should be defined the first time they appear.",
    "A calm and neutral tone is preferred in the response.",
]

CONTEXT_SIZES = [0, 0, 2, 6, 12]


@dataclass(frozen=True)
class WorkItem:
    request_id: str
    prompt_token_ids: tuple[int, ...]
    max_new_tokens: int
    arrival_s: float


def chat_prompt(question: str) -> str:
    return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def build_question(rng: random.Random) -> str:
    question = rng.choice(TEMPLATES).format(t=rng.choice(TOPICS))
    sentences = rng.choice(CONTEXT_SIZES)
    if sentences == 0:
        return question
    return " ".join(rng.sample(FILLER, sentences)) + "\n\n" + question


def generate_workload(
    tokenizer: Tokenizer,
    count: int,
    seed: int,
    rate_per_s: float | None,
    max_new_tokens: int,
    max_prompt_tokens: int,
) -> list[WorkItem]:
    if count < 1 or max_new_tokens < 1 or max_prompt_tokens < 1:
        raise ValueError("count, max_new_tokens and max_prompt_tokens must be >= 1")
    if rate_per_s is not None and rate_per_s <= 0:
        raise ValueError("rate_per_s must be positive")
    rng = random.Random(seed)
    items: list[WorkItem] = []
    clock = 0.0
    for index in range(count):
        ids = tokenizer.encode(chat_prompt(build_question(rng))).ids
        while len(ids) > max_prompt_tokens:
            ids = tokenizer.encode(chat_prompt(build_question(rng))).ids
        if rate_per_s is not None:
            clock += rng.expovariate(rate_per_s)
        items.append(WorkItem(f"req-{index:04d}", tuple(ids), max_new_tokens, clock))
    return items
