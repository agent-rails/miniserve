from dataclasses import dataclass
from typing import Literal

BLOCK_SIZE = 16
MAX_MODEL_LEN = 384
MAX_PROMPT_TOKENS = 220
MAX_NEW_TOKENS = 96
PREFILL_CHUNK = 256
BATCH_MAX_RUNNING = 16
STATIC_WAIT_S = 0.25


@dataclass(frozen=True)
class EngineConfig:
    name: str
    policy: Literal["continuous", "static"]
    reservation: Literal["exact", "max_len"]
    max_running: int


CONFIGS: dict[str, EngineConfig] = {
    c.name: c
    for c in (
        EngineConfig("sequential", "continuous", "exact", 1),
        EngineConfig("static-maxlen", "static", "max_len", BATCH_MAX_RUNNING),
        EngineConfig("static-exact", "static", "exact", BATCH_MAX_RUNNING),
        EngineConfig("continuous-maxlen", "continuous", "max_len", BATCH_MAX_RUNNING),
        EngineConfig("continuous-exact", "continuous", "exact", BATCH_MAX_RUNNING),
    )
}
