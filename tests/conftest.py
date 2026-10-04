import pytest
import torch
from tokenizers import Tokenizer

from miniserve.config import load_config, resolve_snapshot
from miniserve.model import Qwen3


@pytest.fixture(scope="session")
def snapshot():
    return resolve_snapshot()


@pytest.fixture(scope="session")
def config(snapshot):
    return load_config(snapshot)


@pytest.fixture(scope="session")
def tokenizer(snapshot):
    return Tokenizer.from_file(str(snapshot / "tokenizer.json"))


@pytest.fixture(scope="session")
def cpu_model(snapshot, config):
    return Qwen3.load(snapshot, config, torch.device("cpu"), torch.float32)
