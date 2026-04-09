from __future__ import annotations

from eq_generation.generators.base import BaseEQGenerator
from eq_generation.generators.gpt_generator import GPTEQGenerator


def EQGenerator(
    backend: str = "gpt",
    **kwargs,
) -> BaseEQGenerator:
    backend = backend.lower().strip()
    if backend == "gpt":
        return GPTEQGenerator(**kwargs)
    raise ValueError(f"Unknown backend: {backend}. Choose from: gpt")
