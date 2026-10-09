"""Text generation backends: local HF transformers, or any OpenAI-compatible server."""

from __future__ import annotations

import json
import urllib.request
from typing import Any, Protocol


class Generator(Protocol):
    def generate(
        self, prompts: list[str], max_new_tokens: int, temperature: float = 0.0, seed: int = 0
    ) -> list[str]: ...


class HFGenerator:
    """Batched generation with a transformers model (left padding, stops at ``eos``)."""

    def __init__(self, model: Any, tokenizer: Any) -> None:
        self.model = model
        self.tok = tokenizer
        self.tok.padding_side = "left"

    def generate(self, prompts: list[str], max_new_tokens: int, temperature: float = 0.0, seed: int = 0) -> list[str]:
        import torch

        enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens, "pad_token_id": self.tok.pad_token_id}
        if temperature > 0:
            torch.manual_seed(seed)
            kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            kwargs.update(do_sample=False)
        with torch.inference_mode():
            out = self.model.generate(**enc, **kwargs)
        new = out[:, enc["input_ids"].shape[1] :]
        return [self.tok.decode(row, skip_special_tokens=False).split(self.tok.eos_token)[0] for row in new]


class OpenAICompletionsGenerator:
    """Raw-prompt generation against an OpenAI-compatible ``/v1/completions`` endpoint.

    The prompt is rendered by toolforge (same bytes as the HF backend), so llama.cpp's server-side
    tool parsing is bypassed and both backends are scored identically.
    """

    def __init__(self, endpoint: str, model_id: str = "toolforge", timeout: float = 300.0) -> None:
        self.url = endpoint.rstrip("/") + "/completions"
        self.model_id = model_id
        self.timeout = timeout

    def generate(self, prompts: list[str], max_new_tokens: int, temperature: float = 0.0, seed: int = 0) -> list[str]:
        outs = []
        for p in prompts:
            body = {
                "model": self.model_id,
                "prompt": p,
                "max_tokens": max_new_tokens,
                "temperature": temperature,
                "seed": seed,
                "stop": ["<|im_end|>"],
            }
            req = urllib.request.Request(
                self.url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
                data = json.loads(resp.read())
            outs.append(data["choices"][0]["text"])
        return outs
