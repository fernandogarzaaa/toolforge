"""Compatibility shims for running TRL on machines without Triton (CPU-only, aarch64).

Recent TRL computes DPO log-probs through a fused, chunked LM-head kernel written in Triton. When
Triton is unavailable it leaves the kernel as ``None`` and DPO crashes. :func:`patch_trl_logprob`
installs a plain-PyTorch implementation with the same contract. It is only used when the Triton
kernel is missing, and it is exact (it materialises logits only for the supervised tokens).
"""

from __future__ import annotations

from typing import Any

OUTPUT_ORDER = ("entropy", "log_sum_sq_probs", "mean_logits", "is_top1")


class TorchChunkedLogProb:
    """Drop-in for ``trl.kernels.ChunkedLogProbFunction`` (forward and autograd via plain torch ops)."""

    @staticmethod
    def apply(
        hidden: Any,
        weight: Any,
        bias: Any,
        targets: Any,
        temperature: float,
        chunk_size: int,
        final_logit_softcapping: float | None = None,
        logit_scale: float = 1.0,
        outputs: tuple[str, ...] = OUTPUT_ORDER,
    ) -> tuple[Any, ...]:
        import torch

        want = set(outputs or ())
        logps: list[Any] = []
        extras: dict[str, list[Any]] = {k: [] for k in OUTPUT_ORDER}
        n = hidden.shape[0]
        step = max(1, int(chunk_size) if chunk_size else n)
        for start in range(0, n, step):
            h = hidden[start : start + step].float()
            logits = h @ weight.float().t()
            if bias is not None:
                logits = logits + bias.float()
            logits = logits * logit_scale
            if final_logit_softcapping is not None:
                logits = torch.tanh(logits / final_logit_softcapping) * final_logit_softcapping
            logits = logits / temperature
            lsm = torch.log_softmax(logits, dim=-1)
            tgt = targets[start : start + step]
            logps.append(lsm.gather(-1, tgt.unsqueeze(-1)).squeeze(-1))
            with torch.no_grad():
                if "entropy" in want:
                    extras["entropy"].append(-(lsm.exp() * lsm).sum(-1))
                if "log_sum_sq_probs" in want:
                    extras["log_sum_sq_probs"].append(torch.logsumexp(2 * lsm, dim=-1))
                if "mean_logits" in want:
                    extras["mean_logits"].append(logits.mean(-1))
                if "is_top1" in want:
                    extras["is_top1"].append(logits.argmax(-1) == tgt)  # bool: TRL ANDs it with a mask
        log_probs = torch.cat(logps) if logps else hidden.new_zeros(0) * hidden.sum()
        rest = tuple(torch.cat(extras[k]) if (k in want and extras[k]) else None for k in OUTPUT_ORDER)
        return (log_probs, *rest)


def patch_trl_logprob() -> bool:
    """Install :class:`TorchChunkedLogProb` if TRL's Triton kernel is unavailable. Returns True if patched."""
    try:
        import trl.trainer.utils as tu
    except ImportError:
        return False
    if getattr(tu, "_ChunkedLogProbFunction", object()) is None:
        tu._ChunkedLogProbFunction = TorchChunkedLogProb
        return True
    return False
