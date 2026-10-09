"""End-to-end smoke run on a tiny random Qwen2 model: data files -> SFT -> DPO -> eval -> merge.

Needs the ``[train]`` extra. Runs in well under a minute on a laptop CPU; numbers are meaningless,
the point is that every stage runs, checkpoints, resumes, and produces its artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("trl")
pytest.importorskip("datasets")

pytestmark = pytest.mark.smoke

TOOL = {
    "name": "get_weather",
    "description": "Get the weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}


def _records(n: int, prefix: str) -> list[dict]:
    out = []
    cities = ["Paris", "Rome", "Oslo", "Lima"]
    for i in range(n):
        city = cities[i % len(cities)]
        if i % 3 == 2:
            out.append(
                {
                    "id": f"{prefix}{i}",
                    "source": "glaive-function-calling-v2",
                    "category": "irrelevance",
                    "tools": [TOOL],
                    "messages": [{"role": "user", "content": f"Tell me a joke number {i}"}],
                    "target": {"type": "text", "content": "I can only check the weather."},
                    "gold": [],
                }
            )
        else:
            call = {"name": "get_weather", "arguments": {"city": city}}
            out.append(
                {
                    "id": f"{prefix}{i}",
                    "source": "glaive-function-calling-v2",
                    "category": "simple",
                    "tools": [TOOL],
                    "messages": [{"role": "user", "content": f"Weather in {city}? ({i})"}],
                    "target": {"type": "call", "calls": [call]},
                    "gold": [{"get_weather": {"city": [city]}}],
                }
            )
    return out


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _make_tiny_model(model_dir: Path, corpus: list[str]) -> None:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=400, special_tokens=specials, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    tok.train_from_iterator(corpus, trainer)
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>", additional_special_tokens=specials[1:2]
    )
    hf_tok.save_pretrained(model_dir)
    cfg = Qwen2Config(
        vocab_size=len(hf_tok),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=1024,
        tie_word_embeddings=True,
        eos_token_id=hf_tok.eos_token_id,
        pad_token_id=hf_tok.pad_token_id,
    )
    import torch

    torch.manual_seed(0)
    Qwen2ForCausalLM(cfg).save_pretrained(model_dir, safe_serialization=True)


@pytest.fixture(scope="module")
def smoke_cfg(tmp_path_factory):
    from toolforge.config import load_config
    from toolforge.formatting import render_messages, render_target

    root = tmp_path_factory.mktemp("smoke")
    train = _records(12, "t")
    dpo = _records(4, "d")
    ev = _records(6, "e")
    data = root / "data"
    _write_jsonl(data / "train.jsonl", train)
    _write_jsonl(data / "dpo_prompts.jsonl", dpo)
    _write_jsonl(data / "eval.jsonl", ev)
    corpus = [render_messages(r["messages"], r["tools"], True) + render_target(r["target"]) for r in train] * 3
    _make_tiny_model(root / "model", corpus)
    out = root / "out"
    cfg_path = Path(__file__).parents[2] / "configs" / "smoke.yaml"
    overrides = [
        f"model.name_or_path={root / 'model'}",
        f"sft.output_dir={out / 'sft'}",
        f"sft.train_file={data / 'train.jsonl'}",
        f"dpo.output_dir={out / 'dpo'}",
        f"dpo.sft_adapter={out / 'sft' / 'adapter'}",
        f"dpo.prompts_file={data / 'dpo_prompts.jsonl'}",
        f"dpo.pairs_file={out / 'dpo' / 'pairs.jsonl'}",
        f"dpo.generations_file={out / 'dpo' / 'generations.jsonl'}",
        f"eval.files=[{data / 'eval.jsonl'}]",
        f"eval.out_dir={out / 'results'}",
        f"eval.models=[{{name: base}}, {{name: sft, adapters: [{out / 'sft' / 'adapter'}]}}, "
        f"{{name: sft+dpo, adapters: [{out / 'sft' / 'adapter'}, {out / 'dpo' / 'adapter'}]}}]",
        f"export.adapters=[{out / 'sft' / 'adapter'}, {out / 'dpo' / 'adapter'}]",
        f"export.merged_dir={out / 'merged'}",
        "export.convert_script=null",
    ]
    return load_config(cfg_path, overrides), out


def test_completion_loss_matches_full_loss(smoke_cfg):
    import torch

    from toolforge.modeling import load_model
    from toolforge.train_sft import completion_loss

    cfg, _ = smoke_cfg
    model = load_model(cfg.model)
    ids = torch.randint(0, 50, (2, 12))
    labels = ids.clone()
    labels[0, :7] = -100
    labels[1, :9] = -100
    full = model(input_ids=ids, labels=labels).loss
    ours = completion_loss(model, {"input_ids": ids, "labels": labels, "attention_mask": torch.ones_like(ids)})
    assert torch.allclose(full, ours, atol=1e-5)
    # with an explicit token count it is a plain sum / count
    n = int((labels[:, 1:] != -100).sum())
    ours_n = completion_loss(model, {"input_ids": ids, "labels": labels}, num_items_in_batch=n)
    assert torch.allclose(full, ours_n, atol=1e-5)


def test_tokenize_pairs_masks_prompt_and_drops_long(smoke_cfg):
    from toolforge.modeling import load_tokenizer
    from toolforge.train_sft import build_text_pairs, tokenize_pairs

    cfg, _ = smoke_cfg
    tok = load_tokenizer(cfg.model.name_or_path)
    pairs = build_text_pairs(_records(2, "x"))
    assert pairs[0]["completion"].endswith("<|im_end|>")
    rows, dropped = tokenize_pairs(tok, pairs, max_length=10_000)
    assert dropped == 0
    row = rows[0]
    n_prompt = len(tok(pairs[0]["prompt"], add_special_tokens=False)["input_ids"])
    assert row["completion_mask"][:n_prompt] == [0] * n_prompt
    assert all(m == 1 for m in row["completion_mask"][n_prompt:])
    assert row["input_ids"][-1] == tok.convert_tokens_to_ids("<|im_end|>")
    assert row["input_ids"].count(tok.convert_tokens_to_ids("<|im_end|>")) == 3  # system, user, completion
    _, dropped = tokenize_pairs(tok, pairs, max_length=5)
    assert dropped == 2


def test_pipeline_end_to_end(smoke_cfg):
    from toolforge.evaluate import run_eval
    from toolforge.export import run_export
    from toolforge.train_dpo import run_dpo
    from toolforge.train_sft import run_sft

    cfg, out = smoke_cfg
    sft = run_sft(cfg)
    assert sft["global_steps"] == 2
    assert (out / "sft" / "adapter" / "adapter_config.json").exists()
    assert (out / "sft" / "checkpoint-2").exists()

    # resume: asking for one more step continues from checkpoint-2 instead of restarting
    resumed = run_sft(cfg.model_copy(update={"sft": cfg.sft.model_copy(update={"max_steps": 3})}))
    assert resumed["resumed_from"].endswith("checkpoint-2")
    assert resumed["global_steps"] == 3
    summary = json.loads((out / "sft" / "train_summary.json").read_text())
    assert len(summary["sessions"]) == 2

    dpo = run_dpo(cfg)
    assert dpo["global_steps"] == 2
    assert (out / "dpo" / "adapter" / "adapter_config.json").exists()
    pairs = [json.loads(line) for line in (out / "dpo" / "pairs.jsonl").read_text().splitlines()]
    assert pairs and all(p["chosen"] != p["rejected"] for p in pairs)
    gens = (out / "dpo" / "generations.jsonl").read_text().splitlines()
    assert len(gens) == 4 and all(len(json.loads(g)["completions"]) == 2 for g in gens)

    report = run_eval(cfg)
    assert set(report["models"]) == {"base", "sft", "sft+dpo"}
    assert report["n_examples"] == 6
    assert (out / "results" / "RESULTS.md").exists()
    # predictions are resumable: a second run regenerates nothing and gives the same report
    again = run_eval(cfg)
    assert again["models"]["sft"]["micro_accuracy"] == report["models"]["sft"]["micro_accuracy"]

    exported = run_export(cfg)
    assert (Path(exported["merged_dir"]) / "config.json").exists()
    assert "gguf" not in exported


def test_torch_logprob_fallback_matches_reference():
    import torch

    from toolforge.compat import TorchChunkedLogProb

    torch.manual_seed(0)
    hidden = torch.randn(5, 8, requires_grad=True)
    weight = torch.randn(11, 8)
    targets = torch.tensor([0, 3, 10, 2, 2])
    logp, ent, lss, mean, top1 = TorchChunkedLogProb.apply(
        hidden, weight, None, targets, 2.0, 2, None, 1.0, ("entropy",)
    )
    ref = torch.log_softmax((hidden @ weight.t()) / 2.0, -1)
    assert torch.allclose(logp, ref.gather(-1, targets[:, None]).squeeze(-1), atol=1e-6)
    assert torch.allclose(ent, -(ref.exp() * ref).sum(-1), atol=1e-5)
    assert lss is None and mean is None and top1 is None
    logp.sum().backward()
    assert hidden.grad is not None
