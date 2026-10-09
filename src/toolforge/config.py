"""YAML run configuration (validated with pydantic).

One YAML file describes a whole run (data -> sft -> dpo -> eval -> export -> serve). Each CLI
subcommand reads its own section. Values can be overridden from the command line with dotted
keys: ``toolforge sft -c configs/cpu_small.yaml -o sft.max_steps=10 -o sft.lr=1e-4``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


BFCLCategory = Literal["simple", "multiple", "parallel", "irrelevance"]
BFCL_CATEGORIES: tuple[BFCLCategory, ...] = ("simple", "multiple", "parallel", "irrelevance")


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(_Base):
    raw_dir: str = "data/raw"
    out_dir: str = "data/processed"
    card_path: str = "docs/DATA_CARD.md"
    glaive_revision: str = "e7f4b6456019f5d8bcb991ef0dd67d8ff23221ac"
    bfcl_commit: str = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
    bfcl_categories: list[BFCLCategory] = Field(default_factory=lambda: list(BFCL_CATEGORIES))
    bfcl_eval_per_category: dict[str, int] = Field(default_factory=dict)
    glaive_eval_per_category: dict[str, int] = Field(
        default_factory=lambda: {"simple": 100, "multiple": 100, "irrelevance": 100}
    )
    train_per_category: dict[str, int] = Field(
        default_factory=lambda: {"simple": 800, "multiple": 800, "irrelevance": 600, "clarify": 200}
    )
    dpo_prompts: int = 400
    decontam_ngram: int = Field(8, ge=1)
    max_chars: int = Field(3000, gt=0)
    seed: int = 1234


class LoraConfigModel(_Base):
    r: int = Field(16, gt=0)
    alpha: int = Field(32, gt=0)
    dropout: float = Field(0.05, ge=0.0, lt=1.0)
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    # Restrict LoRA to these decoder layers (e.g. the top 8). Backprop then stops at the lowest
    # adapted layer, which is the main CPU speed-up. ``null`` adapts every layer.
    layers_to_transform: list[int] | None = None


class ModelConfig(_Base):
    name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct"
    revision: str | None = None
    # Copy the weights to this local directory before loading. Needed on network/FUSE file systems
    # where memory-mapped safetensors can raise SIGBUS. ``null`` loads in place.
    local_copy_dir: str | None = None
    dtype: Literal["float32", "bfloat16", "float16"] = "float32"
    quantization: Literal["none", "4bit"] = "none"  # 4bit = QLoRA (bitsandbytes, CUDA only)
    gradient_checkpointing: bool = True
    num_threads: int | None = None


class TrainConfig(_Base):
    output_dir: str
    max_length: int = Field(512, gt=0)
    num_examples: int | None = None
    epochs: float = Field(1.0, gt=0)
    max_steps: int = -1
    lr: float = Field(2e-4, gt=0)
    batch_size: int = Field(1, gt=0)
    grad_accum: int = Field(8, gt=0)
    warmup_ratio: float = Field(0.03, ge=0, lt=1)
    weight_decay: float = 0.0
    lr_scheduler: str = "cosine"
    save_steps: int = Field(25, gt=0)
    save_total_limit: int = Field(2, gt=0)
    logging_steps: int = Field(5, gt=0)
    seed: int = 42
    resume: bool = True
    # Compute logits only from the first supervised position onwards (skips the 150k-vocab LM head
    # on prompt tokens). Mathematically identical loss; much cheaper on CPU.
    completion_logits_only: bool = True
    lora: LoraConfigModel = Field(default_factory=LoraConfigModel)


class SFTConfig(TrainConfig):
    output_dir: str = "outputs/sft"
    train_file: str = "data/processed/train.jsonl"


class DPOConfig(TrainConfig):
    output_dir: str = "outputs/dpo"
    sft_adapter: str = "outputs/sft/adapter"
    prompts_file: str = "data/processed/dpo_prompts.jsonl"
    pairs_file: str = "outputs/dpo/pairs.jsonl"
    generations_file: str = "outputs/dpo/generations.jsonl"
    num_prompts: int | None = None
    samples_per_prompt: int = Field(2, gt=0)
    temperature: float = Field(0.8, gt=0)
    max_new_tokens: int = Field(96, gt=0)
    synthetic_negatives: bool = True
    beta: float = Field(0.1, gt=0)
    lr: float = Field(5e-5, gt=0)
    max_prompt_length: int = Field(448, gt=0)


class EvalModelSpec(_Base):
    name: str
    backend: Literal["hf", "openai"] = "hf"
    adapters: list[str] = Field(default_factory=list)  # applied (merged) in order on top of the base
    base: str | None = None  # defaults to model.name_or_path
    endpoint: str | None = None  # for backend=openai, e.g. http://127.0.0.1:8091/v1
    model_id: str | None = None  # model name sent to the openai endpoint


class EvalConfig(_Base):
    files: list[str] = Field(
        default_factory=lambda: ["data/processed/eval_bfcl.jsonl", "data/processed/eval_glaive.jsonl"]
    )
    models: list[EvalModelSpec] = Field(
        default_factory=lambda: [
            EvalModelSpec(name="base"),
            EvalModelSpec(name="sft", adapters=["outputs/sft/adapter"]),
            EvalModelSpec(name="sft+dpo", adapters=["outputs/sft/adapter", "outputs/dpo/adapter"]),
        ]
    )
    limit_per_category: int | None = None
    max_new_tokens: int = Field(160, gt=0)
    batch_size: int = Field(4, gt=0)
    n_bootstrap: int = Field(1000, gt=0)
    seed: int = 0
    out_dir: str = "results"


class ExportConfig(_Base):
    adapters: list[str] = Field(default_factory=lambda: ["outputs/sft/adapter", "outputs/dpo/adapter"])
    merged_dir: str = "outputs/merged"
    gguf_path: str = "outputs/gguf/toolforge-qwen2.5-0.5b-f16.gguf"
    gguf_outtype: str = "f16"
    quantize: list[str] = Field(default_factory=lambda: ["Q8_0"])
    convert_script: str | None = None  # path to llama.cpp convert_hf_to_gguf.py (or $LLAMA_CPP_DIR)
    llama_cpp_bin_dir: str | None = None  # dir with llama-quantize / llama-server (or $LLAMA_CPP_BIN)


class ServeConfig(_Base):
    gguf_path: str = "outputs/gguf/toolforge-qwen2.5-0.5b-q8_0.gguf"
    llama_cpp_bin_dir: str | None = None
    host: str = "127.0.0.1"
    port: int = Field(8091, gt=0, lt=65536)
    ctx_size: int = Field(4096, gt=0)
    threads: int | None = None
    alias: str = "toolforge"
    extra_args: list[str] = Field(default_factory=list)

    @field_validator("port")
    @classmethod
    def _not_reserved(cls, v: int) -> int:
        if v == 8090:
            raise ValueError("port 8090 is reserved on the build box; pick another")
        return v


class TrackingConfig(_Base):
    backend: Literal["mlflow", "wandb", "none"] = "mlflow"
    mlflow_uri: str = "file:./mlruns"
    experiment: str = "toolforge"
    wandb_project: str = "toolforge"
    wandb_mode: Literal["offline", "online"] = "offline"


class RunConfig(_Base):
    run_name: str = "toolforge"
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    sft: SFTConfig = Field(default_factory=SFTConfig)
    dpo: DPOConfig = Field(default_factory=DPOConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    serve: ServeConfig = Field(default_factory=ServeConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)

    @model_validator(mode="after")
    def _qlora_needs_half(self) -> RunConfig:
        if self.model.quantization == "4bit" and self.model.dtype == "float32":
            raise ValueError("QLoRA (quantization=4bit) needs model.dtype bfloat16 or float16")
        return self


def parse_override(item: str) -> tuple[list[str], Any]:
    """``"sft.lr=1e-4"`` -> (["sft", "lr"], 1e-4). The value is parsed as YAML."""
    if "=" not in item:
        raise ValueError(f"override must look like key.path=value, got {item!r}")
    key, raw = item.split("=", 1)
    parts = [p for p in key.strip().split(".") if p]
    if not parts:
        raise ValueError(f"empty override key in {item!r}")
    return parts, yaml.safe_load(raw)


def apply_overrides(data: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    for item in overrides:
        parts, value = parse_override(item)
        node = data
        for p in parts[:-1]:
            child = node.setdefault(p, {})
            if not isinstance(child, dict):
                raise ValueError(f"cannot set {item!r}: {p!r} is not a mapping")
            node = child
        node[parts[-1]] = value
    return data


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> RunConfig:
    """Load a YAML run config (or defaults when ``path`` is None) and apply overrides."""
    data: dict[str, Any] = {}
    if path is not None:
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{path}: top level must be a mapping")
        data = loaded
    apply_overrides(data, overrides or [])
    return RunConfig.model_validate(data)
