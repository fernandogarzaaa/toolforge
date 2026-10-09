from pathlib import Path

import pytest
from pydantic import ValidationError

from toolforge.config import RunConfig, apply_overrides, load_config, parse_override

CONFIGS = sorted((Path(__file__).parent.parent / "configs").glob("*.yaml"))


def test_defaults_load():
    cfg = load_config()
    assert isinstance(cfg, RunConfig)
    assert cfg.model.name_or_path == "Qwen/Qwen2.5-0.5B-Instruct"
    assert cfg.model.quantization == "none"
    assert [m.name for m in cfg.eval.models] == ["base", "sft", "sft+dpo"]
    assert cfg.serve.port != 8090


@pytest.mark.parametrize("path", CONFIGS, ids=[p.name for p in CONFIGS])
def test_shipped_configs_are_valid(path):
    cfg = load_config(path)
    assert cfg.run_name


def test_shipped_configs_exist():
    names = {p.name for p in CONFIGS}
    assert {"cpu_small.yaml", "gpu_qlora.yaml", "smoke.yaml"} <= names


def test_gpu_config_uses_qlora():
    cfg = load_config(Path(__file__).parent.parent / "configs" / "gpu_qlora.yaml")
    assert cfg.model.quantization == "4bit"
    assert cfg.model.dtype in ("bfloat16", "float16")


def test_yaml_file_and_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("run_name: demo\nsft:\n  lr: 0.001\n  lora:\n    r: 4\n")
    cfg = load_config(p, ["sft.max_steps=7", "sft.lora.alpha=8", "model.dtype=bfloat16"])
    assert cfg.run_name == "demo"
    assert cfg.sft.lr == 0.001
    assert cfg.sft.max_steps == 7
    assert cfg.sft.lora.r == 4
    assert cfg.sft.lora.alpha == 8
    assert cfg.model.dtype == "bfloat16"


def test_empty_yaml_gives_defaults(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("")
    assert load_config(p) == RunConfig()


def test_non_mapping_yaml_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("- a\n- b\n")
    with pytest.raises(ValueError, match="top level must be a mapping"):
        load_config(p)


def test_unknown_keys_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("sft:\n  learning_rate: 1\n")
    with pytest.raises(ValidationError):
        load_config(p)


@pytest.mark.parametrize(
    "override",
    [
        "sft.lr=-1",
        "sft.batch_size=0",
        "data.decontam_ngram=0",
        "serve.port=8090",
        "serve.port=70000",
        "model.dtype=int8",
    ],
)
def test_invalid_values_rejected(override):
    with pytest.raises(ValidationError):
        load_config(None, [override])


def test_qlora_requires_half_precision():
    with pytest.raises(ValidationError, match="QLoRA"):
        load_config(None, ["model.quantization=4bit", "model.dtype=float32"])
    cfg = load_config(None, ["model.quantization=4bit", "model.dtype=bfloat16"])
    assert cfg.model.quantization == "4bit"


def test_parse_override_types():
    assert parse_override("a.b=3") == (["a", "b"], 3)
    assert parse_override("a=true") == (["a"], True)
    assert parse_override("a=[1, 2]") == (["a"], [1, 2])
    assert parse_override("a=x=y") == (["a"], "x=y")
    assert parse_override(" a.b =null") == (["a", "b"], None)


@pytest.mark.parametrize("bad", ["novalue", "=3", "..=3"])
def test_parse_override_errors(bad):
    with pytest.raises(ValueError):
        parse_override(bad)


def test_apply_overrides_creates_and_rejects():
    data: dict = {}
    assert apply_overrides(data, ["x.y.z=1"]) == {"x": {"y": {"z": 1}}}
    with pytest.raises(ValueError, match="not a mapping"):
        apply_overrides({"x": 3}, ["x.y=1"])
    assert apply_overrides({"a": 1}, []) == {"a": 1}
