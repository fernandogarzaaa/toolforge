"""`toolforge export`: merge LoRA adapters into the base, convert to GGUF, quantise.

GGUF conversion uses llama.cpp's own ``convert_hf_to_gguf.py`` (point ``export.convert_script`` or
``$LLAMA_CPP_DIR`` at a llama.cpp checkout); quantisation uses the ``llama-quantize`` binary from
``export.llama_cpp_bin_dir`` or ``$LLAMA_CPP_BIN``. Steps whose tools are missing are skipped with a
message, and the merged HF checkpoint is always written.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from toolforge.config import RunConfig


def find_convert_script(explicit: str | None) -> Path | None:
    candidates = [explicit] if explicit else []
    if os.environ.get("LLAMA_CPP_DIR"):
        candidates.append(str(Path(os.environ["LLAMA_CPP_DIR"]) / "convert_hf_to_gguf.py"))
    for c in candidates:
        if c and Path(c).is_file():
            return Path(c)
    return None


def find_binary(name: str, bin_dir: str | None) -> Path | None:
    dirs = [d for d in (bin_dir, os.environ.get("LLAMA_CPP_BIN")) if d]
    for d in dirs:
        p = Path(d) / name
        if p.is_file() and os.access(p, os.X_OK):
            return p
    found = shutil.which(name)
    return Path(found) if found else None


def quant_path(gguf_path: str, qtype: str) -> Path:
    p = Path(gguf_path)
    stem = p.stem
    for suffix in ("-f16", "-bf16", "-f32"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return p.with_name(f"{stem}-{qtype.lower()}.gguf")


def merge(cfg: RunConfig) -> Path:
    import torch

    from toolforge.modeling import apply_adapters, load_model, load_tokenizer

    ec = cfg.export
    model_cfg = cfg.model.model_copy(update={"quantization": "none"})
    model = apply_adapters(load_model(model_cfg), ec.adapters)
    model = model.to(torch.float16 if cfg.model.dtype != "float32" else torch.float32)
    out = Path(ec.merged_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    load_tokenizer(cfg.model.name_or_path, cfg.model.revision).save_pretrained(out)
    (out / "toolforge_export.json").write_text(
        json.dumps({"base": cfg.model.name_or_path, "adapters": ec.adapters}, indent=2)
    )
    return out


def run_export(cfg: RunConfig, skip_merge: bool = False) -> dict[str, Any]:
    ec = cfg.export
    result: dict[str, Any] = {}
    merged = Path(ec.merged_dir) if skip_merge else merge(cfg)
    result["merged_dir"] = str(merged)
    script = find_convert_script(ec.convert_script)
    if script is None:
        print("[toolforge] convert_hf_to_gguf.py not found (set export.convert_script or LLAMA_CPP_DIR); GGUF skipped")
        return result
    gguf = Path(ec.gguf_path)
    gguf.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(script), str(merged), "--outfile", str(gguf), "--outtype", ec.gguf_outtype]
    print("[toolforge] " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    result["gguf"] = str(gguf)
    qbin = find_binary("llama-quantize", ec.llama_cpp_bin_dir)
    result["quantized"] = []
    for qtype in ec.quantize:
        if qbin is None:
            print("[toolforge] llama-quantize not found; quantisation skipped")
            break
        dest = quant_path(str(gguf), qtype)
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = f"{qbin.parent}:{env.get('LD_LIBRARY_PATH', '')}"
        subprocess.run([str(qbin), str(gguf), str(dest), qtype], check=True, env=env)
        result["quantized"].append(str(dest))
    return result
