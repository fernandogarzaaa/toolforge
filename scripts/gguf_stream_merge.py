"""Low-memory GGUF rewrite: F16 GGUF -> Q8_0 (or F16), optionally merging a PEFT LoRA adapter on the way.

Why this exists: on the 7 GB / no-swap shared box used for the measured run, `toolforge export`'s
torch merge of two stacked adapters and `llama-quantize` (which mmaps its input) were OOM-killed or
SIGBUSed (mmapped pages on the JuiceFS workspace mount) repeatedly. This script needs ~150 MB:

- the file is read with plain ``pread`` calls (no mmap), retried on EIO;
- KV metadata is copied byte for byte (``general.file_type`` patched to MOSTLY_Q8_0 for q8_0 output);
- every 2-D F16 tensor is processed in 16 MB row chunks: optionally ``W += (alpha / r) * B @ A`` in
  f32 for the LoRA target matrices, then rounded to Q8_0 (gguf-py's reference quantizer) or F16;
- 1-D tensors (norms, biases) are copied unchanged, as llama-quantize does for Q8_0.

Only plain LoRA is supported (no rslora / DoRA / fan_in_fan_out), with Qwen2 / Llama-style module
names (q/k/v/o_proj, gate/up/down_proj) and an architecture whose converter does not permute q/k
(true for Qwen2). Checked on the measured run: base F16 GGUF + SFT adapter through this script matches
the GGUF produced by `toolforge export` (torch merge -> convert_hf_to_gguf) to within 1 f16 ULP on
8,375 of ~494M weights and bit-identical elsewhere.

usage: python scripts/gguf_stream_merge.py IN.gguf OUT.gguf [--lora ADAPTER_DIR] [--outtype q8_0|f16]
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import time
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType
from gguf.quants import quantize

F32, F16, Q8_0 = 0, 1, 8
MOSTLY_Q8_0 = 7
SCALAR_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
STRING, ARRAY = 8, 9
CHUNK = 16 << 20
HF2GGUF = {
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
}


class Reader:
    """Sequential GGUF header reader on top of retried ``pread`` (no mmap)."""

    def __init__(self, path: str) -> None:
        self.fd = os.open(path, os.O_RDONLY)
        self.pos = 0

    def pread(self, n: int, off: int) -> bytes:
        out = bytearray()
        while len(out) < n:
            for attempt in range(30):
                try:
                    chunk = os.pread(self.fd, min(n - len(out), 8 << 20), off + len(out))
                    break
                except OSError:
                    time.sleep(2 + attempt)
            else:
                raise OSError(f"read failed at offset {off + len(out)}")
            if not chunk:
                raise EOFError(f"unexpected end of file at offset {off + len(out)}")
            out += chunk
        return bytes(out)

    def take(self, n: int) -> bytes:
        b = self.pread(n, self.pos)
        self.pos += n
        return b

    def u32(self) -> int:
        return int(struct.unpack("<I", self.take(4))[0])

    def u64(self) -> int:
        return int(struct.unpack("<Q", self.take(8))[0])

    def string(self) -> bytes:
        return self.take(self.u64())

    def skip_value(self, vtype: int) -> None:
        if vtype in SCALAR_SIZES:
            self.pos += SCALAR_SIZES[vtype]
        elif vtype == STRING:
            n = self.u64()
            self.pos += n
        elif vtype == ARRAY:
            itype, n = self.u32(), self.u64()
            if itype in SCALAR_SIZES:
                self.pos += SCALAR_SIZES[itype] * n
            else:
                for _ in range(n):
                    self.skip_value(itype)
        else:
            raise ValueError(f"unknown GGUF value type {vtype}")


def load_lora(adapter_dir: str) -> dict[bytes, tuple[np.ndarray, np.ndarray, float]]:
    from safetensors.numpy import load as st_load

    cfg = json.loads((Path(adapter_dir) / "adapter_config.json").read_text())
    if cfg.get("use_rslora") or cfg.get("use_dora") or cfg.get("fan_in_fan_out"):
        raise SystemExit("only plain LoRA adapters are supported")
    scale = cfg["lora_alpha"] / cfg["r"]
    sd = st_load((Path(adapter_dir) / "adapter_model.safetensors").read_bytes())
    deltas = {}
    for key in sd:
        if not key.endswith(".lora_A.weight"):
            continue
        mod = key[: -len(".lora_A.weight")]
        layer, rest = mod.split("model.layers.")[1].split(".", 1)
        name = f"blk.{layer}.{HF2GGUF[rest]}.weight".encode()
        deltas[name] = (sd[key].astype(np.float32), sd[mod + ".lora_B.weight"].astype(np.float32), scale)
    return deltas


def nbytes(ne: list[int], ttype: int) -> int:
    n = int(np.prod(ne))
    return {F32: 4 * n, F16: 2 * n, Q8_0: n // 32 * 34}[ttype]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--lora", help="PEFT adapter dir to merge into the weights")
    ap.add_argument("--outtype", default="q8_0", choices=["q8_0", "f16"])
    args = ap.parse_args()
    deltas = load_lora(args.lora) if args.lora else {}
    if deltas:
        print(f"lora: {len(deltas)} target matrices", flush=True)

    r = Reader(args.src)
    if r.take(4) != b"GGUF":
        raise SystemExit("not a GGUF file")
    version = r.u32()
    n_tensors, n_kv = r.u64(), r.u64()
    kv_blobs: list[bytes] = []
    alignment, have_qv = 32, False
    for _ in range(n_kv):
        start = r.pos
        key = r.string()
        vtype = r.u32()
        vstart = r.pos
        r.skip_value(vtype)
        blob = r.pread(r.pos - start, start)
        if key == b"general.file_type" and args.outtype == "q8_0":
            blob = blob[: vstart - start] + struct.pack("<I", MOSTLY_Q8_0)
        if key == b"general.alignment":
            alignment = int(struct.unpack("<I", r.pread(4, vstart))[0])
        have_qv |= key == b"general.quantization_version"
        kv_blobs.append(blob)
    if args.outtype == "q8_0" and not have_qv:
        k = b"general.quantization_version"
        kv_blobs.append(struct.pack("<Q", len(k)) + k + struct.pack("<II", 4, 2))

    def pad(n: int) -> int:
        return (n + alignment - 1) // alignment * alignment

    infos = []
    for _ in range(n_tensors):
        name = r.string()
        ne = [r.u64() for _ in range(r.u32())]
        ttype, off = r.u32(), r.u64()
        infos.append((name, ne, ttype, off))
    data_start = pad(r.pos)

    plan, off = [], 0
    for name, ne, ttype, src_off in infos:
        two_d = len(ne) == 2 and ttype == F16
        if name in deltas and not two_d:
            raise SystemExit(f"LoRA target {name.decode()} is not a 2-D F16 tensor")
        new_type = Q8_0 if two_d and args.outtype == "q8_0" and ne[0] % 32 == 0 else ttype
        plan.append((name, ne, ttype, src_off, new_type, off))
        off = pad(off + nbytes(ne, new_type))
    unused = set(deltas) - {p[0] for p in plan}
    if unused:
        raise SystemExit(f"LoRA targets not found in the GGUF: {sorted(n.decode() for n in unused)[:3]}")

    with open(args.dst, "wb") as out:
        out.write(b"GGUF" + struct.pack("<IQQ", version, n_tensors, len(kv_blobs)))
        for blob in kv_blobs:
            out.write(blob)
        for name, ne, _t, _so, new_type, o in plan:
            out.write(struct.pack("<Q", len(name)) + name + struct.pack("<I", len(ne)))
            out.write(struct.pack(f"<{len(ne)}Q", *ne) + struct.pack("<IQ", new_type, o))
        out.write(b"\0" * (pad(out.tell()) - out.tell()))
        base = out.tell()
        for i, (name, ne, ttype, so, new_type, o) in enumerate(plan):
            out.write(b"\0" * (base + o - out.tell()))
            if new_type == ttype and name not in deltas:
                n = nbytes(ne, ttype)
                for c in range(0, n, CHUNK):
                    out.write(r.pread(min(CHUNK, n - c), data_start + so + c))
                kind = "copy"
            else:
                cols, rows = ne
                step = max(1, CHUNK // (cols * 2))
                for r0 in range(0, rows, step):
                    r1 = min(rows, r0 + step)
                    raw = r.pread((r1 - r0) * cols * 2, data_start + so + r0 * cols * 2)
                    w = np.frombuffer(raw, dtype=np.float16).reshape(r1 - r0, cols).astype(np.float32)
                    if name in deltas:
                        a, b, scale = deltas[name]
                        w += scale * (b[r0:r1] @ a)
                    q = quantize(w, GGMLQuantizationType.Q8_0) if new_type == Q8_0 else w.astype(np.float16)
                    out.write(q.tobytes())
                kind = ("q8_0" if new_type == Q8_0 else "f16") + (" +lora" if name in deltas else "")
            print(f"[{i + 1}/{len(plan)}] {name.decode()} {kind}", flush=True)
        out.write(b"\0" * (pad(out.tell()) - out.tell()))
    print(f"wrote {args.dst}")


if __name__ == "__main__":
    main()
