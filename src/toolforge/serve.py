"""`toolforge serve`: run llama.cpp's ``llama-server`` (OpenAI-compatible) on the exported GGUF."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from toolforge.config import ServeConfig
from toolforge.export import find_binary


def server_command(sc: ServeConfig, binary: str | Path) -> list[str]:
    cmd = [
        str(binary),
        "-m",
        sc.gguf_path,
        "--host",
        sc.host,
        "--port",
        str(sc.port),
        "-c",
        str(sc.ctx_size),
        "--alias",
        sc.alias,
        "--jinja",  # use the GGUF's chat template: enables OpenAI `tools` / `tool_calls`
    ]
    if sc.threads:
        cmd += ["-t", str(sc.threads)]
    return cmd + list(sc.extra_args)


def run_server(sc: ServeConfig, dry_run: bool = False) -> int:
    binary = find_binary("llama-server", sc.llama_cpp_bin_dir)
    if binary is None:
        raise FileNotFoundError("llama-server not found: set serve.llama_cpp_bin_dir or LLAMA_CPP_BIN")
    if not Path(sc.gguf_path).exists() and not dry_run:
        raise FileNotFoundError(f"GGUF not found: {sc.gguf_path} (run `toolforge export` first)")
    cmd = server_command(sc, binary)
    print("[toolforge] " + " ".join(cmd))
    print(f"[toolforge] OpenAI-compatible endpoint: http://{sc.host}:{sc.port}/v1")
    if dry_run:
        return 0
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{binary.parent}:{env.get('LD_LIBRARY_PATH', '')}"
    return subprocess.call(cmd, env=env)
