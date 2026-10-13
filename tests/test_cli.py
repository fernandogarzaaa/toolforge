import json
import os
import stat

import pytest

from toolforge import __version__
from toolforge.cli import build_parser, main
from toolforge.config import ServeConfig
from toolforge.export import find_binary, find_convert_script, quant_path
from toolforge.serve import run_server, server_command


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_subcommands_exist():
    parser = build_parser()
    for cmd in ["data", "sft", "dpo", "eval", "export", "serve", "config"]:
        assert parser.parse_args([cmd]).command == cmd


def test_config_command_prints_resolved_config(capsys):
    assert main(["config", "-o", "sft.lr=0.5", "-o", "run_name=x"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["sft"]["lr"] == 0.5
    assert out["run_name"] == "x"


def test_invalid_config_exit_code(capsys, tmp_path):
    assert main(["config", "-o", "sft.lr=-1"]) == 2
    assert "invalid config" in capsys.readouterr().err
    assert main(["config", "-c", str(tmp_path / "missing.yaml")]) == 2


def _fake_bin(tmp_path, name):
    p = tmp_path / name
    p.write_text("#!/bin/sh\necho fake\n")
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return p


def test_server_command():
    sc = ServeConfig(gguf_path="m.gguf", port=8097, threads=2, extra_args=["--flag"])
    cmd = server_command(sc, "/x/llama-server")
    assert cmd[:3] == ["/x/llama-server", "-m", "m.gguf"]
    assert cmd[cmd.index("--port") + 1] == "8097"
    assert cmd[cmd.index("-t") + 1] == "2"
    assert "--jinja" in cmd
    assert cmd[-1] == "--flag"
    assert "-t" not in server_command(ServeConfig(gguf_path="m.gguf"), "b")


def test_serve_dry_run(tmp_path, capsys):
    _fake_bin(tmp_path, "llama-server")
    sc = ServeConfig(gguf_path=str(tmp_path / "none.gguf"), llama_cpp_bin_dir=str(tmp_path), port=8097)
    assert run_server(sc, dry_run=True) == 0
    assert "http://127.0.0.1:8097/v1" in capsys.readouterr().out


def test_serve_missing_gguf_or_binary(tmp_path, monkeypatch):
    monkeypatch.delenv("LLAMA_CPP_BIN", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="llama-server not found"):
        run_server(ServeConfig(gguf_path="x.gguf", llama_cpp_bin_dir=str(tmp_path / "nope")))
    _fake_bin(tmp_path, "llama-server")
    with pytest.raises(FileNotFoundError, match="GGUF not found"):
        run_server(ServeConfig(gguf_path=str(tmp_path / "x.gguf"), llama_cpp_bin_dir=str(tmp_path)))


def test_find_binary_order(tmp_path, monkeypatch):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    _fake_bin(b, "tool")
    monkeypatch.setenv("LLAMA_CPP_BIN", str(b))
    monkeypatch.setenv("PATH", "")
    assert find_binary("tool", str(a)) == b / "tool"
    _fake_bin(a, "tool")
    assert find_binary("tool", str(a)) == a / "tool"
    assert find_binary("other", str(a)) is None


def test_find_convert_script(tmp_path, monkeypatch):
    monkeypatch.delenv("LLAMA_CPP_DIR", raising=False)
    assert find_convert_script(None) is None
    script = tmp_path / "convert_hf_to_gguf.py"
    script.write_text("")
    assert find_convert_script(str(script)) == script
    monkeypatch.setenv("LLAMA_CPP_DIR", str(tmp_path))
    assert find_convert_script(None) == script
    assert find_convert_script(str(tmp_path / "missing.py")) == script


def test_quant_path():
    assert quant_path("out/m-f16.gguf", "Q8_0") == quant_path("out/m.gguf", "Q8_0")
    assert str(quant_path("out/m-f16.gguf", "Q4_K_M")) == os.path.join("out", "m-q4_k_m.gguf")
    assert str(quant_path("m-bf16.gguf", "Q8_0")) == "m-q8_0.gguf"


def test_malformed_yaml_config_exit_code(capsys, tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("x: [\n")
    assert main(["config", "-c", str(bad)]) == 2
    assert "invalid config" in capsys.readouterr().err


def test_missing_optional_dependency_is_reported(capsys, monkeypatch):
    import toolforge.cli as cli

    def boom(args, cfg):
        raise ModuleNotFoundError("No module named 'datasets'", name="datasets")

    monkeypatch.setattr(cli, "_run", boom)
    assert main(["sft"]) == 2
    err = capsys.readouterr().err
    assert "'datasets'" in err and "toolforge[train]" in err


def test_missing_input_file_is_reported(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["eval", "-o", "run_name=x"]) == 2
    assert "file not found" in capsys.readouterr().err
