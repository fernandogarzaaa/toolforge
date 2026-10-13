"""Command line interface: ``toolforge data | sft | dpo | eval | export | serve``."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import yaml

from toolforge import __version__
from toolforge.config import RunConfig, load_config


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("-c", "--config", help="YAML run config (defaults are used when omitted)")
    p.add_argument(
        "-o",
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value with a dotted key, e.g. -o sft.max_steps=10 (repeatable)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="toolforge", description="Reproducible function-calling fine-tuning pipeline."
    )
    parser.add_argument("--version", action="version", version=f"toolforge {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("data", help="build the cleaned SFT set, eval sets and data card")
    _add_common(p)
    p = sub.add_parser("sft", help="LoRA / QLoRA supervised fine-tuning")
    _add_common(p)
    p = sub.add_parser("dpo", help="mine preference pairs from the SFT model and run DPO")
    _add_common(p)
    p.add_argument("--pairs-only", action="store_true", help="only sample and build pairs, do not train")
    p = sub.add_parser("eval", help="BFCL-style evaluation of the configured models")
    _add_common(p)
    p.add_argument(
        "--only",
        action="append",
        help="evaluate only these model names (repeatable); others are reported from saved predictions",
    )
    p = sub.add_parser("export", help="merge adapters and convert to GGUF")
    _add_common(p)
    p.add_argument("--skip-merge", action="store_true", help="reuse an existing merged checkpoint")
    p = sub.add_parser("serve", help="serve the GGUF with llama.cpp's OpenAI-compatible server")
    _add_common(p)
    p.add_argument("--dry-run", action="store_true", help="print the server command and exit")
    p = sub.add_parser("config", help="print the fully resolved config as JSON")
    _add_common(p)
    return parser


def _print(obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg: RunConfig = load_config(args.config, args.override)
    except (ValueError, OSError, yaml.YAMLError) as exc:
        print(f"toolforge: invalid config: {exc}", file=sys.stderr)
        return 2
    try:
        return _run(args, cfg)
    except ModuleNotFoundError as exc:
        print(
            f"toolforge {args.command}: missing optional dependency {exc.name!r}; "
            "install the extras it needs, e.g. pip install 'toolforge[train]'",
            file=sys.stderr,
        )
        return 2
    except FileNotFoundError as exc:
        print(
            f"toolforge {args.command}: file not found: {exc.filename or exc} "
            "(run the earlier pipeline steps first, e.g. `toolforge data`)",
            file=sys.stderr,
        )
        return 2


def _run(args: argparse.Namespace, cfg: RunConfig) -> int:
    cmd = args.command
    if cmd == "config":
        _print(cfg.model_dump())
    elif cmd == "data":
        from toolforge.data.build import build

        stats = build(cfg.data)
        _print({k: v for k, v in stats.items() if k != "config"})
    elif cmd == "sft":
        from toolforge.train_sft import run_sft

        _print(run_sft(cfg))
    elif cmd == "dpo":
        from toolforge.train_dpo import make_pairs, run_dpo

        if args.pairs_only:
            make_pairs(cfg)
        else:
            _print(run_dpo(cfg))
    elif cmd == "eval":
        from toolforge.evaluate import run_eval

        run_eval(cfg, only=args.only)
    elif cmd == "export":
        from toolforge.export import run_export

        _print(run_export(cfg, skip_merge=args.skip_merge))
    elif cmd == "serve":
        from toolforge.serve import run_server

        return run_server(cfg.serve, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
