# toolforge developer & pipeline targets. CONFIG selects the run profile.
CONFIG ?= configs/cpu_small.yaml
PY ?= python
TF ?= toolforge

.PHONY: help install install-dev lint typecheck test smoke mutation build pipeline data sft dpo eval export serve clean

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

install: ## install with training + export extras
	$(PY) -m pip install -e ".[train,export]"

install-dev: ## install everything needed for development and CI
	$(PY) -m pip install -e ".[train,export,dev]"

lint: ## ruff
	ruff check src tests
	ruff format --check src tests

typecheck: ## mypy
	mypy

test: ## fast unit tests (no torch needed)
	$(PY) -m pytest -m "not smoke and not parity"

smoke: ## tiny random-model end-to-end run (needs [train])
	$(PY) -m pytest -m smoke tests/smoke

mutation: ## testteeth mutation grade of the core modules (scorer, schema, data cleaning, pairs, formatting)
	testteeth run --workers 2 --json mutation.json

build: ## sdist + wheel
	$(PY) -m build

# ---- pipeline -------------------------------------------------------------------
data: ## download + clean + split + decontaminate + data card
	$(TF) data -c $(CONFIG)
sft: ## LoRA / QLoRA SFT (resumes from checkpoints)
	$(TF) sft -c $(CONFIG)
dpo: ## mine pairs from the SFT model, then DPO (resumes)
	$(TF) dpo -c $(CONFIG)
eval: ## base vs SFT vs SFT+DPO, BFCL-style (resumes)
	$(TF) eval -c $(CONFIG)
export: ## merge + GGUF (+ quantise)
	$(TF) export -c $(CONFIG)
serve: ## llama.cpp OpenAI-compatible server
	$(TF) serve -c $(CONFIG)

pipeline: data sft dpo eval export ## the whole thing

clean:
	rm -rf build dist .pytest_cache .mypy_cache .ruff_cache outputs-smoke mutation.json
