# Changelog

All notable changes are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-10

### Added
- `toolforge data`: glaive-function-calling-v2 parsing into decision examples, tool-schema and call
  validation, normalised dedup, held-out glaive split, n-gram decontamination against the eval sets
  (glaive held-out + BFCL v4), stratified subsampling and a generated data card.
- `toolforge sft`: LoRA SFT with PEFT + TRL (prompt-masked loss, completion-only logits, optional
  top-k-layer LoRA for CPU), QLoRA 4-bit path for CUDA (untested), checkpoint resume.
- `toolforge dpo`: pair mining from the SFT model's own mistakes with verified synthetic fallbacks,
  DPO via TRL with the adapter-disabled SFT model as reference, resumable sampling and training,
  pure-PyTorch log-prob fallback when Triton is unavailable.
- `toolforge eval`: BFCL-style AST scorer (simple / multiple / parallel / irrelevance), relevance
  detection, per-category bootstrap CIs and paired bootstrap deltas; HF and OpenAI-compatible backends.
- `toolforge export` / `serve`: adapter merge, GGUF conversion via llama.cpp, quantisation,
  llama-server launcher; Dockerfile and an untested vLLM config.
- YAML configs (CPU run, GPU QLoRA, smoke), MLflow / W&B-offline tracking, Colab notebook, CI (lint,
  mypy, tests on 3.10-3.12, tiny-model smoke run, wheel smoke test), non-blocking mutation-testing
  job, release workflow.
