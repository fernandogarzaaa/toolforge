# toolforge

**A reproducible fine-tuning pipeline that teaches a small open model reliable function calling:
clean public data → LoRA SFT → DPO on the model's own mistakes → BFCL-style evaluation → GGUF served
through llama.cpp's OpenAI-compatible API.**

[![CI](https://github.com/fernandogarzaaa/toolforge/actions/workflows/ci.yml/badge.svg)](https://github.com/fernandogarzaaa/toolforge/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-Apache--2.0-green)

## The problem

Small models (0.5B–3B) are what you can actually afford to run next to an agent loop, on a laptop,
or on CPU. Their tool calling is the weak spot: they call a function when nothing fits, pick the
wrong one of several, drop required arguments, or emit JSON that does not parse. Fine-tuning helps,
but most recipes stop at "loss went down" and never show whether the model calls tools *better*,
on data it has not seen, with error bars.

toolforge is the whole loop, small enough to run end to end and strict about measurement:

- **Data you can audit**: parsed, schema-validated, deduplicated, decontaminated against the eval
  sets, with a generated [data card](docs/DATA_CARD.md).
- **Training that survives restarts**: every stage resumes from its last checkpoint.
- **An honest scorer**: BFCL-style AST matching of function name and arguments, relevance /
  irrelevance detection, per-category bootstrap CIs and paired deltas against the base model.
- **A served artifact**: merged weights → GGUF → `llama-server` with an OpenAI-compatible endpoint.

## Pipeline

```
 glaive-function-calling-v2 (HF)          BFCL v4 (gorilla, pinned commit)
            │                                        │
            ▼                                        │
  toolforge data ── parse → validate schema → dedup ─┼─► held-out glaive split ─┐
            │        decontaminate (8-gram) ◄────────┴──────────────────────────┤
            ▼                                                                   │
   train.jsonl ──► toolforge sft  (LoRA / QLoRA, PEFT + TRL, resumable)         │
                          │                                                     │
                          ▼                                                     │
   dpo_prompts ──► toolforge dpo  sample SFT model → score with the AST scorer  │
                          │       → (gold, own mistake) pairs → DPO (TRL)       │
                          ▼                                                     ▼
                   toolforge eval  base vs SFT vs SFT+DPO ── BFCL-style AST + relevance, bootstrap CIs
                          │
                          ▼
                   toolforge export  merge adapters → HF checkpoint → GGUF (llama.cpp) → Q8_0
                          │
                          ▼
                   toolforge serve   llama-server --jinja → OpenAI-compatible /v1 (tools / tool_calls)
```

## Quickstart

```bash
git clone https://github.com/fernandogarzaaa/toolforge && cd toolforge
python -m venv .venv && . .venv/bin/activate
pip install -e ".[train,export,dev]"          # CPU torch is fine; add [qlora] on CUDA

toolforge data   -c configs/cpu_small.yaml    # ~2 min: download, clean, split, data card
toolforge sft    -c configs/cpu_small.yaml    # resumable; Ctrl-C and re-run continues
toolforge dpo    -c configs/cpu_small.yaml    # mine pairs from the SFT model, then DPO
toolforge eval   -c configs/cpu_small.yaml    # results/RESULTS.md + results/summary.json
LLAMA_CPP_DIR=~/llama.cpp toolforge export -c configs/cpu_small.yaml
LLAMA_CPP_BIN=~/llama.cpp/build/bin toolforge serve -c configs/cpu_small.yaml
```

Then call it like any OpenAI endpoint:

```bash
curl http://127.0.0.1:8091/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "toolforge",
  "messages": [{"role": "user", "content": "What is the weather in Lisbon?"}],
  "tools": [{"type": "function", "function": {"name": "get_weather",
             "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
}'
```

Any value can be overridden without editing YAML: `toolforge sft -c configs/cpu_small.yaml -o sft.max_steps=20 -o sft.lr=1e-4`.
`toolforge config -c <file>` prints the fully resolved config.

| Config | What it is | Status |
|---|---|---|
| `configs/cpu_small.yaml` | bf16 LoRA on the top 8 layers, small subset, 2 CPU threads | **the measured run below** |
| `configs/gpu_qlora.yaml` | 4-bit NF4 QLoRA, all layers, full cleaned set | untested (no GPU) |
| `configs/smoke.yaml` | tiny random Qwen2, seconds | runs in CI |
| `notebooks/toolforge_gpu_colab.ipynb`, `scripts/gpu_run.sh` | Colab / Kaggle GPU path | untested (no GPU) |
| `deploy/vllm.yaml` | vLLM serving with the hermes tool parser | untested (needs a GPU) |

## Results

<!-- RESULTS -->

## How the evaluation works

Each eval example is a set of tools plus a user turn. The model's raw output is parsed for
`<tool_call>{...}</tool_call>` blocks (a bare JSON call is accepted too), then scored like BFCL's AST
checker:

- **simple / multiple**: exactly one call; the function name must match, every required argument
  must be present, no unexpected arguments, and every value must be one of the acceptable values
  (strings compared case-insensitively ignoring spaces and `,./-_*^`; ints accepted for floats;
  lists element-wise; nested dicts recursively; `""` in the acceptable list means "may be omitted").
- **parallel**: the calls must match the gold calls one-to-one in any order (backtracking matcher).
- **irrelevance**: correct only if the model makes *no* call attempt (a malformed call counts as an attempt).
- **relevance detection** is reported separately: call recall on call categories and no-call
  accuracy on irrelevance categories.

Accuracy per category comes with a 95% percentile bootstrap CI; the macro average is the mean of
categories. Model-vs-base deltas use a paired bootstrap over the same examples.

Two eval sets, both excluded from training by decontamination (exact normalised match or any shared
8-word n-gram):

- **BFCL v4** (out-of-distribution): `simple_python`, `multiple`, `parallel`, `irrelevance`, sampled
  with a fixed seed from the gorilla repo at a pinned commit, scored against BFCL's possible answers.
- **glaive held-out** (in-distribution): `simple`, `multiple`, `irrelevance`, split off before
  training sampling.

## Data

See [docs/DATA_CARD.md](docs/DATA_CARD.md) for every filtering step with counts. In short: 112,960
glaive conversations become decision examples (tools + user turn → the assistant's first decision,
a call or a text reply), schema validation drops malformed tools and calls, dedup on normalised user
text + tool names removes ~55k near-copies, and decontamination removes everything overlapping the
eval sets. `data/processed/stats.json` holds the numbers of the committed run.

## DPO pairs from the model's own mistakes

`toolforge dpo` first samples the SFT model on prompts it was not trained on (greedy + temperature
samples), scores every sample with the same AST scorer, and pairs the gold completion (chosen) with
a wrong sample (rejected). Well-formed-but-wrong calls (wrong value, missing argument, wrong
function) are preferred over format failures because they are the contrast worth learning. Where
the model is already right, a verified synthetic corruption of the gold call can be used instead
(dropped required argument, perturbed value, wrong function, spurious or missing call). Every pair
is checked: chosen must score correct, rejected must score wrong. DPO trains a fresh LoRA adapter on
top of the merged SFT weights, so TRL's adapter-disabled reference *is* the SFT model and no second
copy is held in memory.

## Engineering notes

- **Prompt rendering is owned by the project**, not by `apply_chat_template`: training, pair mining
  and evaluation see byte-identical prompts, and the HF and llama.cpp backends are scored on the
  same bytes. A parity test checks the renderer against the real Qwen2.5 chat template.
- **CPU training tricks** (all optional, all in config): LoRA restricted to the top-k decoder layers
  so backprop stops early; logits computed only from the first supervised token (the 151k-vocab LM
  head is skipped for prompt tokens, exact same loss); over-long examples are dropped rather than
  truncated so no target is cut mid-call.
- **Weights are read without mmap**: memory-mapped safetensors on network / FUSE file systems can
  SIGBUS when pages are evicted, so the loader reads bytes and deserialises.
- **Triton-less TRL**: recent TRL computes DPO log-probs with a Triton kernel; on CPU/aarch64 a
  pure-PyTorch implementation with the same contract is patched in (tested against a reference).
- **Tracking**: MLflow local file store by default (`mlruns/`), W&B offline optional; every run also
  writes `metrics.jsonl` and `train_summary.json` next to its outputs.

## Development

```bash
make lint typecheck test   # ruff, mypy, unit tests (no torch needed)
make smoke                 # tiny random-model end-to-end run: data files → SFT → resume → DPO → eval → merge
make mutation              # testteeth mutation grade of scorer, schema, data cleaning, pairs, formatting
```

<!-- TESTING -->

CI runs lint, mypy, unit tests on Python 3.10–3.12, the tiny-model smoke run and a wheel install
smoke test; mutation testing runs as a separate non-blocking job. Tagging `v*` builds the sdist and
wheel, attaches them to a GitHub Release, and publishes to PyPI only when `PYPI_API_TOKEN` is set.

## Limitations

<!-- LIMITATIONS -->

## License

Apache-2.0. Training data: glaive-function-calling-v2 (Apache-2.0). Evaluation data: BFCL / gorilla
(Apache-2.0). Base model: Qwen2.5-0.5B-Instruct (Apache-2.0).
