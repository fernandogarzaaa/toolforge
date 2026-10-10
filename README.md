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

**Measured result of the small CPU run (below): it did not beat the base model.** SFT and SFT+DPO
both score 7 points under base on 200 BFCL items (paired 95% CIs exclude zero). The run is tiny on
purpose (398 SFT examples, 64 DPO pairs, CPU only); the [model card](docs/MODEL_CARD.md) and
[results/RESULTS.md](results/RESULTS.md) have the details.

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

**The CPU run takes hours, not minutes.** On the shared 2-vCPU ARM box used for the numbers below,
`sft` + `dpo` with `configs/cpu_small.yaml` took about two hours of wall time (summed over resumed
attempts). An `eval` of 200 BFCL items through llama.cpp took 4 minutes for the base model on a quiet
box and well over an hour for SFT+DPO when the box was busy and `llama-server` kept being OOM-killed
(predictions are appended per item, so a re-run resumes). To
check that everything is wired up, run the tiny-model smoke test instead (`make smoke`, about a minute).

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
| `configs/cpu_small.yaml` | fp32 LoRA on the top 8 layers, 400-example subset, 2 CPU threads | **the measured run below** |
| `configs/gpu_qlora.yaml` | 4-bit NF4 QLoRA, all layers, full cleaned set | untested (no GPU) |
| `configs/smoke.yaml` | tiny random Qwen2, seconds | runs in CI |
| `notebooks/toolforge_gpu_colab.ipynb`, `scripts/gpu_run.sh` | Colab / Kaggle GPU path | untested (no GPU) |
| `deploy/vllm.yaml` | vLLM serving with the hermes tool parser | untested (needs a GPU) |

## Results

<!-- RESULTS -->
Measured on a shared 2-vCPU aarch64 box (no GPU), 2026-10-10. Everything below is from
`results/summary.json`; per-example predictions are in `results/predictions/` and scored rows in
`results/scored/`.

**What was trained.** Qwen2.5-0.5B-Instruct (pinned revision), plain LoRA (r=16, alpha=32) on the
top 8 of 24 decoder layers, **fp32 on CPU** (not QLoRA, not bf16: see `configs/cpu_small.yaml`).

| Stage | Data | Optimizer steps | Wall time |
|---|---|---|---|
| SFT | 398 examples (first 400 of the 1,349-example train set, 2 dropped as > 512 tokens), 1 epoch, effective batch 8 | 50 | 1 h 40 min (summed over 6 attempts) |
| DPO | 64 verified pairs from 64 held-out prompts (9 from the SFT model's own mistakes, 55 verified synthetic corruptions), beta 0.1, 1 epoch, effective batch 8 | 8 | 22 min 11 s (summed over 2 attempts, incl. pair mining) |

Wall times are summed over resumed attempts (the sandbox restarted / OOM-killed training
6 times; every stage resumed from its last checkpoint). Final train loss: SFT
0.119, DPO 0.617.

**Evaluation.** BFCL v4 (out-of-distribution), 200 examples (50 per
category, the first 50 of each category of the committed eval set), greedy decoding,
max 160 new tokens, served through llama.cpp b11521 (`llama-server`, raw `/v1/completions` with
toolforge-rendered prompts, `-c 2048 -np 1`). **Precision per model:** base = f16 GGUF (200/200); SFT = f16 GGUF for 192 items and Q8_0 for the last 8 (`llama-server` was OOM-killed repeatedly on the shared box while serving the f16 file, so the remaining items were generated from a Q8_0 quantization of the same SFT GGUF); SFT+DPO = Q8_0 GGUF for all 200 (`--no-repack`, batch 1). Precision check: on a 40-item sample (10 per category) the SFT model at Q8_0 produced byte-identical text on 34/40 and the same verdict on 40/40 versus f16 (same accuracy, 28/40); the first 29 SFT+DPO items, generated once with llama.cpp's CPU weight repacking and once without, matched in text on 27/29 and in verdict on 29/29. Precision is therefore not what separates the models below, but it is not perfectly uniform either. Accuracy %, 95% percentile
bootstrap CI (2,000 resamples):

| BFCL category | n | base | sft | sft+dpo |
|---|---|---|---|---|
| bfcl/simple | 50 | 82.0 [72.0, 92.0] | 78.0 [66.0, 88.0] | 82.0 [70.0, 92.0] |
| bfcl/multiple | 50 | 76.0 [64.0, 88.0] | 68.0 [54.0, 80.0] | 70.0 [56.0, 82.0] |
| bfcl/parallel | 50 | 54.0 [40.0, 68.0] | 44.0 [30.0, 58.0] | 40.0 [26.0, 54.0] |
| bfcl/irrelevance | 50 | 48.0 [34.0, 62.0] | 42.0 [28.0, 56.0] | 40.0 [26.0, 54.0] |
| **macro avg** | | **65.0** | **58.0** | **58.0** |
| relevance: call recall | | 100.0 | 100.0 | 100.0 |
| relevance: no-call accuracy | | 48.0 | 42.0 | 40.0 |

Micro accuracy (all 200 items): base 65.0 [58.5, 71.5], sft 58.0 [51.5, 64.5], sft+dpo 58.0 [51.0, 64.5].

Paired comparison (same 200 items, paired bootstrap; "fixed" / "broken" = items the second model gets
right / wrong that the first got wrong / right; per-category columns are pp differences with 95% CIs,
n = 50 each). From `results/paired_analysis.json` (`scripts/paired_analysis.py`):

| Comparison | Δ micro (pp) [95% CI] | fixed | broken | irrelevance | multiple | parallel | simple |
|---|---|---|---|---|---|---|---|
| sft - base | -7.0 [-11.5, -3.0] | 2 | 16 | -6.0 [-14.0, +0.0] | -8.0 [-16.0, -2.0] | -10.0 [-20.0, +0.0] | -4.0 [-12.0, +4.0] |
| sft+dpo - base | -7.0 [-11.0, -3.5] | 1 | 15 | -8.0 [-16.0, -2.0] | -6.0 [-14.0, +0.0] | -14.0 [-24.0, -6.0] | +0.0 [-6.0, +6.0] |
| sft+dpo - sft | +0.0 [-3.0, +2.5] | 4 | 4 | -2.0 [-6.0, +0.0] | +2.0 [+0.0, +6.0] | -4.0 [-12.0, +4.0] | +4.0 [+0.0, +10.0] |

**Reading the numbers: this run did not improve tool calling.** Both fine-tuned models score 7 points *below* the base model on BFCL (paired 95% CIs exclude zero), and DPO on top of SFT changed nothing measurable (+0.0 pp, CI [−3.0, +2.5]; 4 items fixed, 4 broken). The losses are spread over every category, largest on `parallel` (glaive has no parallel calls, so SFT pushes the model towards single calls: `wrong_call_count` 11 → 12 → 15) and `irrelevance` (spurious calls 26 → 29 → 30). Call recall stays at 100% for all three. With 398 SFT examples, 50 SFT steps on the top 8 layers and 8 DPO steps, this is a measurement of a deliberately tiny CPU run, not of what the recipe does at scale; it does show the pipeline end to end and that the scorer is strict enough to register a regression.

The held-out glaive set (150 examples) was **not** evaluated in this run (CPU time).

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
- **Low-memory GGUF path**: `scripts/gguf_stream_merge.py` rewrites an F16 GGUF to Q8_0 (or F16)
  with plain reads in ~150 MB, optionally merging a PEFT LoRA adapter on the way. It produced the
  SFT+DPO GGUF of the measured run when `toolforge export` and `llama-quantize` could not run on the
  shared box; on the SFT model it matches `toolforge export` to within 1 f16 ULP.
- **Tracking**: MLflow local file store by default (`mlruns/`), W&B offline optional; every run also
  writes `metrics.jsonl` and `train_summary.json` next to its outputs.

## Development

```bash
make lint typecheck test   # ruff, mypy, unit tests (no torch needed)
make smoke                 # tiny random-model end-to-end run: data files → SFT → resume → DPO → eval → merge
make mutation              # testteeth mutation grade of scorer, schema, data cleaning, pairs, formatting
```

<!-- TESTING -->
**Tests**: 310 unit tests (`pytest -m "not smoke and not parity"`, no torch needed) plus a
5-test tiny-random-model smoke run (`pytest -m smoke tests/smoke`: data files → SFT →
resume → DPO → eval → merge).

**Mutation testing** ([testteeth](https://pypi.org/project/testteeth/) 0.2.0, run from a fresh clone,
2 workers) of the modules whose bugs would silently corrupt results: `scorer.py`, `schema.py`,
`data/clean.py`, `pairs.py`, `formatting.py`:

| | Mutation score |
|---|---|
| before (tests as first written) | 94.1% (412/438) |
| after strengthening the tests | 98.9% (433/438) |

Reports: [docs/mutation/](docs/mutation/). Score = (killed + timeouts) / mutants. The one real
surviving mutant (`pairs.py` L148) is now killed by `test_every_synthetic_kind_is_used_when_room_allows`
(checked by applying the mutant by hand); the full mutation run was not repeated, so 98.9% is the
last measured score.

CI runs lint, mypy, unit tests on Python 3.10–3.12, the tiny-model smoke run and a wheel install
smoke test; mutation testing runs as a separate non-blocking job. Tagging `v*` builds the sdist and
wheel, attaches them to a GitHub Release, and publishes to PyPI only when `PYPI_API_TOKEN` is set.

## Limitations

<!-- LIMITATIONS -->
- **Untested:** QLoRA (4-bit), every GPU path (`configs/gpu_qlora.yaml`, `scripts/gpu_run.sh`),
  vLLM serving, the Colab notebook, the Docker image, and Hugging Face Hub publishing. None of these
  was run; only the CPU path below was.
- **Tiny run.** 398 SFT examples and 64 DPO pairs on CPU. The numbers show the pipeline
  works end to end and measure *this* run; they are not what the method reaches at full scale.
  CIs on 50 examples per category are wide (roughly ±10–14 points); read the paired deltas, not
  single cells.
- **Mixed eval precision.** Base was served as f16, SFT as f16 for 192/200 items and Q8_0 for 8, SFT+DPO as Q8_0 for all 200, because the f16 server kept being OOM-killed on the shared 7 GB box. A 40-item f16-vs-Q8_0 check showed identical verdicts (40/40), but the comparison is not bit-for-bit like-for-like.
- **CPU LoRA, not QLoRA.** The measured run is fp32 LoRA. The 4-bit QLoRA path (`configs/gpu_qlora.yaml`,
  bitsandbytes, CUDA) is **untested**, as are `scripts/gpu_run.sh`, the Colab notebook
  (`notebooks/toolforge_gpu_colab.ipynb`) and vLLM serving (`deploy/vllm.yaml`): no GPU was available.
- **Docker untested.** The `Dockerfile` was not built; the `ghcr.io/ggml-org/llama.cpp:server-b11521`
  base-image tag it references is **unverified**.
- **Serving check is a smoke test.** With llama.cpp b11521, the SFT+DPO **Q8_0** GGUF was served with `toolforge serve` (`llama-server --jinja`, `-c 2048 -np 1 --no-repack`) and answered the README's `get_weather` request with a proper OpenAI `tool_calls` entry (`get_weather({"city": "Lisbon"})`, `finish_reason: tool_calls`); see `results/gguf_smoke.json`. No load, latency or multi-request testing was done.
- **`toolforge export` of the final model did not complete here.** The SFT GGUF came from
  `toolforge export`; the SFT+DPO GGUF came from `scripts/gguf_stream_merge.py` because the torch
  merge and `llama-quantize` were killed on the shared box (see the model card). `toolforge export`'s
  built-in Q8_0 step (`llama-quantize`) was therefore not exercised in this run.
- **Not published.** No model was pushed to the Hugging Face Hub; there is no hosted demo.
- **Eval scope.** Only BFCL simple / multiple / parallel / irrelevance (no multi-turn, no live/exec
  categories); the held-out glaive split was not evaluated in this run. Parallel calls are never
  trained (glaive has none).
- **glaive is synthetic** and repetitive; see the [data card](docs/DATA_CARD.md).

## License

Apache-2.0. Training data: glaive-function-calling-v2 (Apache-2.0). Evaluation data: BFCL / gorilla
(Apache-2.0). Base model: Qwen2.5-0.5B-Instruct (Apache-2.0).
