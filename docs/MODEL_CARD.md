# Model card: toolforge-qwen2.5-0.5b (CPU demonstration run)

**Not published.** This card describes the artifact produced locally by the measured run in the
README. No weights were uploaded anywhere.

## Model

- **Base**: [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), revision
  `7ae557604adf67be50417f59c2c2f167def9a775` (Apache-2.0).
- **Adapters**: SFT LoRA, then a DPO LoRA trained on top of the merged SFT weights; both merged into
  the exported checkpoint. r=16, alpha=32, dropout 0.05, q/k/v/o/gate/up/down projections of the top
  8 of 24 decoder layers.
- **Precision / hardware**: fp32 LoRA on a shared 2-vCPU aarch64 CPU. **Not QLoRA.**
- **Export**: the SFT+DPO GGUF was produced by `scripts/gguf_stream_merge.py`, not by `toolforge export`: the torch merge of the two stacked adapters and `llama-quantize` (which mmaps its input) were OOM-killed / SIGBUSed repeatedly on the shared box. The script streams the SFT f16 GGUF (itself made by `toolforge export`: torch merge → `convert_hf_to_gguf.py`, llama.cpp b11521), adds the DPO LoRA delta in f32 and writes Q8_0. Checked: base f16 GGUF + SFT adapter through the script matches `toolforge export`'s SFT GGUF to within 1 f16 ULP on 8,375 of ~494M weights and is bit-identical elsewhere.
- **Prompt format**: Qwen2.5 chat template with `<tools>` system block; calls are emitted as
  `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`.

## Intended use

A demonstration of the toolforge pipeline: single-turn function-call decisions (call one function,
call several in parallel, or answer without a call) for small local agents. Not evaluated for
multi-turn tool use, executing calls, safety, or any production use.

**This checkpoint scores below its own base model on BFCL** (−7 pp micro accuracy, paired 95% CI
excludes zero; see Evaluation). If you need a small tool-calling model, use the base
Qwen2.5-0.5B-Instruct rather than this artifact.

## Training data

- **SFT**: 398 decision examples from glaive-function-calling-v2 (Apache-2.0), cleaned,
  schema-validated, deduplicated and decontaminated against the eval sets (see
  [DATA_CARD.md](DATA_CARD.md)). Categories in the trained subset: clarify 41, irrelevance 62, multiple 142, simple 155.
- **DPO**: 64 pairs from 64 disjoint glaive prompts: chosen = gold, rejected = the
  SFT model's own wrong greedy output where it erred, otherwise a verified synthetic corruption
  (9 from the SFT model's own mistakes, 55 verified synthetic corruptions).

## Training procedure

| | SFT | DPO |
|---|---|---|
| examples / pairs | 398 | 64 |
| max length (tokens) | 512 | 512 |
| epochs | 1 | 1 |
| effective batch | 8 | 8 |
| optimizer steps | 50 | 8 |
| learning rate | 2e-4 cosine | 5e-5 cosine |
| final train loss | 0.119 | 0.617 |
| wall time (summed over resumes) | 1 h 40 min (summed over 6 attempts) | 22 min 11 s (summed over 2 attempts, incl. pair mining) |

## Evaluation

BFCL v4 (gorilla commit `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`), 200 examples
(50 per category), AST scoring as described in the README, served through llama.cpp
(base f16; SFT f16 for 192 items + Q8_0 for 8; SFT+DPO Q8_0; see the README for the precision check). Accuracy %, 95% bootstrap CI:

| BFCL category | n | base | sft | sft+dpo |
|---|---|---|---|---|
| bfcl/simple | 50 | 82.0 [72.0, 92.0] | 78.0 [66.0, 88.0] | 82.0 [70.0, 92.0] |
| bfcl/multiple | 50 | 76.0 [64.0, 88.0] | 68.0 [54.0, 80.0] | 70.0 [56.0, 82.0] |
| bfcl/parallel | 50 | 54.0 [40.0, 68.0] | 44.0 [30.0, 58.0] | 40.0 [26.0, 54.0] |
| bfcl/irrelevance | 50 | 48.0 [34.0, 62.0] | 42.0 [28.0, 56.0] | 40.0 [26.0, 54.0] |
| **macro avg** | | **65.0** | **58.0** | **58.0** |
| relevance: call recall | | 100.0 | 100.0 | 100.0 |
| relevance: no-call accuracy | | 48.0 | 42.0 | 40.0 |

Micro accuracy: base 65.0 [58.5, 71.5], sft 58.0 [51.5, 64.5], sft+dpo 58.0 [51.0, 64.5]. Paired comparison (pp, 95% CI):

| Comparison | Δ micro (pp) [95% CI] | fixed | broken | irrelevance | multiple | parallel | simple |
|---|---|---|---|---|---|---|---|
| sft - base | -7.0 [-11.5, -3.0] | 2 | 16 | -6.0 [-14.0, +0.0] | -8.0 [-16.0, -2.0] | -10.0 [-20.0, +0.0] | -4.0 [-12.0, +4.0] |
| sft+dpo - base | -7.0 [-11.0, -3.5] | 1 | 15 | -8.0 [-16.0, -2.0] | -6.0 [-14.0, +0.0] | -14.0 [-24.0, -6.0] | +0.0 [-6.0, +6.0] |
| sft+dpo - sft | +0.0 [-3.0, +2.5] | 4 | 4 | -2.0 [-6.0, +0.0] | +2.0 [+0.0, +6.0] | -4.0 [-12.0, +4.0] | +4.0 [+0.0, +10.0] |

**Reading the numbers: this run did not improve tool calling.** Both fine-tuned models score 7 points *below* the base model on BFCL (paired 95% CIs exclude zero), and DPO on top of SFT changed nothing measurable (+0.0 pp, CI [−3.0, +2.5]; 4 items fixed, 4 broken). The losses are spread over every category, largest on `parallel` (glaive has no parallel calls, so SFT pushes the model towards single calls: `wrong_call_count` 11 → 12 → 15) and `irrelevance` (spurious calls 26 → 29 → 30). Call recall stays at 100% for all three. With 398 SFT examples, 50 SFT steps on the top 8 layers and 8 DPO steps, this is a measurement of a deliberately tiny CPU run, not of what the recipe does at scale; it does show the pipeline end to end and that the scorer is strict enough to register a regression.

## Limitations and honest notes

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

