Accuracy %, 95% bootstrap CI in brackets. 200 examples.

| Category | n | base | sft | sft+dpo |
|---|---|---|---|---|
| bfcl/irrelevance | 50 | 48.0 [34.0, 62.0] | 42.0 [28.0, 56.0] | 40.0 [26.0, 54.0] |
| bfcl/multiple | 50 | 76.0 [64.0, 88.0] | 68.0 [54.0, 80.0] | 70.0 [56.0, 82.0] |
| bfcl/parallel | 50 | 54.0 [40.0, 68.0] | 44.0 [30.0, 58.0] | 40.0 [26.0, 54.0] |
| bfcl/simple | 50 | 82.0 [72.0, 92.0] | 78.0 [66.0, 88.0] | 82.0 [70.0, 92.0] |
| **macro avg** | | **65.0** | **58.0** | **58.0** |
| relevance: call recall | | 100.0 | 100.0 | 100.0 |
| relevance: no-call acc. | | 48.0 | 42.0 | 40.0 |

Paired bootstrap difference in micro accuracy (percentage points, 95% CI):

- sft - base: -7.0 [-11.5, -3.0]
- sft+dpo - base: -7.0 [-11.0, -3.5]

<!-- APPENDIX -->
## Run details (2026-10-10, shared 2-vCPU aarch64 box, no GPU)

Training: SFT 398 examples / 50 steps (fp32 LoRA, top 8 layers); DPO
64 pairs / 8 steps. Eval: BFCL v4, 200 items (50 per category),
greedy, max 160 new tokens, llama.cpp b11521 `llama-server`.

**Precision per model:** base = f16 GGUF (200/200); SFT = f16 GGUF for 192 items and Q8_0 for the last 8 (`llama-server` was OOM-killed repeatedly on the shared box while serving the f16 file, so the remaining items were generated from a Q8_0 quantization of the same SFT GGUF); SFT+DPO = Q8_0 GGUF for all 200 (`--no-repack`, batch 1). Precision check: on a 40-item sample (10 per category) the SFT model at Q8_0 produced byte-identical text on 34/40 and the same verdict on 40/40 versus f16 (same accuracy, 28/40); the first 29 SFT+DPO items, generated once with llama.cpp's CPU weight repacking and once without, matched in text on 27/29 and in verdict on 29/29. Precision is therefore not what separates the models below, but it is not perfectly uniform either.

Micro accuracy (all 200 items): base 65.0 [58.5, 71.5], sft 58.0 [51.5, 64.5], sft+dpo 58.0 [51.0, 64.5].

## Paired comparison

From `results/paired_analysis.json` (`python scripts/paired_analysis.py`), paired bootstrap with
2,000 resamples over the same items. fixed / broken = items the second model gets right / wrong that
the first got wrong / right.

| Comparison | Δ micro (pp) [95% CI] | fixed | broken | irrelevance | multiple | parallel | simple |
|---|---|---|---|---|---|---|---|
| sft - base | -7.0 [-11.5, -3.0] | 2 | 16 | -6.0 [-14.0, +0.0] | -8.0 [-16.0, -2.0] | -10.0 [-20.0, +0.0] | -4.0 [-12.0, +4.0] |
| sft+dpo - base | -7.0 [-11.0, -3.5] | 1 | 15 | -8.0 [-16.0, -2.0] | -6.0 [-14.0, +0.0] | -14.0 [-24.0, -6.0] | +0.0 [-6.0, +6.0] |
| sft+dpo - sft | +0.0 [-3.0, +2.5] | 4 | 4 | -2.0 [-6.0, +0.0] | +2.0 [+0.0, +6.0] | -4.0 [-12.0, +4.0] | +4.0 [+0.0, +10.0] |

**Reading the numbers: this run did not improve tool calling.** Both fine-tuned models score 7 points *below* the base model on BFCL (paired 95% CIs exclude zero), and DPO on top of SFT changed nothing measurable (+0.0 pp, CI [−3.0, +2.5]; 4 items fixed, 4 broken). The losses are spread over every category, largest on `parallel` (glaive has no parallel calls, so SFT pushes the model towards single calls: `wrong_call_count` 11 → 12 → 15) and `irrelevance` (spurious calls 26 → 29 → 30). Call recall stays at 100% for all three. With 398 SFT examples, 50 SFT steps on the top 8 layers and 8 DPO steps, this is a measurement of a deliberately tiny CPU run, not of what the recipe does at scale; it does show the pipeline end to end and that the scorer is strict enough to register a regression.

GGUF smoke check: the SFT+DPO **Q8_0** GGUF was served with `toolforge serve` (`llama-server --jinja`, `-c 2048 -np 1 --no-repack`) and answered the README's `get_weather` request with a proper OpenAI `tool_calls` entry (`get_weather({"city": "Lisbon"})`, `finish_reason: tool_calls`); see `results/gguf_smoke.json`. No load, latency or multi-request testing was done.

