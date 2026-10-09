#!/usr/bin/env bash
# GPU path (Colab / Kaggle / any CUDA box). UNTESTED by the author: no GPU was available.
# QLoRA (4-bit NF4 via bitsandbytes) on the full cleaned set, then DPO, eval, export.
set -euo pipefail
CONFIG=${CONFIG:-configs/gpu_qlora.yaml}

pip install -q -e ".[train,qlora,export]"

toolforge data   -c "$CONFIG"
toolforge sft    -c "$CONFIG"          # resumes from outputs/sft/checkpoint-* if interrupted
toolforge dpo    -c "$CONFIG"          # samples the SFT model, mines pairs, trains DPO (resumable)
toolforge eval   -c "$CONFIG"
# GGUF export needs a llama.cpp checkout for convert_hf_to_gguf.py:
if [ ! -d llama.cpp ]; then git clone --depth 1 https://github.com/ggml-org/llama.cpp; fi
LLAMA_CPP_DIR=$PWD/llama.cpp toolforge export -c "$CONFIG" -o export.quantize=[]
