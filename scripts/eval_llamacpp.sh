#!/usr/bin/env bash
# Evaluate base / SFT / SFT+DPO through llama.cpp (the serving path) instead of HF transformers.
# For each model: merge adapters -> GGUF (f16) -> start llama-server -> `toolforge eval --only <name>`
# -> stop the server. Prompts are rendered by toolforge and sent to /v1/completions, so scoring is
# byte-identical to the HF backend. Needs LLAMA_CPP_DIR (converter) and LLAMA_CPP_BIN (server).
set -euo pipefail
CONFIG=${CONFIG:-configs/cpu_small.yaml}
PORT=${PORT:-8097}
THREADS=${THREADS:-2}
: "${LLAMA_CPP_DIR:?set LLAMA_CPP_DIR to a llama.cpp checkout (for convert_hf_to_gguf.py)}"
: "${LLAMA_CPP_BIN:?set LLAMA_CPP_BIN to the directory containing llama-server}"
export LD_LIBRARY_PATH="${LLAMA_CPP_BIN}:${LD_LIBRARY_PATH:-}"
ENDPOINT="http://127.0.0.1:${PORT}/v1"
MODELS_OVERRIDE="eval.models=[{name: base, backend: openai, endpoint: '${ENDPOINT}'}, {name: sft, backend: openai, endpoint: '${ENDPOINT}'}, {name: sft+dpo, backend: openai, endpoint: '${ENDPOINT}'}]"

adapters_for() {
  case "$1" in
    base) echo "[]" ;;
    sft) echo "[outputs/sft/adapter]" ;;
    sft+dpo) echo "[outputs/sft/adapter, outputs/dpo/adapter]" ;;
    *) echo "unknown model $1" >&2; exit 2 ;;
  esac
}

for name in ${MODELS:-base sft sft+dpo}; do
  slug=${name//+/_}
  gguf=outputs/gguf/eval-${slug}-f16.gguf
  if [ ! -f "$gguf" ]; then
    toolforge export -c "$CONFIG" \
      -o "export.adapters=$(adapters_for "$name")" \
      -o "export.merged_dir=outputs/merged-${slug}" \
      -o "export.gguf_path=${gguf}" \
      -o "export.quantize=[]"
    rm -rf "outputs/merged-${slug}"   # keep only the GGUF
  fi
  "${LLAMA_CPP_BIN}/llama-server" -m "$gguf" --host 127.0.0.1 --port "$PORT" -c 4096 -t "$THREADS" \
    --alias toolforge -np 1 > "outputs/gguf/server-${slug}.log" 2>&1 &
  pid=$!
  trap 'kill $pid 2>/dev/null || true' EXIT
  for _ in $(seq 1 150); do
    curl -fsS "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1 && break
    sleep 2
  done
  toolforge eval -c "$CONFIG" --only "$name" -o "$MODELS_OVERRIDE"
  kill $pid; wait $pid 2>/dev/null || true
  trap - EXIT
done
