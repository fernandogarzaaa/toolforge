# Serving image: llama.cpp's OpenAI-compatible server + the exported GGUF.
#   docker build -t toolforge-serve --build-arg GGUF=outputs/gguf/toolforge-qwen2.5-0.5b-q8_0.gguf .
#   docker run -p 8091:8091 toolforge-serve
# Base image is the upstream llama.cpp server image (multi-arch); pin a tag for reproducibility.
ARG LLAMA_CPP_IMAGE=ghcr.io/ggml-org/llama.cpp:server-b11521
FROM ${LLAMA_CPP_IMAGE}

ARG GGUF=outputs/gguf/toolforge-qwen2.5-0.5b-q8_0.gguf
COPY ${GGUF} /models/toolforge.gguf

ENV LLAMA_ARG_MODEL=/models/toolforge.gguf \
    LLAMA_ARG_HOST=0.0.0.0 \
    LLAMA_ARG_PORT=8091 \
    LLAMA_ARG_CTX_SIZE=4096 \
    LLAMA_ARG_ALIAS=toolforge \
    LLAMA_ARG_JINJA=1

EXPOSE 8091
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD curl -fsS http://localhost:8091/health || exit 1
