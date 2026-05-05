#!/usr/bin/env bash
# scripts/start-triton.sh — container entrypoint for Track B
#
# Launches the OpenAI-compatible frontend bundled in the Triton 25.05 image.
# main.py starts its OWN embedded Triton server (via Python bindings) internally,
# loads the model repository, and then starts Uvicorn on OPENAI_PORT.
#
# Do NOT start tritonserver separately — main.py embeds Triton. Running both
# causes GPU resource contention and "Failed to deserialize cuda engine" failures.
#
# Ports:
#   9000  OpenAI-compatible HTTP (/v1/chat/completions, /v1/models)
#   8000  Triton native HTTP (/v2/…) — opened by the embedded Triton
#   8001  Triton gRPC — opened by the embedded Triton
#   8002  Triton metrics — opened by the embedded Triton
#
# ENV vars (set in compose.yaml / $TRTLLM_HOME/.env):
#   TRITON_MODEL_REPO          path to model repository  (default: /model_repo)
#   OPENAI_TOKENIZER           path or HF name for tokenizer
#                              (default: /workspace/trtllm/models/qwen-coder-7b)
#   OPENAI_PORT                port for the OpenAI frontend  (default: 9000)
#   DEFAULT_MODEL              pipeline to auto-load at startup: qwen|llama|vision|''
#   TRITON_MODEL_CONTROL_MODE  explicit (default) | none

set -euo pipefail

TRITON_MODEL_REPO="${TRITON_MODEL_REPO:-/model_repo}"
OPENAI_TOKENIZER="${OPENAI_TOKENIZER:-/workspace/trtllm/models/qwen-coder-7b}"
OPENAI_PORT="${OPENAI_PORT:-9000}"

# triton_wrapper.py monkey-patches tritonserver.Server to inject EXPLICIT model-control
# mode before delegating to main.py.  Mounted at /triton_wrapper.py by compose.yaml.
WRAPPER="/triton_wrapper.py"

echo "[start-triton] Starting OpenAI frontend (via wrapper) on port ${OPENAI_PORT}"
echo "[start-triton] Model repository: ${TRITON_MODEL_REPO}"
echo "[start-triton] Tokenizer: ${OPENAI_TOKENIZER}"
echo "[start-triton] DEFAULT_MODEL: ${DEFAULT_MODEL:-'(none)'}"

# exec replaces this shell so Docker tracks the Python process directly.
# --enable-kserve-frontends activates the KServe HTTP control plane on port 8000
# (POST /v2/repository/models/{name}/load|unload) used by `just models-*` recipes.
exec python3 "${WRAPPER}" \
    --model-repository "${TRITON_MODEL_REPO}" \
    --tokenizer "${OPENAI_TOKENIZER}" \
    --openai-port "${OPENAI_PORT}" \
    --enable-kserve-frontends
