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
#   TRITON_MODEL_REPO   — path to model repository  (default: /model_repo)
#   OPENAI_TOKENIZER    — path or HF name for tokenizer
#                         (default: /workspace/trtllm/models/qwen-coder-7b)
#   OPENAI_PORT         — port for the OpenAI frontend  (default: 9000)

set -euo pipefail

TRITON_MODEL_REPO="${TRITON_MODEL_REPO:-/model_repo}"
OPENAI_TOKENIZER="${OPENAI_TOKENIZER:-/workspace/trtllm/models/qwen-coder-7b}"
OPENAI_PORT="${OPENAI_PORT:-9000}"

OPENAI_FRONTEND="/opt/tritonserver/python/openai/openai_frontend/main.py"

echo "[start-triton] Starting OpenAI frontend on port ${OPENAI_PORT}"
echo "[start-triton] Model repository: ${TRITON_MODEL_REPO}"
echo "[start-triton] Tokenizer: ${OPENAI_TOKENIZER}"

# exec replaces this shell so Docker tracks the Python process directly.
# main.py starts an embedded Triton, loads models, then starts Uvicorn.
exec python3 "${OPENAI_FRONTEND}" \
    --model-repository "${TRITON_MODEL_REPO}" \
    --tokenizer "${OPENAI_TOKENIZER}" \
    --openai-port "${OPENAI_PORT}"
