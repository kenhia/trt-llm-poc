# justfile — TRT-LLM / Triton lifecycle management
# Loads .env from the repo root automatically (set dotenv-load := true).
# Usage: cp .env.example .env, fill in values, then run `just` from the repo root.

set dotenv-load := true

# === Images ===
# Track A: TRT-LLM dev image — for interactive engine builds and experimentation.
IMAGE_DEV    := "nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13"
# Track B: Triton serving image — exposes HTTP/gRPC endpoints for Rust client calls.
IMAGE_TRITON := "nvcr.io/nvidia/tritonserver:25.05-trtllm-python-py3"

# === Container name (used by compose and one-off run targets) ===
SERVICE := "triton-trtllm"

# === TRTLLM_HOME fallback (overridden by .env) ===
TRTLLM_HOME := env_var_or_default("TRTLLM_HOME", "/ai/trtllm-poc")

# Default target — list all available recipes
default:
    @just --list

# --- Init ---
# Create the required directory layout under TRTLLM_HOME on the host.
# Run this once before any other recipe.
init:
    mkdir -p {{TRTLLM_HOME}}/models \
             {{TRTLLM_HOME}}/engines \
             {{TRTLLM_HOME}}/cache \
             {{TRTLLM_HOME}}/logs \
             {{TRTLLM_HOME}}/model_repo
    @echo "TRTLLM_HOME initialised at {{TRTLLM_HOME}}"

# Install compose.yaml and all scripts/ into TRTLLM_HOME so `docker compose` can find them.
# Re-run after any change to compose.yaml or scripts/ in the repo.
compose-install:
    cp compose.yaml {{TRTLLM_HOME}}/compose.yaml
    mkdir -p {{TRTLLM_HOME}}/scripts
    cp scripts/start-triton.sh {{TRTLLM_HOME}}/scripts/start-triton.sh
    cp scripts/triton_wrapper.py {{TRTLLM_HOME}}/scripts/triton_wrapper.py
    cp scripts/setup-model-repo.sh {{TRTLLM_HOME}}/scripts/setup-model-repo.sh
    cp scripts/vision-smoke-test.py {{TRTLLM_HOME}}/scripts/vision-smoke-test.py
    chmod +x {{TRTLLM_HOME}}/scripts/start-triton.sh
    chmod +x {{TRTLLM_HOME}}/scripts/setup-model-repo.sh
    @echo "compose.yaml + scripts/ installed to {{TRTLLM_HOME}}"

# Create or rename Triton model pipeline directories in TRTLLM_HOME/model_repo/.
# Run once per pipeline before loading it for the first time.
#   just setup-model-repo qwen    # rename bare dirs to *_qwen (existing qwen-coder pipeline)
#   just setup-model-repo llama   # create *_llama dirs from inflight_batcher_llm template
#   just setup-model-repo vision  # create vision pipeline from multimodal template
setup-model-repo pipeline:
    TRTLLM_HOME={{TRTLLM_HOME}} bash {{TRTLLM_HOME}}/scripts/setup-model-repo.sh {{pipeline}}

# --- Images ---
# Pull both NGC images. This will take ~30 GB on first run.
# Record the pulled tags/digests in docs/ansible-handoff.md (T012).
pull:
    docker pull {{IMAGE_DEV}}
    docker pull {{IMAGE_TRITON}}
    @echo "Run: docker images --digests | grep -E 'tensorrt-llm|tritonserver'"

# --- Track A: interactive dev shell ---
# Opens a shell inside the TRT-LLM dev container with GPU access and TRTLLM_HOME mounted.
# Use this to build engines with trtllm-build and run inference smoke tests.
# --ipc=host: required to avoid shared-memory failures with large models.
# --ulimit memlock=-1: allow locking all GPU memory (needed for TRT-LLM).
dev:
    docker run --rm -it \
      --gpus all \
      --ipc=host \
      --ulimit memlock=-1 \
      --ulimit stack=67108864 \
      -e HF_TOKEN="$HF_TOKEN" \
      -v "{{TRTLLM_HOME}}":/workspace/trtllm \
      -w /workspace \
      {{IMAGE_DEV}}

# --- Track B: Triton via compose ---
# Start the Triton serving container in detached mode.
# restart: unless-stopped means it will come back after a reboot automatically.
compose-up:
    cd {{TRTLLM_HOME}} && docker compose up -d

# Stop and remove the Triton container (does not remove image or volumes).
compose-down:
    cd {{TRTLLM_HOME}} && docker compose down

# Follow Triton container logs. Press Ctrl-C to stop following.
# Watch for "Started GRPCInferenceService" to know the server is ready.
compose-logs:
    cd {{TRTLLM_HOME}} && docker compose logs -f

# Restart the Triton container without rebuilding (useful after config.pbtxt changes).
compose-restart:
    cd {{TRTLLM_HOME}} && docker compose restart

# --- Track B: one-off Triton run (non-compose) ---
# Use this to test Triton interactively without autostart. Requires manual cleanup.
# --shm-size=2g: shared memory for Triton; increase to 4g/8g if you see IPC errors.
run-triton:
    docker run --rm -it \
      --gpus all \
      --ulimit memlock=-1 \
      --ulimit stack=67108864 \
      --shm-size=2g \
      -e HF_TOKEN="$HF_TOKEN" \
      -p 8000:8000 \
      -p 8001:8001 \
      -p 8002:8002 \
      -v "{{TRTLLM_HOME}}":/workspace/trtllm \
      -w /workspace \
      {{IMAGE_TRITON}}

# --- Diagnostics ---
# Confirm the RTX 5090 is visible to the host (SC-001 prerequisite).
gpu:
    nvidia-smi

# Show all containers (running and stopped).
ps:
    docker ps -a

# --- Model control (KServe v2 repository API, port 8000) ---
# These recipes require Triton to be running in EXPLICIT model-control mode
# (the default when using triton_wrapper.py).
#
# Pipelines: qwen | llama | vision
# Load order: preprocessing → tensorrt_llm → postprocessing → tensorrt_llm_bls → ensemble
# Unload order: reverse of above (ensemble first, preprocessing last)
#
# Usage:
#   just models-load qwen     # load the qwen-coder pipeline
#   just models-unload llama  # free VRAM used by the llama pipeline
#   just models-status        # list all pipelines and their states

KSERVE_HTTP := env_var_or_default("KSERVE_HTTP_PORT", "8000")
KSERVE_BASE := "http://localhost:" + KSERVE_HTTP

# Load a model pipeline. Sends load requests in dependency order.
# Exits non-zero on first failure. Idempotent: re-loading a ready model is safe.
models-load pipeline:
    #!/usr/bin/env bash
    set -euo pipefail
    BASE="{{KSERVE_BASE}}"
    P="{{pipeline}}"
    case "$P" in
        qwen)   MODELS="preprocessing_qwen tensorrt_llm_qwen postprocessing_qwen tensorrt_llm_bls_qwen ensemble_qwen" ;;
        llama)  MODELS="preprocessing_llama tensorrt_llm_llama postprocessing_llama tensorrt_llm_bls_llama ensemble_llama" ;;
        vision) MODELS="preprocessing_vision multimodal_encoders tensorrt_llm_vision postprocessing_vision ensemble_vision" ;;
        *) echo "Unknown pipeline: $P  (valid: qwen | llama | vision)" >&2; exit 1 ;;
    esac
    for model in $MODELS; do
        echo "Loading ${model}..."
        status=$(curl -s -o /dev/null -w "%{http_code}" -X POST "${BASE}/v2/repository/models/${model}/load")
        if [ "$status" -ne 200 ]; then
            echo "  FAILED (HTTP ${status})" >&2
            exit 1
        fi
        echo "  OK"
    done
    echo "Pipeline '${P}' loaded."

# Unload a model pipeline. Sends unload requests in reverse dependency order.
# Idempotent: unloading an already-unloaded model is not an error.
models-unload pipeline:
    #!/usr/bin/env bash
    set -euo pipefail
    BASE="{{KSERVE_BASE}}"
    P="{{pipeline}}"
    case "$P" in
        qwen)   MODELS="ensemble_qwen tensorrt_llm_bls_qwen postprocessing_qwen tensorrt_llm_qwen preprocessing_qwen" ;;
        llama)  MODELS="ensemble_llama tensorrt_llm_bls_llama postprocessing_llama tensorrt_llm_llama preprocessing_llama" ;;
        vision) MODELS="ensemble_vision postprocessing_vision tensorrt_llm_vision multimodal_encoders preprocessing_vision" ;;
        *) echo "Unknown pipeline: $P  (valid: qwen | llama | vision)" >&2; exit 1 ;;
    esac
    for model in $MODELS; do
        echo "Unloading ${model}..."
        status=$(curl -s -o /dev/null -w "%{http_code}" -X POST "${BASE}/v2/repository/models/${model}/unload")
        if [ "$status" -ne 200 ] && [ "$status" -ne 404 ]; then
            echo "  WARNING: unexpected HTTP ${status} — continuing"
        else
            echo "  OK"
        fi
    done
    echo "Pipeline '${P}' unloaded."

# Show the state of all Triton models. Groups output by pipeline.
# Requires Triton KServe HTTP frontend (port 8000) to be active.
models-status:
    #!/usr/bin/env bash
    BASE="{{KSERVE_BASE}}"
    echo "Querying ${BASE}/v2/repository/index ..."
    result=$(curl -s -X POST "${BASE}/v2/repository/index" \
        -H 'Content-Type: application/json' \
        -d '{"ready": false}')
    if [ -z "$result" ]; then
        echo "ERROR: no response from ${BASE} — is Triton running?" >&2
        exit 1
    fi
    echo ""
    echo "Model                          State"
    echo "─────────────────────────────  ──────────"
    echo "$result" | python3 -c 'import json,sys;d=json.load(sys.stdin);[print("{:<30}  {}".format(m.get("name",""),m.get("state","UNKNOWN"))) for m in sorted(d,key=lambda m:m.get("name",""))]'

# --- Engine builds (runs inside the Triton 25.05 container with GPU) ---

# Build the Llama 3.1 8B Instruct TRT-LLM engine.
# Requires: HF_TOKEN env var (gated model), GPU, ~16 GB VRAM, ~30-60 min.
# WARNING: Stop the Triton serving container first (`just compose-down`) — both
#          the build and serving containers claim the full GPU, causing OOM if run together.
# Output: TRTLLM_HOME/engines/llama/rank0.engine + config.json
build-llama:
    docker run --rm --gpus all \
      --ulimit memlock=-1 \
      --ulimit stack=67108864 \
      --shm-size=2g \
      -e HF_TOKEN="$HF_TOKEN" \
      -v "{{TRTLLM_HOME}}":/workspace/trtllm \
      -w /workspace/trtllm \
      {{IMAGE_TRITON}} \
      bash -c '\
        pip install -q 'huggingface_hub[hf_xet]' && \
        huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct \
          --local-dir /workspace/trtllm/models/llama-3.1-8b \
          --token "$HF_TOKEN" && \
        python3 /app/examples/llama/convert_checkpoint.py \
          --model_dir /workspace/trtllm/models/llama-3.1-8b \
          --output_dir /workspace/trtllm/engines/llama/ckpt \
          --dtype float16 && \
        trtllm-build \
          --checkpoint_dir /workspace/trtllm/engines/llama/ckpt \
          --output_dir /workspace/trtllm/engines/llama \
          --gemm_plugin float16 \
          --max_batch_size 4 \
          --max_input_len 2048 \
          --max_seq_len 4096 \
      '
    sudo chown -R "$USER:$USER" "{{TRTLLM_HOME}}/engines/llama"
    @echo "Llama engine built at {{TRTLLM_HOME}}/engines/llama/"

# Build the LLaVA 1.5 7B vision engine (vision encoder + LLaMA-7B backbone).
# Uses the multimodal Triton pipeline (not inflight_batcher_llm).
# LLaVA 1.5 7B is confirmed [x] supported in TRT-LLM 0.19.0 C++ runtime.
# Requires: GPU, ~15 GB VRAM, ~20-40 min.
# WARNING: Stop the Triton serving container first (`just compose-down`) — both
#          the build and serving containers claim the full GPU, causing OOM if run together.
# Output: TRTLLM_HOME/engines/llava/vision/ + engines/llava/llm/
build-llava:
    docker run --rm --gpus all \
      --ulimit memlock=-1 \
      --ulimit stack=67108864 \
      --shm-size=2g \
      -v "{{TRTLLM_HOME}}":/workspace/trtllm \
      -w /workspace/trtllm \
      {{IMAGE_TRITON}} \
      bash -c '\
        pip install -q 'huggingface_hub[hf_xet]' && \
        huggingface-cli download llava-hf/llava-1.5-7b-hf \
          --local-dir /workspace/trtllm/models/llava-1.5-7b && \
        python3 /app/examples/multimodal/build_multimodal_engine.py \
          --model_path /workspace/trtllm/models/llava-1.5-7b \
          --output_dir /workspace/trtllm/engines/llava/vision \
          --model_type llava && \
        python3 /app/examples/llama/convert_checkpoint.py \
          --model_dir /workspace/trtllm/models/llava-1.5-7b \
          --output_dir /workspace/trtllm/engines/llava/ckpt \
          --dtype float16 && \
        trtllm-build \
          --checkpoint_dir /workspace/trtllm/engines/llava/ckpt \
          --output_dir /workspace/trtllm/engines/llava/llm \
          --gemm_plugin float16 \
          --max_multimodal_len 2304 \
          --max_batch_size 4 \
          --max_input_len 2048 \
          --max_seq_len 4096 \
      '
    sudo chown -R "$USER:$USER" "{{TRTLLM_HOME}}/engines/llava"
    @echo "LLaVA engine built at {{TRTLLM_HOME}}/engines/llava/"

# --- Smoke tests ---

# Run the Rust client smoke-test against the currently-loaded pipeline.
# Caption (LLaVA 1.5) is automatically delegated to smoke-vision when
# ensemble_vision is available; the Rust client skips it (binary protocol needed).
smoke:
    #!/usr/bin/env bash
    set -euo pipefail
    cd rust-client && cargo run -- smoke-test
    # If vision pipeline is loaded, also run the Python binary-protocol test.
    if curl -sf http://localhost:8000/v2/models/ensemble_vision/ready >/dev/null 2>&1; then
        just smoke-vision
    fi

# Run the vision pipeline smoke test via the Triton native HTTP binary API.
# LLaVA 1.5 requires client-side image preprocessing (FP16 pixel values),
# so the OpenAI /v1/chat/completions path is not used for vision.
# Requires: vision pipeline loaded (just models-load vision).
smoke-vision:
    python3 {{TRTLLM_HOME}}/scripts/vision-smoke-test.py

# Run the full three-pipeline smoke test sequence:
# load qwen → test Code → unload → load llama → test Chat → unload → load vision → test Caption → unload.
# Requires all three engines to be built and pipelines set up in model_repo/.
smoke-all:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "=== smoke-all: qwen (Code) ==="
    just models-load qwen
    (cd rust-client && cargo run -- smoke-test) || { just models-unload qwen; exit 1; }
    just models-unload qwen

    echo "=== smoke-all: llama (Chat) ==="
    just models-load llama
    (cd rust-client && cargo run -- smoke-test) || { just models-unload llama; exit 1; }
    just models-unload llama

    echo "=== smoke-all: vision (Caption) ==="
    just models-load vision
    just smoke-vision || { just models-unload vision; exit 1; }
    just models-unload vision

    echo "=== smoke-all PASSED ==="

# --- Cleanup ---
# Stop compose services, remove pulled images, and prune dangling Docker objects.
# WARNING: this removes the NGC images (~30 GB); you will need to re-pull.
purge:
    -cd {{TRTLLM_HOME}} && docker compose down --remove-orphans
    -docker rmi {{IMAGE_DEV}} {{IMAGE_TRITON}}
    -docker system prune -f
