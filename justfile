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

# Install compose.yaml and scripts/ into TRTLLM_HOME so `docker compose` can find them.
# Re-run after any change to compose.yaml or scripts/ in the repo.
compose-install:
    cp compose.yaml {{TRTLLM_HOME}}/compose.yaml
    mkdir -p {{TRTLLM_HOME}}/scripts
    cp scripts/start-triton.sh {{TRTLLM_HOME}}/scripts/start-triton.sh
    chmod +x {{TRTLLM_HOME}}/scripts/start-triton.sh
    @echo "compose.yaml + scripts/ installed to {{TRTLLM_HOME}}"

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

# --- Cleanup ---
# Stop compose services, remove pulled images, and prune dangling Docker objects.
# WARNING: this removes the NGC images (~30 GB); you will need to re-pull.
purge:
    -cd {{TRTLLM_HOME}} && docker compose down --remove-orphans
    -docker rmi {{IMAGE_DEV}} {{IMAGE_TRITON}}
    -docker system prune -f
