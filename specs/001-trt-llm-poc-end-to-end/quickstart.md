# Quickstart: TRT-LLM POC End-to-End

**Feature**: `001-trt-llm-poc-end-to-end`  
**Date**: 2026-05-01  
**Purpose**: Get from zero to a working Rust smoke-test in the fewest steps

> This is the fast path. For full detail (flags, troubleshooting, ansible notes) see
> `docs/setup.md`. For architecture decisions see `research.md`.

---

## Prerequisites

These must be true on the host before starting. If any fail, see `docs/setup.md`.

```bash
nvidia-smi                           # RTX 5090 visible, driver 595+
docker info | grep -i nvidia         # NVIDIA runtime available
docker compose version               # Docker Compose V2 available
just --version                       # just task runner installed
cargo --version                      # Rust stable toolchain installed
```

---

## Step 1 — Clone and Init

```bash
git clone <repo-url> trt-llm-poc
cd trt-llm-poc

# Copy environment template to repo root and fill in your token
# (just loads .env from the repo root via set dotenv-load := true)
cp .env.example .env
$EDITOR .env
# TRTLLM_HOME=/ai/trtllm-poc is the default — change only if you want a different location
# Set: HF_TOKEN=hf_...   (needed for gated models)

# Create directory layout under TRTLLM_HOME
just init

# Copy .env to TRTLLM_HOME so docker compose can read it when running from there
cp .env $TRTLLM_HOME/
```

---

## Step 2 — Pull Images (Track A + Track B)

```bash
just pull
# Downloads:
#   nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13   (~20 GB)
#   nvcr.io/nvidia/tritonserver:25.05-trtllm-python-py3  (~15 GB)
# This will take a while on first run.
```

---

## Step 3 — Track A: Prove GPU + Build an Engine

```bash
just dev
# Opens an interactive shell inside the TRT-LLM dev container

# Inside the container:
nvidia-smi                            # Confirm RTX 5090 is visible
python -c "import tensorrt_llm; print(tensorrt_llm.__version__)"

# Download a model.
# Qwen2.5-Coder-7B is ungated and one of the three planned models — use it first.
# (Llama 3.1 8B requires HuggingFace access approval; swap in once approved.)
cd /workspace/trtllm
huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct \
  --local-dir /workspace/trtllm/models/qwen-coder-7b \
  --local-dir-use-symlinks False

# Convert HuggingFace weights to TRT-LLM checkpoint format
python /app/tensorrt_llm/examples/models/core/qwen/convert_checkpoint.py \
  --model_dir /workspace/trtllm/models/qwen-coder-7b \
  --output_dir /workspace/trtllm/models/qwen-coder-7b-ckpt \
  --dtype float16

# Build TRT-LLM engine (output goes to the real target dir for this model)
trtllm-build \
  --checkpoint_dir /workspace/trtllm/models/qwen-coder-7b-ckpt \
  --output_dir /workspace/trtllm/engines/qwen-coder \
  --gemm_plugin float16 \
  --max_batch_size 1

# Quick inference sanity check
python /app/tensorrt_llm/examples/run.py \
  --engine_dir /workspace/trtllm/engines/qwen-coder \
  --tokenizer_dir /workspace/trtllm/models/qwen-coder-7b \
  --input_text "Write a Python function that returns the nth Fibonacci number." \
  --max_output_len 128

# Exit container when done
exit
```

> Engines are persisted to `$TRTLLM_HOME/engines/` on the host. Track B uses them directly.

---

## Step 4 — Track B: Stand Up Triton Serving

```bash
# Install compose.yaml into TRTLLM_HOME
just compose-install

# Start Triton (detached, restarts on reboot)
just compose-up

# Watch startup logs until "Started GRPCInferenceService" appears
just compose-logs
# Ctrl-C to stop following logs

# Verify HTTP endpoint is reachable
curl http://localhost:8000/v1/models
# Expected: {"object":"list","data":[...]}
```

---

## Step 5 — Build and Run the Rust Client

```bash
cd rust-client

# Build
cargo build

# Run routing unit tests
cargo test

# Smoke test against live endpoint (all three capabilities)
cargo run -- smoke-test
# Expected output:
#   [chat]    "Hello! I'm an AI assistant..." (or similar)
#   [code]    "fn fibonacci(n: u32) -> u32 {..." (or similar)
#   [caption] "The image shows a..." (or similar)
#   All requests logged with latency + token counts
```

---

## Step 6 — Verify Acceptance Checklist

```bash
# From repo root, confirm each item in the spec acceptance checklist:

just gpu                 # SC-001: nvidia-smi shows RTX 5090
just ps                  # SC-003: triton-trtllm container is Up
cargo test -p rust-client  # SC-005: all tests pass
# Manual: reboot host, confirm container restarts → SC-003
# Manual: check docs/setup.md + docs/ansible-handoff.md are complete → SC-006
```

---

## Troubleshooting Quick Reference

| Symptom | First thing to try |
|---------|--------------------|
| `nvidia-smi` fails inside container | Check `docker info \| grep -i nvidia`; reinstall NVIDIA Container Toolkit |
| NGC image pull fails | Check NGC tag is still published; try `latest` tag and note actual version |
| Engine build OOM | Reduce `--max_batch_size`; ensure no other GPU processes are running |
| Triton container exits on startup | `just compose-logs` — look for missing engine path or config.pbtxt error |
| `curl /v1/models` returns 503 | Model not loaded yet; wait 30–60s after `compose-up` |
| Rust build fails | `rustup update stable`; check `Cargo.toml` dependency versions |

For detailed troubleshooting see `docs/setup.md` §6.
