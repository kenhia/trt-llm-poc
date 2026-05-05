# Quickstart: Three-Model POC

**Phase 1 output for `002-three-model-poc`**
**Date**: 2026-05-03
**Audience**: Developer picking up this sprint cold.

---

## Prerequisites

- Sprint 001 complete: `compose-install` done, qwen-coder engine at
  `/ai/trtllm-poc/engines/qwen-coder/`, `model_repo/` working.
- `TRTLLM_HOME=/ai/trtllm-poc` (default; override in `.env`).
- HuggingFace token with Llama access exported as `HF_TOKEN` in shell.
- ~40 GB free disk at `$TRTLLM_HOME` (two additional engines).

---

## Step 1 — Rename Qwen Pipeline to Suffixed Names

The existing qwen pipeline uses standard names (`ensemble`, `preprocessing`, etc.).
Rename them to `ensemble_qwen`, `preprocessing_qwen`, etc. to coexist with future pipelines.

```bash
# On the host, inside $TRTLLM_HOME/model_repo/
cd /ai/trtllm-poc/model_repo

for comp in preprocessing tensorrt_llm postprocessing ensemble tensorrt_llm_bls; do
  mv "${comp}" "${comp}_qwen"
done
```

Update all `config.pbtxt` files (the `setup-model-repo.sh` script does this):

```bash
# From repo root
just setup-model-repo qwen
```

Verify Triton still serves qwen after rename:

```bash
just compose-up
just models-status       # should show ensemble_qwen READY
cargo run -- smoke-test  # Code + Chat (still ensemble_qwen) should pass
```

---

## Step 2 — Build Llama 3.1 8B Engine (inside container)

```bash
# Start a disposable build container (GPU required)
docker run --rm -it --gpus all \
  -v /ai/trtllm-poc:/workspace/trtllm \
  -e HF_TOKEN=$HF_TOKEN \
  nvcr.io/nvidia/tritonserver:25.05-trtllm-python-py3 \
  bash

# Inside the container:
# 1. Download weights
pip install -q huggingface_hub
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct \
  --local-dir /workspace/trtllm/models/llama-3.1-8b \
  --token $HF_TOKEN

# 2. Convert checkpoint
python3 /app/examples/llama/convert_checkpoint.py \
  --model_dir /workspace/trtllm/models/llama-3.1-8b \
  --output_dir /workspace/trtllm/engines/llama/ckpt \
  --dtype float16

# 3. Build TRT engine
trtllm-build \
  --checkpoint_dir /workspace/trtllm/engines/llama/ckpt \
  --output_dir /workspace/trtllm/engines/llama \
  --gemm_plugin float16 \
  --max_batch_size 4 \
  --max_input_len 2048 \
  --max_seq_len 4096

exit
```

Or via `just` recipe (runs the docker build command for you):

```bash
just build-llama
```

Expected output: `/ai/trtllm-poc/engines/llama/rank0.engine` + `config.json`.

---

## Step 3 — Set Up Llama Pipeline in model_repo

```bash
just setup-model-repo llama
```

This creates the 5 `*_llama` model directories from the `inflight_batcher_llm` template,
fills in engine path and tokenizer, and updates all cross-references to use `_llama` suffixes.

```bash
just models-load llama
just models-status   # ensemble_llama should be READY
```

Test inference:

```bash
curl -s localhost:9000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"ensemble_llama","messages":[{"role":"user","content":"Hello, who are you?"}]}'
```

---

## Step 4 — Build LLaVA 1.5 7B Engine (inside container)

LLaVA 1.5 uses the `multimodal` Triton pipeline template.

```bash
docker run --rm -it --gpus all \
  -v /ai/trtllm-poc:/workspace/trtllm \
  nvcr.io/nvidia/tritonserver:25.05-trtllm-python-py3 \
  bash

# Inside the container:
# 1. Download weights (public, no token needed)
pip install -q huggingface_hub
huggingface-cli download llava-hf/llava-1.5-7b-hf \
  --local-dir /workspace/trtllm/models/llava-1.5-7b

# 2. Build vision encoder TRT engine
python3 /app/examples/multimodal/build_multimodal_engine.py \
  --model_path /workspace/trtllm/models/llava-1.5-7b \
  --output_dir /workspace/trtllm/engines/llava/vision \
  --model_type llava

# 3. Convert LLM checkpoint (LLaVA 1.5 uses LLaMA-7B backbone)
python3 /app/examples/llama/convert_checkpoint.py \
  --model_dir /workspace/trtllm/models/llava-1.5-7b \
  --output_dir /workspace/trtllm/engines/llava/ckpt \
  --dtype float16

# 4. Build LLM TRT engine with multimodal length support
trtllm-build \
  --checkpoint_dir /workspace/trtllm/engines/llava/ckpt \
  --output_dir /workspace/trtllm/engines/llava/llm \
  --gemm_plugin float16 \
  --max_multimodal_len 2048 \
  --max_batch_size 4 \
  --max_input_len 2048 \
  --max_seq_len 4096

exit
```

Or:

```bash
just build-llava
```

---

## Step 5 — Set Up Vision Pipeline in model_repo

```bash
just setup-model-repo vision
```

Creates the `multimodal_encoders`, `tensorrt_llm_vision`, `postprocessing_vision`,
and `ensemble_vision` directories from the `/app/all_models/multimodal/` template.

```bash
just models-load vision
just models-status   # ensemble_vision should be READY
```

Test image captioning:

```bash
# Encode a test image
B64=$(base64 -w0 /path/to/test.jpg)
curl -s localhost:9000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"ensemble_vision\",\"messages\":[{\"role\":\"user\",\"content\":[
    {\"type\":\"text\",\"text\":\"Describe this image.\"},
    {\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/jpeg;base64,${B64}\"}}
  ]}]}"
```

---

## Step 6 — Full Three-Model Smoke Test

Load each pipeline one at a time (VRAM constraint) and run the Rust smoke test:

```bash
just models-unload vision   # start clean
just models-load qwen
cargo run -- smoke-test     # Code capability → ensemble_qwen

just models-unload qwen
just models-load llama
cargo run -- smoke-test     # Chat capability → ensemble_llama

just models-unload llama
just models-load vision
cargo run -- smoke-test     # Caption capability → ensemble_vision
```

Final acceptance check — all three in sequence:

```bash
just smoke-all  # runs all three loads/tests/unloads automatically
```

---

## Model Control Workflow Reference

```bash
just models-load <qwen|llama|vision>    # load a pipeline
just models-unload <qwen|llama|vision>  # unload a pipeline
just models-status                       # list all pipelines and states
```

The `DEFAULT_MODEL` env var (in `.env` or `compose.yaml`) auto-loads a pipeline after
Triton becomes ready:

```bash
DEFAULT_MODEL=qwen   # auto-load qwen pipeline at startup
```

Set `DEFAULT_MODEL=` (empty) to start with nothing loaded.
