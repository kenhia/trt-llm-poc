# Setup Guide: TRT-LLM POC End-to-End

**Project**: TRT-LLM POC — RTX 5090  
**Last updated**: 2026-05-03

This guide covers the complete setup from a bare host to a running Triton inference
endpoint. Follow sections in order; each section's "Verify" step confirms you can
proceed to the next.

> For the fast path, see [quickstart.md](../specs/001-trt-llm-poc-end-to-end/quickstart.md).  
> For architecture decisions, see [docs/decisions/](decisions/).  
> For ansible-k promotion notes, see [docs/ansible-handoff.md](ansible-handoff.md).

---

## §0 — Host Prerequisites

These must be satisfied before any Docker work. They are assumed to be already present
on this machine but are listed here for reproducibility.

### §0.1 NVIDIA Driver

```bash
nvidia-smi
```

Expected: RTX 5090 listed, Driver Version ≥ 595.xx, CUDA Version ≥ 13.x.

**Why**: TRT-LLM NGC images require a matching CUDA runtime on the host. The driver
exposes the GPU to the container runtime via `/dev/nvidia*` devices.

If `nvidia-smi` fails: reinstall the NVIDIA driver from
`https://www.nvidia.com/Download/index.aspx` for your kernel version.

### §0.2 Docker + NVIDIA Container Toolkit

```bash
docker info | grep -i "server version"
docker info | grep -i "runtime\|nvidia"
```

Expected: Docker server version ≥ 24.x, `nvidia` listed under `Runtimes`.

**Why**: The NVIDIA Container Toolkit exposes GPU devices to Docker containers. Without
it, `--gpus all` fails with "failed to discover GPU vendor from CDI: no known GPU vendor
found" — even when `nvidia-smi` works fine on the host (the driver is separate from the
container bridge).

Install on Ubuntu 24.04 if `nvidia` is absent from `Runtimes` (record version in
`docs/ansible-handoff.md`):
```bash
# 1. Add signed repo
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 2. Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 3. Register the nvidia runtime with Docker and restart
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 4. Confirm
docker info | grep -i "runtime\|nvidia"
# Expected: "nvidia" listed under Runtimes
```

For other distros see:
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

### §0.3 Docker Compose V2

```bash
docker compose version
```

Expected: `Docker Compose version v2.x.x`.

**Why**: `compose.yaml` and `just compose-*` recipes require Compose V2 (`docker compose`,
not the legacy `docker-compose`). Compose V2 ships with Docker Desktop and as a plugin
for Docker Engine.

### §0.4 `just` Task Runner

```bash
just --version
```

Expected: `just x.y.z`.

Install if missing (record version in `docs/ansible-handoff.md`):
```bash
# Via cargo (preferred — Rust toolchain already required):
cargo install just

# Or via package manager (distro-dependent):
# brew install just  (macOS)
# apt install just   (Debian/Ubuntu ≥ 24.04)
```

### §0.5 Rust Stable Toolchain

```bash
cargo --version
rustc --version
```

Expected: `cargo 1.x.x` and `rustc 1.x.x (stable)`.

Install if missing (record version in `docs/ansible-handoff.md`):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

**Verify §0 complete**: all five commands above return expected output without errors.

---

## §1 — Clone and Initialize

### §1.1 Clone the Repo

```bash
git clone <repo-url> trt-llm-poc
cd trt-llm-poc
```

### §1.2 Set Up Environment

Copy the environment template to the **repo root** (where `just` picks it up via
`set dotenv-load := true`) and fill in your values:

```bash
cp .env.example .env
$EDITOR .env
```

Set both variables:
- `TRTLLM_HOME=/ai/trtllm-poc` — root for all TRT-LLM artifacts on this host (default; change if preferred)
- `HF_TOKEN=hf_...` — your HuggingFace access token (required for gated Llama models)

**Why `TRTLLM_HOME` is separate from the repo**: model weights and engine files are
gigabytes in size and must not be committed to git. `/ai/trtllm-poc` is the single
root that holds everything — models, engines, compose.yaml, model_repo — keeping the
git repo clean.

### §1.3 Create the Directory Layout

```bash
just init
```

This creates: `models/`, `engines/`, `cache/`, `logs/`, `model_repo/` under `$TRTLLM_HOME`.

Then copy `.env` to `$TRTLLM_HOME` so that `docker compose` can read it when
`just compose-*` recipes cd there:
```bash
cp .env $TRTLLM_HOME/
```

**Verify §1 complete**:
```bash
ls -A $TRTLLM_HOME/
# Expected: .env  cache  compose.yaml(after compose-install)  engines  logs  model_repo  models
```

---

## §2 — Pull NGC Images (Track A + Track B)

```bash
just pull
```

This pulls two images:
- **Track A** (`nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13`) — ~20 GB — for engine
  builds and interactive experimentation
- **Track B** (`nvcr.io/nvidia/tritonserver:25.05-trtllm-python-py3`) — ~15 GB — for
  HTTP serving

**Why these specific tags**: These tags are pinned because they ship coherent
CUDA/TensorRT/TRT-LLM stacks known to support Blackwell (SM 12.0). Using `latest`
risks pulling a build that predates SM 12.x support. See
[decisions/003-model-selection.md](decisions/003-model-selection.md) and R-001.

After pull, record the actual digests (for `docs/ansible-handoff.md`):
```bash
docker images --digests | grep -E "tensorrt-llm|tritonserver"
```

**Why record digests**: NGC tags can be re-pushed with different content. The digest is
the immutable identifier; recording it makes the environment reproducible.

**Verify §2 complete**:
```bash
docker images | grep -E "tensorrt-llm|tritonserver"
# Expected: both images listed with correct tags
```

---

## §3 — Track A: GPU Validation + Engine Build

Track A is the TRT-LLM dev container. Use it to confirm the GPU is visible, import the
library, download a model, and build a TRT-LLM engine. Engines built here are persisted
to `$TRTLLM_HOME/engines/` and reused by Track B (Triton).

### §3.1 Open the Dev Container

```bash
just dev
```

This runs:
```
docker run --rm -it --gpus all --ipc=host --ulimit memlock=-1 ...
  -v $TRTLLM_HOME:/workspace/trtllm
  nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13
```

**Why `--ipc=host`**: Large models use shared memory for inter-process tensor transfers.
Without host IPC, you may see random crashes or "out of shared memory" errors.
**Why `--ulimit memlock=-1`**: Required to allow the GPU driver to lock all GPU memory
pages, which TRT-LLM needs for engine loading.

### §3.2 Validate GPU Visibility (SC-001)

Inside the container:

```bash
nvidia-smi
```

Expected: RTX 5090 listed, `595.xx` driver, correct VRAM (32 GB). This confirms
SC-001: GPU visible within 30 seconds of container start.

If `nvidia-smi` fails inside the container: check that the NVIDIA Container Toolkit
runtime is configured (`docker info | grep -i nvidia` on the host).

### §3.3 Validate TRT-LLM Installation

```bash
python -c "import tensorrt_llm; print(tensorrt_llm.__version__)"
```

Expected output (warnings are normal, the version line is what matters):

```
Skipping import of cpp extensions due to incompatible torch version ...
...UserWarning: transformers version X.Y.Z is incompatible with nvidia-modelopt...
🚨 Config not found for parakeet. ...
[TensorRT-LLM] TensorRT LLM version: 1.3.0rc13
1.3.0rc13
```

The three warnings are harmless: `torchao` cpp extensions are optional quantization
optimizations; the `modelopt` warning only matters if using its HuggingFace integration
directly; the `parakeet` warnings are for a speech model config unrelated to LLM inference.
The critical signal is the `[TensorRT-LLM] TensorRT LLM version:` line — if that prints,
the library initialized successfully.

If this fails entirely (no version line): the image tag may not support SM 12.0. Check the
[TRT-LLM release notes](https://nvidia.github.io/TensorRT-LLM/) for SM 12.x support
and fall back to a later release tag.

### §3.4 Download a Model

> **Note on gated models**: `meta-llama/Meta-Llama-3.1-8B-Instruct` requires HuggingFace
> access approval (visit the model page to request). Approval can take hours to a week.
> `Qwen/Qwen2.5-Coder-7B-Instruct` is one of the three planned models and is fully open —
> use it for initial pipeline validation while Llama access is pending.

**Option A — Qwen2.5-Coder-7B-Instruct (ungated, recommended first)**:

```bash
# Inside the container:
cd /workspace/trtllm

huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct \
  --local-dir /workspace/trtllm/models/qwen-coder-7b \
  --local-dir-use-symlinks False
```

This will take several minutes (~14 GB download). The weights are stored in
`$TRTLLM_HOME/models/qwen-coder-7b/` on the host.

**Option B — Llama 3.1 8B Instruct (gated, requires HF access approval)**:

```bash
# Inside the container — HF_TOKEN is already set via just dev, so login is optional.
# If you need to verify or use a different token:
hf auth login --token $HF_TOKEN
# (Note: 'huggingface-cli login' is deprecated; use 'hf auth login' instead.)

huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct \
  --local-dir /workspace/trtllm/models/llama3-8b-instruct \
  --local-dir-use-symlinks False
```

**Why `--local-dir-use-symlinks False`**: Avoids HuggingFace's default symlink layout
that can confuse tools expecting flat file trees inside containers.

### §3.5 Convert to TRT-LLM Checkpoint Format

Before building an engine, TRT-LLM requires converting the HuggingFace weights to its
checkpoint format. The conversion script path depends on the model:

**For Qwen2.5-Coder-7B (Option A)**:
```bash
python /app/tensorrt_llm/examples/models/core/qwen/convert_checkpoint.py \
  --model_dir /workspace/trtllm/models/qwen-coder-7b \
  --output_dir /workspace/trtllm/models/qwen-coder-7b-ckpt \
  --dtype float16
```

**For Llama 3.1 8B (Option B)**:
```bash
python /app/tensorrt_llm/examples/models/core/llama/convert_checkpoint.py \
  --model_dir /workspace/trtllm/models/llama3-8b-instruct \
  --output_dir /workspace/trtllm/models/llama3-8b-instruct-ckpt \
  --dtype float16
```

**Why this step**: TRT-LLM's engine builder (`trtllm-build`) consumes its own checkpoint
format, not raw HuggingFace safetensors. The conversion script is bundled in the image.

> **Script path note**: In the NGC image the examples live under `/app/tensorrt_llm/examples/`.
> Model-specific converters are under `models/core/<arch>/convert_checkpoint.py`.
> The generic inference script is at `/app/tensorrt_llm/examples/run.py`.

### §3.6 Build the TRT-LLM Engine (SC-001, SC-002)

**For Qwen2.5-Coder-7B (Option A)**:
```bash
trtllm-build \
  --checkpoint_dir /workspace/trtllm/models/qwen-coder-7b-ckpt \
  --output_dir /workspace/trtllm/engines/qwen-coder \
  --gemm_plugin float16 \
  --max_batch_size 1 \
  --max_input_len 2048 \
  --max_seq_len 3072
```

**For Llama 3.1 8B (Option B)**:
```bash
trtllm-build \
  --checkpoint_dir /workspace/trtllm/models/llama3-8b-instruct-ckpt \
  --output_dir /workspace/trtllm/engines/llama3-chat \
  --gemm_plugin float16 \
  --max_batch_size 1 \
  --max_input_len 2048 \
  --max_seq_len 3072
```

Flag rationale:
- `--gemm_plugin float16`: enables the TRT-LLM GEMM plugin for float16, which is
  required for correct operation on most models
- `--max_batch_size 1`: minimal for POC; increase for throughput experiments
- `--max_input_len / --max_seq_len`: set conservatively for 32 GB; increase if needed

This takes several minutes. The built engine files land in
`$TRTLLM_HOME/engines/llama3-chat/` (persisted on the host).

### §3.7 Run an Inference Smoke Test (SC-002)

**For Qwen2.5-Coder-7B (Option A)**:
```bash
python /app/tensorrt_llm/examples/run.py \
  --engine_dir /workspace/trtllm/engines/qwen-coder \
  --tokenizer_dir /workspace/trtllm/models/qwen-coder-7b \
  --input_text "Write a Python function that returns the nth Fibonacci number." \
  --max_output_len 128
```

**For Llama 3.1 8B (Option B)**:
```bash
python /app/tensorrt_llm/examples/run.py \
  --engine_dir /workspace/trtllm/engines/llama3-chat \
  --tokenizer_dir /workspace/trtllm/models/llama3-8b-instruct \
  --input_text "What is the capital of France?" \
  --max_output_len 64
```

Expected: a coherent text response (e.g., "The capital of France is Paris.") within
60 seconds on first run (SC-002). Subsequent runs will be faster (engine stays loaded).

Note the wall-clock time for this first inference and record it — this is the SC-002
measurement.

**Verify §3 complete**:
- [ ] `nvidia-smi` showed RTX 5090 inside the container (SC-001)
- [ ] `tensorrt_llm.__version__` printed without errors
- [ ] Engine built successfully in `$TRTLLM_HOME/engines/llama3-chat/`
- [ ] Inference smoke test returned a coherent response within 60 seconds (SC-002)

Exit the container when done — engines persist on the host:
```bash
exit
```

---

## §4 — Track B: Triton Model Repository Setup

Track B uses the Triton Inference Server container (`tritonserver:25.05-trtllm-python-py3`)
to serve built engines via an OpenAI-compatible HTTP endpoint on port 9000.

> **Critical version constraint**: The Triton 25.05 image bundles TRT-LLM **0.19.0**, not
> the 1.3.0rc13 in the Track A dev container. Engines built in the dev container (Track A)
> are **incompatible** with Triton 25.05 and will fail to deserialize at load time. All
> engines intended for Triton must be built **inside the Triton container**.

> **Multi-model pipeline**: Triton 25.05 requires the full `inflight_batcher_llm` five-model
> pipeline layout (`preprocessing`, `tensorrt_llm`, `postprocessing`, `ensemble`,
> `tensorrt_llm_bls`). A single `config.pbtxt` is not sufficient. Use the templates bundled
> in the container and `fill_template.py` to generate the correct model repository.

### §4.1 Install the Compose File

```bash
# From repo root (on the host, not inside any container):
just compose-install
```

This copies `compose.yaml` to `$TRTLLM_HOME/compose.yaml`. The `compose-up` recipe
reads it from there.

### §4.2 Build the Engine Inside the Triton Container

The engine must be built inside the Triton 25.05 container so it targets TRT-LLM 0.19.0.

```bash
# 1. Start an interactive Triton container (Track B image, not the dev container):
docker run -it --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v $TRTLLM_HOME:/workspace/trtllm \
  nvcr.io/nvidia/tritonserver:25.05-trtllm-python-py3 \
  bash

# 2. Inside the container — convert HF weights to TRT-LLM checkpoint format:
#    (Qwen2.5-Coder-7B shown; adjust paths for other models)
python3 /opt/tritonserver/backends/tensorrtllm/examples/qwen/convert_checkpoint.py \
  --model_dir /workspace/trtllm/models/qwen-coder-7b \
  --output_dir /workspace/trtllm/models/qwen-coder-7b-ckpt-0.19 \
  --dtype float16

# 3. Build the TRT-LLM engine:
trtllm-build \
  --checkpoint_dir /workspace/trtllm/models/qwen-coder-7b-ckpt-0.19 \
  --output_dir /workspace/trtllm/engines/qwen-coder \
  --gemm_plugin float16 \
  --max_batch_size 1 \
  --max_input_len 2048 \
  --max_seq_len 3072

# 4. Exit the container (engines persist on the host via the volume mount):
exit
```

**If the engine directory is owned by root** (because it was written by root inside the
container), fix permissions on the host:
```bash
sudo chown -R $USER:$USER $TRTLLM_HOME/engines/
```

### §4.3 Set Up the Triton Model Repository (inflight_batcher_llm)

Triton 25.05 requires the full five-model pipeline. Generate it from the container's
bundled templates using `fill_template.py`:

```bash
# 1. Copy the pipeline templates out of the container image:
docker run --rm \
  nvcr.io/nvidia/tritonserver:25.05-trtllm-python-py3 \
  bash -c "tar -cf - /opt/tritonserver/backends/tensorrtllm/inflight_batcher_llm" \
  | tar -xf - --strip-components=6 -C $TRTLLM_HOME/model_repo

# 2. Run fill_template.py inside the container to substitute engine/tokenizer paths:
docker run --rm \
  -v $TRTLLM_HOME:/workspace/trtllm \
  nvcr.io/nvidia/tritonserver:25.05-trtllm-python-py3 \
  bash -c "
    cd /workspace/trtllm/model_repo/tensorrt_llm
    python3 /opt/tritonserver/backends/tensorrtllm/tools/fill_template.py \
      --in_place config.pbtxt \
      decoupled_mode:False,\
      engine_dir:/workspace/trtllm/engines/qwen-coder,\
      max_tokens_in_paged_kv_cache:2560,\
      batch_scheduler_policy:guaranteed_no_evict,\
      kv_cache_free_gpu_mem_fraction:0.9,\
      max_num_sequences:1,\
      executor_worker_path:/opt/tritonserver/backends/tensorrtllm/trtllmExecutorWorker
  "

# 3. Create the required version subdirectory for each model:
for model in preprocessing tensorrt_llm postprocessing ensemble tensorrt_llm_bls; do
  mkdir -p $TRTLLM_HOME/model_repo/$model/1
done
```

**Why five models**: The pipeline separates concerns — `preprocessing` tokenizes input,
`tensorrt_llm` runs the engine, `postprocessing` detokenizes output, `ensemble` chains
them, and `tensorrt_llm_bls` provides the Business Logic Script entry point for the
OpenAI frontend.

### §4.4 Verify the Model Repository

```bash
ls $TRTLLM_HOME/model_repo/
# Expected: ensemble/  postprocessing/  preprocessing/  tensorrt_llm/  tensorrt_llm_bls/

# Each model dir must have a version subdirectory:
ls $TRTLLM_HOME/model_repo/tensorrt_llm/
# Expected: 1/  config.pbtxt

# Confirm engine path is correct in the tensorrt_llm config:
grep gpt_model_path $TRTLLM_HOME/model_repo/tensorrt_llm/config.pbtxt
# Expected: string_value: "/workspace/trtllm/engines/qwen-coder"
```

**Verify §4 complete**:
- [ ] Engine built inside Triton 25.05 container and owned by non-root user
- [ ] `$TRTLLM_HOME/model_repo/` has all five pipeline model directories
- [ ] Each model directory has a `1/` version subdirectory
- [ ] `grep gpt_model_path` shows the correct engine path

---

## §5 — Track B: Start Triton and Verify the Endpoint

### §5.1 Start Triton

```bash
just compose-up
```

This runs `docker compose up -d` from `$TRTLLM_HOME`. The container starts detached with
`restart: unless-stopped` — it will come back automatically after a host reboot (SC-003).

The container runs `scripts/start-triton.sh`, which execs `python3 openai_frontend/main.py`.
`main.py` starts an **embedded Triton** (via Python bindings), loads the model repository,
and starts Uvicorn on port 9000. There is no separate `tritonserver` process — they are
combined.

> **Why embedded Triton**: the Triton 25.05 Python API (`tritonserver` bindings) embeds the
> Triton runtime directly in the Python process. Starting a standalone `tritonserver` binary
> alongside `main.py` causes GPU resource contention and `Failed to deserialize cuda engine`
> failures. Always use `main.py` alone as the entrypoint.

### §5.2 Watch Startup Logs

```bash
just compose-logs
```

Watch for these key messages in order:

1. `[start-triton] Starting OpenAI frontend` — startup script running
2. `Loading model...` — embedded Triton is loading the TRT-LLM engine
3. `Loaded engine size: XXXX MiB` — engine loaded into GPU memory
4. `All models are ready` — all five pipeline models ready
5. `Uvicorn running on http://0.0.0.0:9000` — OpenAI frontend accepting requests

Full startup typically takes 20–60 seconds. Press `Ctrl-C` to stop following logs; the
container continues running.

**If Triton exits immediately**: run `just compose-logs` to see the error. Common causes:
- Missing or invalid `config.pbtxt` (name mismatch, bad path)
- Engine directory doesn't exist or is empty
- `Failed to deserialize cuda engine` — engine was built with wrong TRT-LLM version
  (see §4.2 — engine must be built inside the Triton container)
- Insufficient VRAM (another process holding GPU memory)

See [docs/setup.md §6](#6--troubleshooting) for more.

### §5.3 Health Check — Triton Native API

```bash
# Triton native readiness (should return HTTP 200):
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/v2/health/ready

# List loaded models via native v2 API:
curl -s -X POST http://localhost:8000/v2/repository/index | python -m json.tool
# Expected: all five pipeline models listed with state READY
```

### §5.4 Health Check — OpenAI-Compatible API (SC-002 prerequisite)

```bash
# List models via OpenAI API (port 9000):
curl -s http://localhost:9000/v1/models | python -m json.tool
# Expected: {"object":"list","data":[{"id":"ensemble",...},{"id":"tensorrt_llm_bls",...},...]}
```

### §5.5 Inference Smoke Test via curl

```bash
curl -s http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ensemble",
    "messages": [{"role": "user", "content": "Write a one-line Python hello world."}],
    "max_tokens": 64
  }' | python -m json.tool
```

Expected: a JSON response with `choices[0].message.content` containing Python code.

> **Model name**: Use `"ensemble"` (or `"tensorrt_llm_bls"`) in all requests — these are
> the pipeline entry-point model names that Triton has loaded. Individual component models
> (`preprocessing`, `tensorrt_llm`, `postprocessing`) are not valid inference targets.

**Verify §5 complete**:
- [ ] `just compose-up` started the container without error
- [ ] `just compose-logs` showed `All models are ready` then `Uvicorn running on ... 9000`
- [ ] `curl localhost:8000/v2/health/ready` returned `200`
- [ ] `curl localhost:9000/v1/models` returned model list (SC-002 prerequisite met)
- [ ] `curl localhost:9000/v1/chat/completions` returned a coherent code response
- [ ] Rebooted host and confirmed container restarted automatically (SC-003)

---

*Continue in §6 (Rust client + troubleshooting) — added during T018/T019.*

---

## §6 — Troubleshooting

This section covers every failure mode encountered during the POC. Each entry gives the
symptom, root cause, and resolution.

### §6.1 Docker + GPU

| Symptom | Cause | Fix |
|---------|-------|-----|
| `just dev` fails: "failed to discover GPU vendor from CDI: no known GPU vendor found" | NVIDIA Container Toolkit not installed or CDI spec not generated | Install toolkit (§0.2), reboot if kernel upgrade pending, then `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker` |
| `docker info` shows no `nvidia` under Runtimes | `nvidia-ctk runtime configure` not run | `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker` |
| `nvidia-cdi-refresh.service` fails at boot with `result=9 NVML_ERROR_DRIVER_NOT_LOADED` | Kernel upgrade installed but not yet booted into | Reboot; the service auto-runs on next boot with the correct kernel module |
| `nvidia-smi` works on host but fails inside container | Container runtime not configured for nvidia | See above — toolkit + runtime configure + restart |
| `apt-get update` hangs during toolkit install | Flaky apt repo (observed: `ppa:neovim-ppa/unstable`) | Remove the flaky PPA: `sudo add-apt-repository --remove ppa:neovim-ppa/unstable` then retry |

### §6.2 TRT-LLM Track A (dev container)

| Symptom | Cause | Fix |
|---------|-------|-----|
| `python -c "import tensorrt_llm"` prints warnings but no version | Normal — warnings about torchao, modelopt, parakeet are benign | Check the last line: `[TensorRT-LLM] TensorRT LLM version: ...` — if present, you're good |
| `huggingface-cli download` returns 403 / "access restricted" | Gated model (Llama) requires HF access approval | Request access at the model page; use `Qwen/Qwen2.5-Coder-7B-Instruct` (ungated) while waiting |
| `huggingface-cli login` shows deprecation warning | CLI renamed | Use `hf auth login --token $HF_TOKEN` instead; or skip — `$HF_TOKEN` env var is already set by `just dev` |
| `convert_checkpoint.py: No such file or directory` | Wrong path — examples are in `/app/tensorrt_llm/examples/`, not `/usr/local/lib/` | Use `/app/tensorrt_llm/examples/models/core/<arch>/convert_checkpoint.py` |
| `run.py: error: the following arguments are required: --max_output_len` | `--max_output_len` is required | Add `--max_output_len 128` (or any positive integer) to the command |
| `run.py` emits FutureWarning about "legacy TensorRT engine-build workflow" | `ModelRunnerCpp` is deprecated in 1.3.0rc13 | Normal for POC — does not affect correctness; note for future migration to `trtllm-serve` |
| Engine build OOM | Insufficient free VRAM | Ensure no other GPU processes: `nvidia-smi`; reduce `--max_batch_size`; restart container |
| Engine build very slow | Normal — first build compiles TRT kernels | Allow 10–30 min on first build; subsequent builds of the same config are cached |

### §6.3 Triton Track B (serving)

| Symptom | Cause | Fix |
|---------|-------|-----|
| `just compose-up` exits immediately | `TRTLLM_HOME/.env` missing or malformed | Check `cat $TRTLLM_HOME/.env`; ensure `TRTLLM_HOME` and `HF_TOKEN` are set |
| Triton container exits on startup | Missing or invalid `config.pbtxt`, or engine dir doesn't exist | `just compose-logs` — look for "failed to load model" or path errors; verify engine_dir exists |
| `config.pbtxt` error: "name mismatch" | `name:` in config.pbtxt doesn't match the directory name | Directory name must exactly match the `name:` field |
| `curl /v1/models` returns `Connection refused` | Triton not running | `just ps` to check container status; `just compose-up` if stopped |
| `curl /v1/models` returns `{}` or empty data | No models loaded | Check config.pbtxt engine_dir points to a built engine; `just compose-logs` for load errors |
| `curl /v1/chat/completions` returns 503 | Model not ready yet | Wait 30–60s after startup; watch `just compose-logs` for "All models are ready" |
| `curl /v1/chat/completions` returns 400 | Bad request body — wrong model ID or malformed JSON | Verify `model` value matches `name:` in config.pbtxt exactly |

### §6.4 Rust Client

| Symptom | Cause | Fix |
|---------|-------|-----|
| `cargo build` fails: `cannot find module rust_client` | Tests import from lib crate which requires `src/lib.rs` | Ensure `src/lib.rs` exists and declares all modules with `pub mod` |
| `cargo run -- smoke-test` fails: "Startup probe failed" | Triton not running or port wrong | Confirm `just ps` shows container up; check `TRTLLM_HOST`/`TRTLLM_PORT` env vars |
| `cargo run -- smoke-test` fails on Caption only | LLaVA model not loaded / multi-modal pipeline not configured | Expected — LLaVA requires separate visual encoder engine; see decisions/003-model-selection.md fallback plan |
| JSON log output garbled | `RUST_LOG` set to unexpected value | Unset or set `RUST_LOG=info` |
| Routing tests fail | Routing logic mismatch | Check `resolve_capability` in `capability.rs` matches data-model.md rules |
