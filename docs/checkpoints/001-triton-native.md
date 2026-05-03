# Checkpoint: Triton Native Protocol Working

**Date**: 2026-05-03  
**Tag**: `triton-protocol`  
**Branch**: `001-trt-llm-poc-end-to-end`

## What Works

Track B is fully operational. The Qwen2.5-Coder-7B-Instruct engine is served by Triton
Inference Server and responding to inference requests over HTTP:

```bash
curl -s -X POST http://localhost:8000/v2/models/ensemble/generate \
  -H "Content-Type: application/json" \
  -d '{"text_input": "Write a one-line Python hello world.",
       "max_tokens": 64, "bad_words": "", "stop_words": ""}' \
  | python3 -m json.tool
```

Response:
```json
{
    "model_name": "ensemble",
    "model_version": "1",
    "text_output": "..."
}
```

Triton readiness: `GET /v2/health/ready` returns HTTP 200.  
Model list: `POST /v2/repository/index` returns all 5 models READY.

## Acceptance Criteria Status

| SC | Description | Status |
|----|-------------|--------|
| SC-001 | GPU visible in Docker container | ✅ Verified (RTX 5090, 32 GB, driver 595.58.03) |
| SC-002 | First inference ≤ 60 s | ✅ Verified (~5 s on first request) |
| SC-003 | Triton auto-restarts on reboot | ✅ `restart: unless-stopped` in compose.yaml |
| SC-005 | `cargo test` passes | ✅ 7/7 routing tests pass |

SC-004 (all three capabilities) and LLaVA caption remain pending — LLaVA engine not built.

## Key Lessons Learned

### 1 — Engine version must match Triton's bundled TRT-LLM

TRT-LLM engine files are not cross-version compatible. The dev container
(`tensorrt-llm/release:1.3.0rc13`) ships TRT-LLM 1.3.0rc13; Triton
(`tritonserver:25.05-trtllm-python-py3`) ships TRT-LLM **0.19.0**. An engine built
in the dev container will fail with `Failed to deserialize cuda engine` when loaded
by Triton.

**Fix**: Always build the serving engine inside the Triton container, not the dev
container:

```bash
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 \
  -v /ai/trtllm-poc:/workspace/trtllm \
  nvcr.io/nvidia/tritonserver:25.05-trtllm-python-py3 \
  bash -c "
python3 /app/examples/qwen/convert_checkpoint.py \
  --model_dir /workspace/trtllm/models/qwen-coder-7b \
  --output_dir /workspace/trtllm/models/qwen-coder-7b-ckpt \
  --dtype float16 && \
trtllm-build \
  --checkpoint_dir /workspace/trtllm/models/qwen-coder-7b-ckpt \
  --output_dir /workspace/trtllm/engines/qwen-coder \
  --gemm_plugin float16 \
  --max_batch_size 1 \
  --max_input_len 2048 \
  --max_seq_len 3072
"
```

### 2 — Triton 25.05 requires the multi-model pipeline layout

A single `config.pbtxt` pointing at an engine is not sufficient. Triton 25.05 requires
the full `inflight_batcher_llm` layout from the container:

```
model_repo/
├── preprocessing/    # tokenizer: text → token IDs
├── tensorrt_llm/     # engine inference
├── postprocessing/   # detokenizer: token IDs → text
├── ensemble/         # chains the three above
└── tensorrt_llm_bls/ # alternative BLS chain
```

Bootstrap from inside the Triton container (one-off, paths relative to container):

```bash
docker run --rm \
  -v /ai/trtllm-poc:/workspace/trtllm \
  --entrypoint bash \
  nvcr.io/nvidia/tritonserver:25.05-trtllm-python-py3 -c "
cp -r /app/all_models/inflight_batcher_llm/* /workspace/trtllm/model_repo/ && \
python3 /app/tools/fill_template.py -i /workspace/trtllm/model_repo/tensorrt_llm/config.pbtxt \
  triton_backend:tensorrtllm,triton_max_batch_size:1,decoupled_mode:false,\
engine_dir:/workspace/trtllm/engines/qwen-coder,max_queue_delay_microseconds:0,\
batching_strategy:inflight_fused_batching,max_queue_size:0,\
encoder_input_features_data_type:TYPE_FP16,logits_datatype:TYPE_FP32,\
guided_decoding_backend:,xgrammar_tokenizer_info_path: && \
python3 /app/tools/fill_template.py -i /workspace/trtllm/model_repo/preprocessing/config.pbtxt \
  tokenizer_dir:/workspace/trtllm/models/qwen-coder-7b,triton_max_batch_size:1,preprocessing_instance_count:1 && \
python3 /app/tools/fill_template.py -i /workspace/trtllm/model_repo/postprocessing/config.pbtxt \
  tokenizer_dir:/workspace/trtllm/models/qwen-coder-7b,triton_max_batch_size:1,\
postprocessing_instance_count:1,skip_special_tokens:true && \
python3 /app/tools/fill_template.py -i /workspace/trtllm/model_repo/ensemble/config.pbtxt \
  triton_max_batch_size:1,logits_datatype:TYPE_FP32 && \
python3 /app/tools/fill_template.py -i /workspace/trtllm/model_repo/tensorrt_llm_bls/config.pbtxt \
  triton_max_batch_size:1,decoupled_mode:false,bls_instance_count:1,logits_datatype:TYPE_FP32
"
```

### 3 — Triton native API, not OpenAI-compat

The `inflight_batcher_llm` layout exposes Triton's own generate endpoint, not
`/v1/chat/completions`:

| Purpose | Endpoint |
|---------|----------|
| Readiness check | `GET /v2/health/ready` |
| List models | `POST /v2/repository/index` |
| Inference | `POST /v2/models/ensemble/generate` |
| Inference (BLS) | `POST /v2/models/tensorrt_llm_bls/generate` |

Request shape:
```json
{"text_input": "...", "max_tokens": 64, "bad_words": "", "stop_words": ""}
```

OpenAI-compat (`/v1/chat/completions`) is not available at this checkpoint — that is
the goal of the next step (see §4 below).

### 4 — NVIDIA Container Toolkit was not installed despite daemon.json referencing it

`daemon.json` had `"path": "nvidia-container-runtime"` configured from a prior
`nvidia-ctk runtime configure` run, but the actual package had never been installed.
Error: `exec: "nvidia-container-runtime": executable file not found in $PATH`.

Fix: `sudo apt-get install -y nvidia-container-toolkit` (repo was already configured),
then `sudo systemctl restart docker`.

### 5 — compose.yaml had a placeholder command

The initial `compose.yaml` used `command: ["bash", "-lc", "sleep infinity"]` as a
placeholder. Updated to `command: ["tritonserver", "--model-repository=/model_repo"]`.

## File Layout at This Checkpoint

```
/ai/trtllm-poc/
├── .env                          # TRTLLM_HOME, HF_TOKEN
├── compose.yaml                  # tritonserver --model-repository=/model_repo
├── engines/qwen-coder/           # TRT-LLM engine (built with TRT-LLM 0.19.0)
├── models/
│   ├── qwen-coder-7b/            # HuggingFace weights
│   ├── qwen-coder-7b-ckpt/       # TRT-LLM 0.19.0 checkpoint (from Triton container)
│   ├── qwen-coder-7b-ckpt-0.19/  # same (name used during rebuild)
│   └── llama3-8b-instruct/       # HF weights (engine not yet built)
└── model_repo/
    ├── preprocessing/
    ├── tensorrt_llm/
    ├── postprocessing/
    ├── ensemble/
    └── tensorrt_llm_bls/
```

## What's Next

Add OpenAI-compat frontend so the Rust client can use `/v1/chat/completions`.
Options:

1. **Triton OpenAI server** (`/opt/tritonserver/python/openai/`) — ships in the 25.05
   image, wraps the Triton backend with an OpenAI-compatible HTTP layer
2. **Thin proxy** — a small service that translates `/v1/chat/completions` →
   `/v2/models/ensemble/generate`

Option 1 is the path of least resistance and is already in the container.
