# Contracts: Three-Model POC

**Phase 1 output for `002-three-model-poc`**
**Date**: 2026-05-03

This document defines the external-facing interfaces exposed by this sprint. Internal
implementation details (config.pbtxt structure, Python wrapper internals) are in
[data-model.md](../data-model.md).

---

## 1. OpenAI-Compatible REST API (`/v1/chat/completions`)

**Endpoint**: `POST http://localhost:9000/v1/chat/completions`
**Served by**: `main.py` OpenAI frontend (via `triton_wrapper.py`)

### Text request (Chat / Code)

```json
{
  "model": "ensemble_qwen",
  "messages": [
    { "role": "user", "content": "Write a bubble sort in Rust." }
  ],
  "stream": false
}
```

**`model` field** must match the pipeline entry point name:

| Capability | `model` value |
|---|---|
| Code | `"ensemble_qwen"` |
| Chat | `"ensemble_llama"` |
| Caption | `"ensemble_vision"` |

> The `model` value is set by `capability.rs`'s `model_id()` method in the Rust client.
> Sprint 001 used `"ensemble"` for all three. This sprint updates them to the suffixed names.

### Vision request (Caption)

```json
{
  "model": "ensemble_vision",
  "messages": [
    {
      "role": "user",
      "content": [
        { "type": "text", "text": "Describe this image." },
        { "type": "image_url", "image_url": { "url": "data:image/jpeg;base64,<b64>" } }
      ]
    }
  ],
  "stream": false
}
```

> **Implementation note**: Whether `main.py`'s OpenAI frontend passes `image_url` content
> to the multimodal Triton pipeline must be verified at implementation time (see R6 in
> [research.md](research.md)). If not supported, the Rust client falls back to the Triton
> v2 native API (port 8000).

### Response format (unchanged from Sprint 001)

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "model": "ensemble_qwen",
  "choices": [
    {
      "index": 0,
      "message": { "role": "assistant", "content": "..." },
      "finish_reason": "stop"
    }
  ],
  "usage": { "prompt_tokens": N, "completion_tokens": M, "total_tokens": P }
}
```

---

## 2. Triton v2 Repository Control API

**Base URL**: `http://localhost:8000` (KServe HTTP, enabled via `--enable-kserve-frontends`)

### Load a model

```
POST /v2/repository/models/{model_name}/load
```

No request body required. Returns `200 OK` on success, `400` if load fails.

**Example** — load the full Llama pipeline:

```bash
curl -s -X POST localhost:8000/v2/repository/models/preprocessing_llama/load
curl -s -X POST localhost:8000/v2/repository/models/tensorrt_llm_llama/load
curl -s -X POST localhost:8000/v2/repository/models/postprocessing_llama/load
curl -s -X POST localhost:8000/v2/repository/models/tensorrt_llm_bls_llama/load
curl -s -X POST localhost:8000/v2/repository/models/ensemble_llama/load
```

### Unload a model

```
POST /v2/repository/models/{model_name}/unload
```

Returns `200 OK` on success.

**Example** — unload the full Llama pipeline (reverse dependency order):

```bash
curl -s -X POST localhost:8000/v2/repository/models/ensemble_llama/unload
curl -s -X POST localhost:8000/v2/repository/models/tensorrt_llm_bls_llama/unload
curl -s -X POST localhost:8000/v2/repository/models/postprocessing_llama/unload
curl -s -X POST localhost:8000/v2/repository/models/tensorrt_llm_llama/unload
curl -s -X POST localhost:8000/v2/repository/models/preprocessing_llama/unload
```

### Model ready check

```
GET /v2/models/{model_name}/ready
```

Returns `200` if ready, `404` if not loaded.

### Repository index (status)

```
POST /v2/repository/index
Content-Type: application/json

{"ready": false}
```

Returns JSON array of all known models with their state.

---

## 3. `just` Recipe CLI

The `just` task runner is the primary operator interface. Recipes must match this contract.

### Model control

```
just models-load <pipeline>
```
- `<pipeline>` ∈ `qwen` | `llama` | `vision`
- Loads all 5 (or 4 for vision) Triton models for the pipeline in dependency order.
- Exits 0 on success; prints load result for each model.
- Exits non-zero if any load fails (Triton returns non-200).

```
just models-unload <pipeline>
```
- Unloads all models in reverse dependency order.
- Exits 0 even if a model is already unloaded (idempotent).

```
just models-status
```
- Calls `POST /v2/repository/index` and displays all models grouped by pipeline with state.

### Engine build (run inside Triton 25.05 container)

```
just build-llama        # convert checkpoint + trtllm-build for Llama 3.1 8B
just build-llava        # convert LLaVA 1.5 7B weights + build_multimodal_engine.py
```

### Existing recipes (unchanged)

```
just compose-up         # start Triton container
just compose-down       # stop container
just compose-install    # copy compose.yaml + scripts to $TRTLLM_HOME
just smoke              # run Rust client smoke-test
```

---

## 4. Rust Client CLI

**Binary**: `trtllm-client`

```
cargo run -- smoke-test
```

Sends one request per capability (Code, Chat, Caption). Exits 0 if all succeed.
Caption failure is non-fatal (warning only) until vision pipeline is confirmed working.

**Environment variables** (unchanged from Sprint 001):

| Var | Default | Description |
|---|---|---|
| `TRTLLM_HOST` | `localhost` | Triton host |
| `TRTLLM_PORT` | `9000` | OpenAI frontend port |

**Logging** (unchanged):
- Logs model ID, token counts, and latency per request to stderr.
- Format: structured JSON to stderr.
