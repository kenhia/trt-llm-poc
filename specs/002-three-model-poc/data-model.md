# Data Model: Three-Model POC

**Phase 1 output for `002-three-model-poc`**
**Date**: 2026-05-03

This document describes the configuration entities, directory structures, and state
machines that govern the three-model serving setup.

---

## Entities

### Pipeline

A named set of five Triton model directories backed by a single compiled TRT-LLM engine.

| Field | Type | Description |
|---|---|---|
| `name` | string | Short identifier: `qwen`, `llama`, `vision` |
| `entry_point` | string | Triton model name for inference: `ensemble_<name>` |
| `models` | list[string] | All 5 Triton model directory names with suffix |
| `engine_dir` | path | Absolute path to compiled `.engine` file(s) |
| `model_weights` | path | HuggingFace model source dir |
| `vram_fp16_gb` | float | Approximate VRAM usage when loaded (FP16) |
| `tokenizer` | path | Path to HF tokenizer for OpenAI frontend |

**Known pipeline instances**:

| `name` | `entry_point` | `vram_fp16_gb` | `engine_dir` |
|---|---|---|---|
| `qwen` | `ensemble_qwen` | ≈14 | `/ai/trtllm-poc/engines/qwen-coder/` |
| `llama` | `ensemble_llama` | ≈16 | `/ai/trtllm-poc/engines/llama/` |
| `vision` | `ensemble_vision` | ≈15 | `/ai/trtllm-poc/engines/llava/llm/` + `vision/` |

---

### TritonModel

A single model directory inside `model_repo/` watched by Triton.

| Field | Type | Description |
|---|---|---|
| `dir_name` | string | Directory name, e.g., `preprocessing_qwen` |
| `name` | string | Must match `name:` field in `config.pbtxt` |
| `pipeline` | string | Which pipeline this belongs to |
| `backend` | string | `"tensorrtllm"`, `"python"`, or `"ensemble"` |
| `state` | enum | `READY` \| `UNAVAILABLE` \| `LOADING` \| `UNLOADING` |

**Standard naming rule**: `{component}_{pipeline_name}` where `component` ∈
`{preprocessing, tensorrt_llm, postprocessing, ensemble, tensorrt_llm_bls, multimodal_encoders}`.

---

### Config Cross-References (per pipeline)

Dependencies encoded in `config.pbtxt` that must be updated when renaming:

```
ensemble_<X>/config.pbtxt
  ensemble_scheduling.step[0].model_name  → "preprocessing_<X>"
  ensemble_scheduling.step[1].model_name  → "tensorrt_llm_<X>"
  ensemble_scheduling.step[2].model_name  → "postprocessing_<X>"

tensorrt_llm_bls_<X>/config.pbtxt
  parameters.tensorrt_llm_model_name.string_value  → "tensorrt_llm_<X>"

tensorrt_llm_bls_<X>/1/model.py  (default arg values, must match config)
  preproc_model_name   → "preprocessing_<X>"
  postproc_model_name  → "postprocessing_<X>"
  llm_model_name       → "tensorrt_llm_<X>"

All *_<X>/config.pbtxt
  name:  → "<component>_<X>"
```

---

### Engine

A compiled TRT-LLM binary artifact.

| Field | Type | Description |
|---|---|---|
| `model_arch` | string | e.g., `LlamaForCausalLM`, `QWenForCausalLM`, `LlavaForCausalLM` |
| `trtllm_version` | string | Must be `0.19.0` (pinned by Triton 25.05) |
| `dtype` | string | `float16` (FP8 is a stretch goal) |
| `max_batch_size` | int | Build-time limit |
| `max_input_len` | int | Build-time limit |
| `max_seq_len` | int | Build-time limit |
| `files` | list[path] | `.engine` file + `config.json` |

---

### Environment Configuration

Env vars controlling the serving container at runtime:

| Variable | Default | Description |
|---|---|---|
| `TRITON_MODEL_REPO` | `/model_repo` | Path to model repository inside container |
| `OPENAI_TOKENIZER` | `/workspace/trtllm/models/qwen-coder-7b` | HF tokenizer path |
| `OPENAI_PORT` | `9000` | OpenAI-compat frontend port |
| `TRITON_MODEL_CONTROL_MODE` | `explicit` | `none` \| `explicit` (injected by wrapper) |
| `TRITON_STARTUP_MODELS` | `""` | Comma-separated model names to load at startup |
| `DEFAULT_MODEL` | `qwen` | Pipeline to auto-load after server is ready (if set) |
| `KSERVE_HTTP_PORT` | `8000` | KServe v2 API port (load/unload control plane) |

---

### Rust Capability Routing (post-sprint target state)

| Capability | `model_id()` | Source |
|---|---|---|
| `Code` | `ensemble_qwen` | Unchanged from Sprint 001 |
| `Chat` | `ensemble_llama` | Updated in this sprint |
| `Caption` | `ensemble_vision` | Updated in this sprint |

---

## Model Repository Directory Layout

```text
$TRTLLM_HOME/model_repo/
├── preprocessing_qwen/
│   ├── config.pbtxt          # name: "preprocessing_qwen"
│   └── 1/model.py
├── tensorrt_llm_qwen/
│   ├── config.pbtxt          # name: "tensorrt_llm_qwen"
│   └── 1/model.py
├── postprocessing_qwen/
│   ├── config.pbtxt          # name: "postprocessing_qwen"
│   └── 1/model.py
├── ensemble_qwen/
│   ├── config.pbtxt          # name: "ensemble_qwen"; refs *_qwen
│   └── 1/ (empty)
├── tensorrt_llm_bls_qwen/
│   ├── config.pbtxt          # name: "tensorrt_llm_bls_qwen"; param → tensorrt_llm_qwen
│   └── 1/
│       ├── model.py          # preproc/postproc/llm names updated
│       └── lib/
├── preprocessing_llama/      # (same structure, _llama suffix)
├── tensorrt_llm_llama/
├── postprocessing_llama/
├── ensemble_llama/
├── tensorrt_llm_bls_llama/
├── multimodal_encoders/      # vision: Python backend, NO suffix — bare name by convention
├── tensorrt_llm_vision/
├── postprocessing_vision/
└── ensemble_vision/          # refs multimodal_encoders (unsuffixed), tensorrt_llm_vision, postprocessing_vision
```

> Note: LLaVA 1.5 uses the `multimodal` pipeline template (not `inflight_batcher_llm`).
> The vision pipeline does not have a `tensorrt_llm_bls` component.

---

## Pipeline State Machine

```
          start (EXPLICIT mode)
               │
               ▼
         ┌──────────┐
         │  KNOWN   │  ← model exists in repo, not loaded
         └────┬─────┘
              │  POST /v2/repository/models/{name}/load
              ▼
         ┌──────────┐
         │ LOADING  │
         └────┬─────┘
              │  success
              ▼
         ┌──────────┐
         │  READY   │  ◄─── inference requests served
         └────┬─────┘
              │  POST /v2/repository/models/{name}/unload
              ▼
         ┌──────────┐
         │UNLOADING │
         └────┬─────┘
              │  complete
              ▼
         ┌──────────┐
         │  KNOWN   │
         └──────────┘
```

VRAM is occupied only in `READY` state. Transition to `LOADING` blocks until VRAM is
allocated. Load failure transitions back to `KNOWN`.

---

## VRAM Budget

With explicit mode (one pipeline active at a time, RTX 5090 32 GB):

| State | Active Pipeline | VRAM Used | Headroom |
|---|---|---|---|
| Idle | none | ~1 GB (OS + container) | ~31 GB |
| Qwen loaded | qwen | ~15 GB | ~17 GB |
| Llama loaded | llama | ~17 GB | ~15 GB |
| Vision loaded | vision | ~16 GB | ~16 GB |

Simultaneous load of two text pipelines (qwen + llama ≈ 31 GB) is theoretically possible
but leaves < 1 GB headroom — avoid in practice.
