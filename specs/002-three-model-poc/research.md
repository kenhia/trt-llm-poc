# Research: Three-Model POC

**Phase 0 output for `002-three-model-poc`**
**Date**: 2026-05-03

---

## R1 — Explicit Model-Control Mode in `main.py`

**Question**: Does `main.py`'s embedded `tritonserver.Server()` support `model_control_mode=EXPLICIT`,
and is it exposed as a CLI arg?

**Findings**:
- `tritonserver.ModelControlMode.EXPLICIT` **exists** (enum value confirmed via `python3 -c 'import
  tritonserver; print(dir(tritonserver.ModelControlMode))'`).
- `tritonserver.Options` signature: `model_control_mode: ModelControlMode = NONE`,
  `startup_models: list[str] = []`.
- `main.py` at `/opt/tritonserver/python/openai/openai_frontend/main.py` constructs
  `tritonserver.Server()` with only `model_repository`, `log_verbose`, `log_info`, `log_warn`,
  `log_error` — **no `model_control_mode` or `startup_models`**.
- `main.py`'s `parse_args()` has no `--model-control-mode` flag.

**Decision**: Create `scripts/triton_wrapper.py` that monkey-patches `tritonserver.Server` to
inject `model_control_mode=EXPLICIT` and `startup_models` before delegating to `main.main()`.
`start-triton.sh` execs the wrapper instead of `main.py` directly.

**Rationale**: Wrapper avoids copying `main.py` (which would diverge from container version on
image updates). Environment variables `TRITON_MODEL_CONTROL_MODE` and `TRITON_STARTUP_MODELS`
control behaviour.

**Alternatives rejected**:
- Fork/copy `main.py` — copied file diverges from container on image updates
- Stand-alone `tritonserver` binary — causes GPU conflict (confirmed broken in Sprint 001)
- POLL mode — scans model_repo on timer; requires file-system tricks to load/unload

---

## R2 — Multi-Pipeline Naming and Cross-References

**Question**: Can multiple `inflight_batcher_llm` pipelines coexist in a single `model_repo/`?
What config cross-references need updating when renaming models?

**Findings** (from `/ai/trtllm-poc/model_repo/` grep):

1. `ensemble_<model>/config.pbtxt` — `ensemble_scheduling` hardcodes three `model_name:` values:
   `"preprocessing"`, `"tensorrt_llm"`, `"postprocessing"`.
2. `tensorrt_llm_bls_<model>/config.pbtxt` — exposes `tensorrt_llm_model_name` as a template
   param; defaults to `"${tensorrt_llm_model_name}"`.
3. `tensorrt_llm_bls_<model>/1/model.py` — uses `preproc_model_name="preprocessing"` and
   `postproc_model_name="postprocessing"` as hardcoded defaults (but they can be overridden via
   config params).
4. `preprocessing`, `tensorrt_llm`, `postprocessing` — do NOT reference other models by name.

**Decision**: Suffix `_<model>` on all five pipeline model directories per pipeline:

| Standard name | Qwen | Llama | Vision |
|---|---|---|---|
| `preprocessing` | `preprocessing_qwen` | `preprocessing_llama` | `preprocessing_vision` |
| `tensorrt_llm` | `tensorrt_llm_qwen` | `tensorrt_llm_llama` | `tensorrt_llm_vision` |
| `postprocessing` | `postprocessing_qwen` | `postprocessing_llama` | `postprocessing_vision` |
| `ensemble` | `ensemble_qwen` | `ensemble_llama` | `ensemble_vision` |
| `tensorrt_llm_bls` | `tensorrt_llm_bls_qwen` | `tensorrt_llm_bls_llama` | `tensorrt_llm_bls_vision` |

All five pipelines live in the same `model_repo/`. Triton EXPLICIT mode uses `startup_models`
to load only one pipeline set at a time.

**Config updates required per pipeline**:
- Each `config.pbtxt` → `name:` field must match directory name.
- `ensemble_<model>/config.pbtxt` → update three `model_name:` refs in `ensemble_scheduling`.
- `tensorrt_llm_bls_<model>/config.pbtxt` → update `tensorrt_llm_model_name` default value.
- `tensorrt_llm_bls_<model>/1/model.py` → update `preproc_model_name` and `postproc_model_name`
  default args (or rely on config param override).

**Rationale**: Single repo is simpler than managing multiple bind-mount paths. EXPLICIT mode
with `startup_models` prevents simultaneous loading.

**Alternatives rejected**:
- Multiple model_repo directories with standard names — Triton errors on name conflicts when all
  repos are mounted simultaneously (server needs all repos visible to support load/unload).
- `model_namespacing=True` — adds namespacing complexity to all API calls; not worth it for POC.

---

## R3 — Triton v2 Repository Load/Unload API

**Question**: What are the exact HTTP endpoints for loading and unloading a model at runtime?

**Findings** (Triton KServe HTTP API, standard):
- Load: `POST /v2/repository/models/{model_name}/load`
- Unload: `POST /v2/repository/models/{model_name}/unload`
- Model ready: `GET /v2/models/{model_name}/ready`
- Model index: `POST /v2/repository/index` with `{"ready": false}` → lists all models + state

**Requirement**: KServe HTTP frontend must be active — `--enable-kserve-frontends` flag to
`main.py` (or `triton_wrapper.py`). This exposes port 8000 (KServe HTTP, default).

**`just` recipe behaviour**:
- `just models-load qwen` → posts load to all 5 qwen pipeline models (in dependency order: preprocessing, tensorrt_llm, postprocessing, tensorrt_llm_bls, ensemble)
- `just models-unload qwen` → posts unload to all 5 (reverse order: ensemble, tensorrt_llm_bls, ...)
- `just models-status` → `POST /v2/repository/index` + formatted output

**Rationale**: KServe v2 API is the standard Triton control plane; `curl` can drive it from
`just` recipes without any additional tooling.

---

## R4 — Vision Model Selection for TRT-LLM 0.19.0

**Question**: Does LLaVA-1.6 (LLaVA-NeXT) work with TRT-LLM 0.19.0 Triton serving?
If not, what is the best supported alternative?

**Findings** (from `/app/examples/multimodal/README.md` inside the Triton 25.05 container):

| Model | C++ Runtime Support | Notes |
|---|---|---|
| **LLaVA 1.5 7B** | `[x]` **SUPPORTED** | Full C++ runtime support |
| LLaVA-NeXT (1.6) | `[ ]` NOT SUPPORTED | "Model requires post processing its encoder output features, which is not supported" (footnote [^2]) |
| LLaVA-OneVision | `[ ]` NOT SUPPORTED | Same footnote [^2] |
| VILA | `[x]` SUPPORTED | Single image per request only |
| Qwen2-VL | `[ ]` NOT SUPPORTED in C++ | Python runtime only |

The `[ ]` in the README specifically marks the **C++ runtime** (`cpp` mode in `run.py`).
LLaVA-NeXT has build instructions but cannot serve via the Triton `inflight_batcher_llm`
pipeline (which uses C++ backend).

**Decision**: **LLaVA 1.5 7B** — `llava-hf/llava-1.5-7b-hf` (HuggingFace).

**Rationale**: Only fully `[x]`-supported vision model that is POC-sized (7B ≈ 14 GB VRAM FP16)
and fits cleanly in the `multimodal` Triton pipeline template at
`/app/all_models/multimodal/` (separate from `inflight_batcher_llm`). LLaVA-NeXT fails at the
post-processing step in C++ mode.

**Alternatives rejected**:
- LLaVA-NeXT Mistral-7B — `[ ]` not supported in C++ runtime for Triton serving
- Qwen2-VL — `[ ]` not supported in C++ runtime
- VILA — supported but documented as single-image-only; LLaVA 1.5 is cleaner

**Build path** (inside Triton 25.05 container):
```bash
# Download weights
git clone https://huggingface.co/llava-hf/llava-1.5-7b-hf /ai/trtllm-poc/models/llava-1.5-7b

# Build vision encoder TRT engine
python3 /app/examples/multimodal/build_multimodal_engine.py \
  --model_path /ai/trtllm-poc/models/llava-1.5-7b \
  --output_dir /ai/trtllm-poc/engines/llava/vision \
  --model_type llava

# Build LLM TRT engine (LLaVA 1.5 uses LLaMA-7B backbone)
python3 /app/examples/llama/convert_checkpoint.py \
  --model_dir /ai/trtllm-poc/models/llava-1.5-7b \
  --output_dir /ai/trtllm-poc/engines/llava/ckpt \
  --dtype float16

trtllm-build \
  --checkpoint_dir /ai/trtllm-poc/engines/llava/ckpt \
  --output_dir /ai/trtllm-poc/engines/llava/llm \
  --gemm_plugin float16 \
  --max_multimodal_len 2048
```

**Triton pipeline**: Uses `/app/all_models/multimodal/` template (not `inflight_batcher_llm`):
`multimodal_encoders` (Python backend) + `tensorrt_llm` + `preprocessing` + `postprocessing` + `ensemble`.
Rename with `_vision` suffix for coexistence.

---

## R5 — Llama 3.1 8B Engine Build

**Question**: What is the build procedure for Llama 3.1 8B inside the Triton 25.05 container?

**Findings**:
- TRT-LLM 0.19.0 supports `LlamaForCausalLM` — same architecture as Qwen (transformer decoder).
- Build scripts at `/app/examples/llama/convert_checkpoint.py` and `trtllm-build`.
- Model `meta-llama/Meta-Llama-3.1-8B-Instruct` — gated; requires HF token.
- VRAM: ~16 GB FP16. With explicit mode (only one pipeline at a time), fits within 32 GB.

**Build path** (inside Triton 25.05 container, with GPU):
```bash
# Download (requires HF_TOKEN)
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct \
  --local-dir /ai/trtllm-poc/models/llama-3.1-8b

# Convert checkpoint
python3 /app/examples/llama/convert_checkpoint.py \
  --model_dir /ai/trtllm-poc/models/llama-3.1-8b \
  --output_dir /ai/trtllm-poc/engines/llama/ckpt \
  --dtype float16

# Build TRT engine
trtllm-build \
  --checkpoint_dir /ai/trtllm-poc/engines/llama/ckpt \
  --output_dir /ai/trtllm-poc/engines/llama \
  --gemm_plugin float16 \
  --max_batch_size 4 \
  --max_input_len 2048 \
  --max_seq_len 4096
```

**Triton pipeline**: Standard `inflight_batcher_llm` — same as qwen-coder. Rename with `_llama`
suffix. `fill_template.py` generates config.pbtxt files from templates.

---

## R6 — OpenAI Frontend Vision Input Handling

**Question**: Does `main.py`'s `/v1/chat/completions` endpoint pass image content through to
the Triton multimodal pipeline?

**Findings**: The `TritonLLMEngine` class wraps `tritonserver.Server` and handles request
formatting. The `main.py` `--backend` arg controls input/output tensor names. The multimodal
pipeline uses a different tensor schema than `inflight_batcher_llm`.

**Decision**: Verify during implementation whether `main.py` passes `image_url` content to
the `multimodal_encoders` model. If not, the Rust client's `Caption` capability may need to
encode images differently or the smoke test acceptance criterion for US3 may need adjustment.
This is a known implementation-time research item, not a blocker for planning.

**Fallback**: If `main.py` does not pass image content to the multimodal pipeline, use the
Triton v2 native API (port 8000) for caption requests from the Rust client directly.
