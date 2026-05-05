# Feature Specification: Three-Model POC — Llama3, LLaVA, and Model-Control Mode

**Feature Branch**: `002-three-model-poc`
**Created**: 2026-05-03
**Status**: Draft
**Input**: [specs/prep-for-sprint-002.md](../prep-for-sprint-002.md)

## User Scenarios & Testing

### User Story 1 — GPU Reclaim via Model Load/Unload (Priority: P1)

As the developer, I want to unload the currently running Triton model from GPU memory
and reload it on demand, without restarting the Triton container, so that I can switch
between the TRT-LLM POC and other GPU-intensive projects (e.g. `multae-viae`) without
a 30–60 second full container restart cycle.

**Why this priority**: This is the most immediately useful capability — it unblocks
adjacent project work today and is a prerequisite for safely running all three engines
on a 32 GB GPU. Without this, adding more models causes OOM at startup.

**Independent Test**: With Triton running and qwen-coder loaded, run `just models-unload qwen`
and confirm `nvidia-smi` shows VRAM freed. Then run `just models-load qwen` and confirm
inference still works. The container stays up throughout.

**Acceptance Scenarios**:

1. **Given** Triton is running with qwen-coder loaded, **When** `just models-unload qwen` is run, **Then** VRAM drops by ≥20 GB and `just models-status` shows the pipeline as not loaded
2. **Given** the pipeline is unloaded, **When** `just models-load qwen` is run, **Then** VRAM returns to previous level and `curl /v1/chat/completions` with `model: "ensemble"` returns a response within 60 seconds
3. **Given** the Triton container is started fresh, **When** no explicit load has been issued, **Then** a configurable default model is loaded automatically at startup (via `DEFAULT_MODEL` env var)
4. **Given** one pipeline is loaded, **When** `just models-load llama` is run, **Then** Triton loads the Llama pipeline and `just models-status` reflects the change

---

### User Story 2 — Dedicated Chat Engine (Llama 3.1 8B) (Priority: P2)

As the developer, I want the Chat capability to route to a dedicated Llama 3.1 8B
Instruct engine rather than the coding-focused Qwen model, so that general conversation
and instruction-following quality is demonstrably better than using a code model.

**Why this priority**: Completing the original three-model goal — separate, purpose-built
models for each capability. Requires REQ-001 (engine build) and US1 (model-control mode)
to use it within the 32 GB VRAM budget.

**Independent Test**: With the Llama pipeline loaded, run `cargo run -- smoke-test` with
Chat capability only and confirm the response comes from Llama (model name in the log
will be the Llama pipeline entry point, not `ensemble`).

**Acceptance Scenarios**:

1. **Given** `meta-llama/Meta-Llama-3.1-8B-Instruct` weights are downloaded and a TRT-LLM engine is built inside the Triton 25.05 container, **When** `just models-load llama` is run, **Then** all five Llama pipeline models show READY in `just models-status`
2. **Given** the Llama pipeline is loaded, **When** `POST /v1/chat/completions` is called with `model: "ensemble_llama"` (or the pipeline's entry-point name), **Then** a coherent conversational response is returned within 60 seconds
3. **Given** the Rust client's `Chat` capability is updated to route to the Llama pipeline, **When** `cargo run -- smoke-test` is run with Chat, **Then** the log shows the Llama model ID and the response is non-empty

---

### User Story 3 — Vision/Caption Engine (LLaVA or Qwen2-VL) (Priority: P3)

As the developer, I want the Caption capability to route to a working vision-language
model that can describe an image, so that the POC demonstrates all three intended
inference modalities: text chat, code generation, and image captioning.

**Why this priority**: This has been deferred across multiple projects and is the
sprint's stated goal to finally complete. P3 because it is the most technically
uncertain (multi-modal pipeline complexity in TRT-LLM 0.19.0) and requires both
US1 (model-control) and a successful research step to unblock.

**Independent Test**: With the vision pipeline loaded, `cargo run -- smoke-test` Caption
capability sends a base64 image and receives a non-empty text description (not a 400 error).

**Acceptance Scenarios**:

1. **Given** a vision model engine and pipeline are built, **When** `just models-load llava` (or `qwen-vl`) is run, **Then** the pipeline models show READY
2. **Given** the vision pipeline is loaded, **When** `POST /v1/chat/completions` is called with a base64-encoded image in the message content, **Then** a textual description is returned (not HTTP 400)
3. **Given** the Rust client's `Caption` capability is updated to route to the vision pipeline, **When** `cargo run -- smoke-test` is run, **Then** all three capabilities (Chat, Code, Caption) return non-error responses and the smoke-test exits 0

---

### Edge Cases

- What happens when `just models-load` is called for a pipeline that is already loaded? Triton should be idempotent (return success); the recipe should not error.
- What happens when `just models-load` is called but Triton is not running? The recipe should fail fast with a clear error message.
- What happens if VRAM is insufficient to load a second pipeline while another is loaded? Triton returns a load error; `just models-load` should surface the error to stdout.
- What happens if the HF token is missing when downloading gated Llama weights? Download fails with 403; setup doc must clearly show where to set `HF_TOKEN`.
- What happens if the LLaVA engine format is incompatible with TRT-LLM 0.19.0? Research vision-capable models confirmed to work with TRT-LLM 0.19.0, select the best fit, and document the investigation and rationale before proceeding with an alternative.

---

## Requirements

### Functional Requirements

- **FR-001**: Triton MUST start in explicit model-control mode so no pipeline is loaded automatically at startup (unless `DEFAULT_MODEL` is set)
- **FR-002**: A `DEFAULT_MODEL` environment variable MUST cause the named pipeline to be loaded automatically after Triton becomes ready, enabling a zero-friction restart experience
- **FR-003**: `just models-load <pipeline>` MUST send a load request to the Triton v2 repository API and report success or failure
- **FR-004**: `just models-unload <pipeline>` MUST send an unload request and confirm VRAM is freed (verify via `just models-status`)
- **FR-005**: `just models-status` MUST list all available pipelines with their current state (READY or not loaded)
- **FR-006**: A Llama 3.1 8B Instruct engine MUST be built inside the Triton 25.05 container (TRT-LLM 0.19.0), producing a compatible `.engine` file
- **FR-007**: The Llama pipeline MUST use a distinct naming scheme from the qwen-coder pipeline so both can coexist in `model_repo/` simultaneously
- **FR-008**: A vision-language engine (LLaVA 1.5 7B, `llava-hf/llava-1.5-7b-hf` — confirmed supported in TRT-LLM 0.19.0 C++ runtime) MUST be built and served via a dedicated Triton pipeline
- **FR-009**: The Rust client `Chat` capability MUST route to the Llama pipeline entry point
- **FR-010**: The Rust client `Code` capability MUST continue routing to the qwen-coder pipeline entry point
- **FR-011**: The Rust client `Caption` capability MUST route to the vision pipeline entry point and successfully return a response when an image is provided
- **FR-012**: `cargo run -- smoke-test` MUST exit 0 with non-error responses for all three capabilities (Chat, Code, Caption) when appropriate pipelines are loaded
- **FR-013**: `docs/setup.md` §4 MUST document the engine build procedure for all three models and the multi-pipeline model repository layout
- **FR-014**: `docs/setup.md` MUST document explicit model-control mode setup and the `just models-*` workflow

### Key Entities

- **Pipeline**: A named set of five Triton models (`preprocessing_<name>`, `tensorrt_llm_<name>`, `postprocessing_<name>`, `ensemble_<name>`, `tensorrt_llm_bls_<name>`) backed by a single `.engine` file
- **Engine**: A compiled TRT-LLM binary (`.engine` file + `config.json`) built for a specific model architecture and TRT-LLM version (0.19.0 for Triton 25.05)
- **Model Repository**: The directory (`$TRTLLM_HOME/model_repo/`) containing all pipeline directories watched by Triton

## Success Criteria

### Measurable Outcomes

- **SC-001**: `just models-unload <pipeline>` frees ≥20 GB VRAM within 30 seconds as reported by `nvidia-smi`
- **SC-002**: `just models-load <pipeline>` makes the pipeline ready for inference within 60 seconds of invocation
- **SC-003**: `just models-status` accurately reflects the loaded/unloaded state of all known pipelines
- **SC-004**: Llama 3.1 8B engine builds successfully inside the Triton 25.05 container and loads without error
- **SC-005**: A vision-language model (LLaVA-1.6 or Qwen2-VL) builds and serves image description requests via `/v1/chat/completions`
- **SC-006**: `cargo run -- smoke-test` exits 0 with all three capabilities (Chat via Llama, Code via Qwen, Caption via vision model) returning non-empty responses
- **SC-007**: `docs/setup.md` covers the full three-engine setup from scratch with no undocumented steps

## Assumptions

- The Triton 25.05 container remains the serving target; TRT-LLM version is 0.19.0 and cannot be changed
- All engines must be built inside the Triton 25.05 container (not the Track A dev container) due to TRT-LLM version incompatibility
- `$TRTLLM_HOME=/ai/trtllm-poc` is the artifact root; sufficient disk space exists for two additional engines (~14 GB each)
- HF access token for `meta-llama/Meta-Llama-3.1-8B-Instruct` is available (gated model)
- The `main.py` OpenAI frontend (embedded Triton via Python bindings) supports explicit model-control mode through the `tritonserver.Server()` API — this must be verified as a sprint-opening research task
- Multi-pipeline coexistence uses a naming convention of `<component>_<model>` (e.g., `ensemble_llama`, `ensemble_qwen`) — exact naming to be validated against Triton's constraints
- LLaVA-1.6 multi-modal support in TRT-LLM 0.19.0 requires a research step before engine build; if incompatible, the fallback model will be determined by researching which vision-capable models are confirmed to work with TRT-LLM 0.19.0
- FP16 quantization is used for all engines initially; FP8 is a stretch goal if VRAM proves insufficient with explicit model-control mode
- The qwen-coder engine already built and working is retained and reused; no rebuild needed
