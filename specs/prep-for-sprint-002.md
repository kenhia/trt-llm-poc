# Prep Notes for Sprint 002

**Prepared**: 2026-05-03  
**Source sprint**: `001-trt-llm-poc-end-to-end`  
**Intended use**: input to `/speckit.specify` for the next sprint

---

## What Sprint 001 Delivered

Sprint 001 proved the TRT-LLM + Triton stack end-to-end on a single model (Qwen2.5-Coder-7B):

- RTX 5090 GPU visible inside both containers (SC-001 ✅)
- Engine built inside Triton 25.05 container (TRT-LLM 0.19.0) and served via Triton (SC-002 ✅)
- OpenAI-compatible frontend (`/v1/chat/completions`) running on port 9000 (SC-002 ✅)
- Rust client (`cargo run -- smoke-test`) exercising Chat and Code capabilities via live endpoint ✅
- Persistent serving with `restart: unless-stopped` (SC-003 ✅)
- Full `inflight_batcher_llm` five-model pipeline in Triton model repository ✅
- All 27 tasks marked complete in `specs/001-trt-llm-poc-end-to-end/tasks.md`

**What was NOT completed in sprint 001** (deliberately deferred):

- Llama 3.1 8B engine (Chat capability via dedicated model)
- LLaVA-1.6 engine (Caption / vision capability)
- Model load/unload control (Triton explicit model-control mode)

---

## Sprint 002 Goals

Complete the original three-model POC:

1. **Add Llama 3.1 8B Instruct** as the dedicated Chat model (replacing qwen-coder for chat)
2. **Add LLaVA-1.6** as the Caption/vision model — even if complex, this is a deliberate goal not to defer again
3. **Add Triton model-control mode** so models can be loaded/unloaded individually via API without restarting the container
4. **Add `just` recipes** for model load/unload so the GPU can be reclaimed during adjacent development work (e.g. `multae-viae` project)

---

## Known Context / Constraints

### Hardware

- **Host**: Ubuntu 24.04.4, kernel 6.8.0-111, driver 595.58.03, RTX 5090 (32 GB VRAM)
- **VRAM budget**: 32 GB total. Current qwen-coder engine (FP16) uses ~30.5 GB alone.
  All three models **cannot** run concurrently in FP16. Options:
  - Rebuild engines in FP8 (~9-12 GB each → ~30 GB for all three)
  - Use explicit model-control mode to load/unload on demand (avoids rebuilding)
  - Decision: use explicit model-control mode as primary strategy; FP8 rebuilds are a
    stretch goal / optimization

### Version Mismatch (Critical)

- Track A dev container: TRT-LLM **1.3.0rc13**
- Triton 25.05 container: TRT-LLM **0.19.0**
- Engines for Triton **must** be built inside the Triton 25.05 container, not the dev container
- This constraint applies to all three engines

### Existing Infrastructure

- `TRTLLM_HOME=/ai/trtllm-poc` — all artifacts live here
- `compose.yaml` — Triton serving container, OpenAI frontend on port 9000
- `scripts/start-triton.sh` — container entrypoint; execs `main.py` (embedded Triton)
- `model_repo/` — currently has the `inflight_batcher_llm` pipeline for qwen-coder only
- Rust client crate at `rust-client/` — `LlmClient` trait, routing tests, smoke-test binary

### Convert Script Paths (inside Triton 25.05 container)

Verified paths during sprint 001:
- Qwen: `/opt/tritonserver/backends/tensorrtllm/examples/qwen/convert_checkpoint.py`
- Llama: `/opt/tritonserver/backends/tensorrtllm/examples/llama/convert_checkpoint.py`
- LLaVA: unknown — needs research inside the container

---

## Detailed Requirements for Sprint 002

### REQ-001 — Llama 3.1 8B Chat Engine

Build and serve Llama 3.1 8B Instruct as a second model in Triton.

- HF model: `meta-llama/Meta-Llama-3.1-8B-Instruct` (gated — requires HF access token)
- Build inside Triton 25.05 container targeting TRT-LLM 0.19.0
- Checkpoint output: `$TRTLLM_HOME/models/llama3-8b-ckpt-0.19/`
- Engine output: `$TRTLLM_HOME/engines/llama3-chat/`
- Separate `inflight_batcher_llm` pipeline in `model_repo/` with distinct names
  (e.g., `preprocessing_llama`, `tensorrt_llm_llama`, `postprocessing_llama`,
  `ensemble_llama`, `tensorrt_llm_bls_llama` — or however Triton supports multi-engine)
- Update `capability.rs` `Chat` arm to route to the Llama pipeline's entry point
- Quantization: FP16 initially (matching qwen); FP8 if VRAM is insufficient

### REQ-002 — LLaVA-1.6 Caption Engine (vision/multi-modal)

Build and serve a vision-language model for image captioning. This has been deferred
across multiple projects and should be completed in this sprint.

**Primary model**: `llava-hf/llava-v1.6-mistral-7b-hf`  
**Fallback**: `Qwen/Qwen2-VL-7B-Instruct` (Qwen2-VL is in TRT-LLM support matrix and
may have simpler multi-modal integration)

Key unknowns to research at sprint start:
- Does TRT-LLM 0.19.0 (Triton 25.05 container) support LLaVA-1.6 multi-modal?
  - Check: `ls /opt/tritonserver/backends/tensorrtllm/examples/llava/` inside container
  - Check: `ls /opt/tritonserver/backends/tensorrtllm/examples/multimodal/`
- LLaVA requires two engines: visual encoder + language model decoder. Does the
  `inflight_batcher_llm` pipeline handle this, or is a different pipeline layout needed?
- VRAM budget: LLaVA-1.6 Mistral 7B estimated ~12 GB FP16. With all three models
  concurrent: ~30 GB + ~12 GB = OOM. This is why model-control mode (REQ-003) is
  a prerequisite or co-deliverable.

If LLaVA-1.6 proves incompatible with 0.19.0, fall back to `Qwen2-VL-7B-Instruct`.
Document the blocker and rationale before falling back.

### REQ-003 — Triton Explicit Model-Control Mode

Enable loading and unloading individual models via the Triton v2 API, without
restarting the container.

**Changes needed**:
- `compose.yaml`: pass `--model-control-mode=explicit` and `--load-model=<name>` to
  `tritonserver` (or `main.py` equivalent)
- Research how `main.py` (OpenAI frontend) handles `--model-control-mode=explicit`;
  it may need to pass through to the embedded Triton
- Add `just` recipes:
  ```
  just models-load <pipeline>    # load a named pipeline
  just models-unload <pipeline>  # unload a named pipeline
  just models-status             # list loaded/available models
  ```
- The recipes should call `POST localhost:8000/v2/repository/models/<name>/load` etc.
- At startup, Triton should load no models by default (explicit mode); each pipeline
  is loaded on demand

**Desired workflow** (motivation: GPU reclaim during adjacent dev work):
```bash
# Before switching to multae-viae:
just models-unload ensemble     # frees ~30 GB VRAM; Triton container stays up

# When returning to trt-llm-poc:
just models-load ensemble       # reloads in ~30s; no full compose-up needed
```

### REQ-004 — Rust Client Updates

Update `rust-client` to support multi-model routing and model-control:
- `capability.rs`: `Chat` routes to Llama pipeline, `Code` routes to qwen pipeline,
  `Caption` routes to LLaVA pipeline
- `client.rs`: optionally add `load_model(name)` / `unload_model(name)` methods
  wrapping the Triton v2 repository API
- `smoke-test` subcommand: all three capabilities should produce real responses
  (not degrade to warnings)

### REQ-005 — docs/setup.md Updates

Extend `§4.2` and `§4.3` to cover all three engines:
- `§4.2`: Add Llama 3.1 8B and LLaVA-1.6 engine build procedures alongside the
  existing qwen-coder procedure
- `§4.3`: Document multi-engine model repository layout — how to have multiple
  `inflight_batcher_llm` pipelines co-existing in `model_repo/`
- Add a new `§4.5` (or equivalent) for explicit model-control mode setup
- Update `§5` startup/health-check steps to reflect multi-model environment

---

## Open Questions for Sprint 002 Planning

1. **Multi-pipeline naming convention**: Triton requires all model names to be unique.
   What naming scheme to use for a three-engine setup?
   Options: `{model_name}_preprocessing` / `{model_name}_ensemble` / etc., or
   separate top-level directories for each pipeline.

2. **`main.py` + explicit model-control**: The OpenAI frontend embeds Triton via
   Python bindings. Does `tritonserver.Server()` accept `model_control_mode=EXPLICIT`?
   Needs verification inside the container before committing to this approach.

3. **LLaVA pipeline for 0.19.0**: The multi-modal pipeline layout (visual encoder + LM)
   may differ from `inflight_batcher_llm`. The sprint should begin with a research
   task to map the correct pipeline structure before any engine builds.

4. **Startup model selection**: With explicit mode, should the compose container start
   with no models loaded (fully on-demand), or load one model by default?
   Recommendation: load the most-recently-used model at startup via an env var
   (`DEFAULT_MODEL=qwen-coder`).

---

## Files Changed in Sprint 001 (for handoff awareness)

| File | What changed |
|------|-------------|
| `.env.example` | Added `TRTLLM_HOST`, `TRTLLM_PORT=9000`, `OPENAI_TOKENIZER`, `OPENAI_PORT` |
| `compose.yaml` | Added port 9000, `OPENAI_TOKENIZER`/`OPENAI_PORT` env, script volume mount, updated command |
| `justfile` | `compose-install` copies `scripts/` in addition to `compose.yaml` |
| `scripts/start-triton.sh` | New — container entrypoint, execs `main.py` directly |
| `rust-client/src/capability.rs` | All three capabilities route to `"ensemble"` (single engine POC) |
| `rust-client/src/client.rs` | Default port 8000 → 9000 |
| `rust-client/src/main.rs` | Caption failure non-fatal (SC-004 placeholder) |
| `docs/setup.md` | §4 rewritten for Triton-container build + multi-model pipeline; §5 updated for embedded Triton architecture |
| `specs/001-trt-llm-poc-end-to-end/tasks.md` | Phase 8 (T024–T027) added and marked complete |

---

## Suggested Sprint 002 Opening Prompt

After running `/speckit.specify`, the context doc to provide is this file.
Suggested prompt:

```
/speckit.specify use specs/prep-for-sprint-002.md as the basis for the next sprint
```

The speckit agent should produce a new spec under `specs/002-three-model-poc/` (or
similar branch name) covering REQ-001 through REQ-005 above.
