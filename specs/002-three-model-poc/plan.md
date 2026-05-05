# Implementation Plan: Three-Model POC

**Branch**: `002-three-model-poc` | **Date**: 2026-05-03 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `specs/002-three-model-poc/spec.md`

## Summary

Add explicit model-control mode to Triton serving (US1/P1) so the GPU can be reclaimed
without restarting the container. Build a Llama 3.1 8B Instruct engine for the Chat
capability (US2/P2) and a LLaVA 1.5 7B engine for image captioning (US3/P3). All three
pipelines coexist in a single `model_repo/` using suffixed names (`ensemble_qwen`,
`ensemble_llama`, `ensemble_vision`); only one loads at a time via EXPLICIT mode.

**Technical approach** (resolved in research):
- `main.py` does not expose `model_control_mode` — a thin wrapper `triton_wrapper.py`
  monkey-patches `tritonserver.Server` before delegating to `main.main()`.
- KServe HTTP frontend (`--enable-kserve-frontends`, port 8000) provides the
  `POST /v2/repository/models/{name}/load|unload` control plane.
- LLaVA-NeXT (1.6) is NOT supported in TRT-LLM 0.19.0 C++ runtime; **LLaVA 1.5 7B**
  (`llava-hf/llava-1.5-7b-hf`) is used instead — fully `[x]` supported.

## Technical Context

**Language/Version**: Rust 1.75+ (client); Python 3.12 (Triton 25.05 container); Bash (scripts)  
**Primary Dependencies**: `tritonserver` Python API 0.19.0; TRT-LLM 0.19.0; Docker Compose; `reqwest`/`serde` (Rust)  
**Storage**: Filesystem — `$TRTLLM_HOME`: `engines/`, `model_repo/`, `models/`, `logs/`  
**Testing**: `cargo test` (Rust routing); manual smoke-test per SC-001–SC-007  
**Target Platform**: Linux server — Ubuntu 24.04, NVIDIA RTX 5090 (32 GB VRAM), Driver 595.58.03  
**Project Type**: ML serving infrastructure (CLI + service)  
**Performance Goals**: Single-pipeline exclusive VRAM occupancy; inference latency not a primary POC goal  
**Constraints**: TRT-LLM 0.19.0 pinned (Triton 25.05); 32 GB VRAM; one pipeline active at a time; `cargo test` must pass  
**Scale/Scope**: 3 models, 1 GPU, 1 container instance, POC time-budget  

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|---|---|---|
| I. Documentation First | ✅ PASS | All decisions documented in research.md and data-model.md as they were made. `docs/setup.md` update is a required task (FR-013, FR-014). |
| II. Ansible-k Handoff | ✅ PASS | No new host-level changes in this sprint. Engine builds and model_repo setup are container-internal. No changes to nvidia-container-toolkit or systemd. |
| III. POC Pragmatism (YAGNI) | ✅ PASS | Monkey-patch wrapper is minimal (< 30 lines). Suffixed naming avoids model_namespacing complexity. No abstractions for one-time ops. |
| IV. Spec-Driven | ✅ PASS | All 14 FRs and 7 SCs from spec.md are addressed. No scope changes after spec write. |
| V. Test-Driven | ✅ PASS | `cargo test` required before done. Smoke-test (SC-006) exercises all three capabilities. Routing tests in `tests/routing.rs` will be updated for new model IDs. |
| VI. Observability | ✅ PASS | Rust client logs model_id, token counts, latency. `just models-status` provides operational visibility. |

**Gate result: PASS — proceed to implementation.**

**Post-design re-check** (after Phase 1):
All principles still pass. The suffix naming convention and separate pipeline dirs add
moderate config complexity but are the simplest approach that satisfies EXPLICIT mode
coexistence without name collisions. No violations introduced.

## Project Structure

### Documentation (this feature)

```text
specs/002-three-model-poc/
├── plan.md              # This file (/speckit.plan output)
├── research.md          # Phase 0 output — all NEEDS CLARIFICATION resolved
├── data-model.md        # Phase 1 output — entities, naming, VRAM budget
├── quickstart.md        # Phase 1 output — step-by-step engine build & model control
├── contracts/
│   └── interfaces.md   # Phase 1 output — OpenAI API, Triton v2 API, just CLI contracts
└── tasks.md             # Phase 2 output (/speckit.tasks — NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
scripts/
├── start-triton.sh          # Updated: exec triton_wrapper.py instead of main.py directly
├── triton_wrapper.py         # NEW: monkey-patch tritonserver.Server for explicit mode
└── setup-model-repo.sh       # NEW: create/rename model_repo pipeline dirs with suffix

model_repo/                   # (at $TRTLLM_HOME, not in repo — managed by setup script)
├── preprocessing_qwen/       # renamed from preprocessing
├── tensorrt_llm_qwen/        # renamed from tensorrt_llm
├── postprocessing_qwen/      # renamed from postprocessing
├── ensemble_qwen/            # renamed from ensemble
├── tensorrt_llm_bls_qwen/    # renamed from tensorrt_llm_bls
├── preprocessing_llama/      # NEW: Llama pipeline
├── tensorrt_llm_llama/
├── postprocessing_llama/
├── ensemble_llama/
├── tensorrt_llm_bls_llama/
├── multimodal_encoders/      # NEW: vision pipeline (multimodal template)
├── tensorrt_llm_vision/
├── postprocessing_vision/
└── ensemble_vision/

rust-client/src/
└── capability.rs             # Updated: Chat→ensemble_llama, Caption→ensemble_vision

justfile                      # Updated: models-load, models-unload, models-status,
                              #          build-llama, build-llava, setup-model-repo,
                              #          smoke-all recipes

compose.yaml                  # Updated: TRITON_STARTUP_MODELS, KSERVE_HTTP_PORT,
                              #          --enable-kserve-frontends passed to wrapper

.env.example                  # Updated: DEFAULT_MODEL var documented

docs/setup.md                 # Updated: §4.2 Llama build, §4.3 LLaVA build,
                              #          §4.4 multi-pipeline layout,
                              #          §5 explicit mode + model control workflow
```

**Structure Decision**: Single project. The repo root holds Rust crate, Docker config,
scripts, and docs. No new top-level directories needed. `model_repo/` lives at
`$TRTLLM_HOME` (host filesystem, not tracked in git) and is managed by the
`setup-model-repo.sh` script.

## Complexity Tracking

No Constitution violations requiring justification. The monkey-patch wrapper and suffixed
naming are both minimal and directly required by the acceptance criteria.
