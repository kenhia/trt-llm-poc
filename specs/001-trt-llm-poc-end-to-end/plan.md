# Implementation Plan: TensorRT-LLM POC — End-to-End on RTX 5090

**Branch**: `001-trt-llm-poc-end-to-end` | **Date**: 2026-05-01 | **Spec**: [spec.md](spec.md)  
**Input**: Feature specification from `specs/001-trt-llm-poc-end-to-end/spec.md`

## Summary

Prove TensorRT-LLM works end-to-end on an RTX 5090 (Blackwell/SM 12.0) using NVIDIA NGC
Docker images. Build a minimal Rust client that calls a Triton HTTP endpoint to exercise
three inference capabilities — chat, code, and image captioning. All host changes are
documented for ansible-k handoff. The two-track progression (Track A: engine build proof
→ Track B: Triton serving) is the core architecture.

## Technical Context

**Language/Version**: Rust stable (latest at build time) for client; Python managed inside NGC containers  
**Primary Dependencies**: `reqwest`, `serde`/`serde_json`, `base64`, `tokio` (Rust client); NGC images for inference  
**Storage**: Filesystem only — `$TRTLLM_HOME/{models,engines,cache,logs}` on host, bind-mounted into containers  
**Testing**: `cargo test` (Rust unit tests for routing/models); manual smoke tests (integration acceptance checklist)  
**Target Platform**: Linux host, single RTX 5090 (SM 12.0 Blackwell), Docker with NVIDIA Container Toolkit  
**Project Type**: POC — configuration/scripts + single Rust client crate  
**Performance Goals**: Cold-start first inference ≤60s; container up after reboot ≤2min  
**Constraints**: 32 GB VRAM; single GPU; weekend time budget; no over-engineering (YAGNI)  
**Scale/Scope**: Single developer, local machine only; not intended for multi-user or production use

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design: ✅ PASS*

| Principle | Check | Notes |
|-----------|-------|-------|
| I. Documentation First | ✅ | `docs/setup.md` + `docs/ansible-handoff.md` + inline comments required by FR-011/FR-012 |
| II. Ansible-k Handoff | ✅ | `docs/ansible-handoff.md` is a first-class deliverable (FR-011, SC-006) |
| III. POC Pragmatism | ✅ | Scope locked to 3 capabilities; no abstractions beyond what the spec requires |
| IV. Spec-Driven | ✅ | spec.md with acceptance criteria exists before implementation begins |
| V. TDD (Relaxed) | ✅ | `cargo test` required for Rust (FR-010, SC-005); integration tests = acceptance checklist |
| VI. Observability | ✅ | Per-request logging (FR-009); `nvidia-smi` gates model work |

**No violations. No complexity justification required.**

## Project Structure

### Documentation (this feature)

```text
specs/001-trt-llm-poc-end-to-end/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   └── http-api.md      # Triton/OpenAI-compat HTTP contract
└── tasks.md             # Phase 2 output (speckit.tasks — not created by speckit.plan)
```

### Source Code (repository root)

```text
justfile                   # lifecycle recipes: pull/dev/compose-up/down/logs/restart/purge/gpu/ps
compose.yaml               # Triton serving container, restart: unless-stopped
.env.example               # TRTLLM_HOME and HF_TOKEN template (committed; .env is gitignored)

rust-client/               # Rust crate — the Rust integration client
├── Cargo.toml
├── src/
│   ├── main.rs            # CLI entry point: run smoke tests / interactive mode
│   ├── client.rs          # LlmClient trait + reqwest HTTP implementation
│   ├── capability.rs      # Capability enum (Chat/Code/Caption) + routing logic
│   ├── models.rs          # InferenceRequest / InferenceResponse types (serde)
│   └── logging.rs         # per-request structured log (id, latency, model, tokens)
└── tests/
    └── routing.rs         # unit tests: routing logic for all three capabilities

docs/
├── setup.md               # end-to-end setup procedure (FR-012)
├── ansible-handoff.md     # host-change log for ansible-k promotion (FR-011)
└── decisions/             # ADR-style decision records
    ├── 001-compose-vs-systemd.md
    ├── 002-openai-compat-vs-triton-native.md
    └── 003-model-selection.md
```

**Structure Decision**: Single project layout with a Rust crate under `rust-client/` alongside
top-level infra files (`justfile`, `compose.yaml`). Docs live under `docs/` per the constitution.
No `src/` at repo root — the Rust crate is scoped to `rust-client/` to keep infra and code
cleanly separated.

## Complexity Tracking

No constitution violations detected. Table not required.
