<!--
SYNC IMPACT REPORT
==================
Version change: (none) → 1.0.0
Modified principles: N/A (initial ratification)
Added sections: Core Principles (6), Technology Context, Development Workflow, Governance
Removed sections: N/A
Templates requiring updates:
  - .specify/templates/plan-template.md ✅ reviewed, no changes required
  - .specify/templates/spec-template.md ✅ reviewed, no changes required
  - .specify/templates/tasks-template.md ✅ reviewed, no changes required
Follow-up TODOs: none — all placeholders resolved
-->

# TRT-LLM POC Constitution

## Core Principles

### I. Documentation First

Every setup step, architectural decision, model choice, and tunable parameter MUST be documented
as it happens — not retroactively. Documentation MUST be usable by both the human owner and an
AI coding agent picking up the project cold.

- Each `just` recipe and Docker flag MUST have a comment explaining the "why", not just the "what"
- Decision records (ADR-style, even if informal) MUST capture the options considered and the
  reason for the choice made
- Any deviation from the prep document (`TRT-LLM POC Weekend Spec Prep.md`) MUST be noted inline
- All docs live under `docs/` at the repo root; agent-consumable context goes in `.specify/`

### II. Ansible-k Handoff Required for Host Changes

Any modification to the host machine — installing a tool, adding a Docker network/volume,
changing NVIDIA runtime config, editing `/etc/systemd/`, or any persistent system change — MUST
be captured in a form suitable for promotion to the ansible-k playbooks at
`/home/ken/src/config-src/ansible-k`.

- Document new tools as: tool name, install command, version pinned, purpose
- Document new system config as: file path, content or diff, service dependency
- A `docs/ansible-handoff.md` file accumulates these items throughout the POC
- This is non-negotiable even for a POC; host drift that isn't captured is technical debt

### III. POC Pragmatism (YAGNI Enforced)

This is a throwaway learning project with a weekend time budget. Working over perfect.
Completeness over elegance. Learning over production-readiness.

- MUST NOT over-engineer: no abstractions for one-time operations, no premature
  generalization, no speculative features
- MUST follow the two-track progression: Track A (dev/engine-build) proves TRT-LLM works;
  Track B (Triton serving) exercises the service boundary for Rust integration
- Feature scope is locked to the three capabilities defined in the prep doc:
  general chat, coding assistant, image captioning
- If a direction is blocked, document the blocker and take the next simplest path

### IV. Spec-Driven (SDD — Relaxed for POC)

Specifications drive implementation. For this POC, lightweight specs are acceptable, but the
intent and acceptance criteria MUST be written down before code is written.

- User stories and acceptance criteria defined in `spec.md` MUST be referenced by tasks
- "Spec first" means: write what done looks like, then implement to that description
- Formal spec review is waived; the prep document serves as the initial specification seed
- Any scope change after spec is written MUST be reflected back in the spec, not just the code

### V. Test-Driven (TDD — Relaxed for POC)

Tests are required for Rust client code. Integration smoke-tests are required for the three
acceptance criteria (GPU visible, serving endpoint reachable, three capabilities respond).
Unit test coverage targets are relaxed given POC scope.

- Rust crate: `cargo test` MUST pass before a feature is considered done
- Integration acceptance: the checklist in the prep doc (`5.5 Acceptance criteria`) serves as
  the integration test suite; each item MUST be manually or automatically verified and checked off
- Test infrastructure (fixtures, mock servers) MUST NOT be more complex than the code under test
- No test = no merge for Rust API boundary code; scripts and config are exempt

### VI. Observability and Simplicity

The system MUST be debuggable at every layer without specialized tooling.

- `nvidia-smi` MUST confirm GPU visibility before any model work proceeds
- Latency, model chosen, and token counts MUST be logged per request in the Rust client
  (as defined in the prep doc observability hooks)
- Log to stdout/stderr; structured JSON preferred but plain text acceptable for POC
- When a simpler implementation and a more capable one both satisfy the acceptance criteria,
  the simpler one MUST be chosen

## Technology Context

**Host**: NVIDIA GeForce RTX 5090 (Blackwell, SM 12.0) · Driver 595.58.03 · CUDA 13.2  
**Docker images**:
- Track A (dev): `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13`
- Track B (serving): `nvcr.io/nvidia/tritonserver:25.05-trtllm-python-py3`

**Rust stack**: `reqwest`, `serde`/`serde_json`, `base64`, `tokio`  
**Lifecycle tooling**: `just` (justfile), Docker Compose (`restart: unless-stopped`)  
**Persistence root**: `$TRTLLM_HOME` (default `/ai/trtllm-poc`), with `models/`, `engines/`,
`cache/`, `logs/`, `model_repo/` subdirectories

**Model targets (POC-sane, 32 GB-friendly)**:
- Chat: Llama 3.x 8B Instruct
- Code: Qwen2.5-Coder 7B Instruct
- Caption: LLaVA-family small VLM (TRT-LLM supported)

**Ansible-k project**: `/home/ken/src/config-src/ansible-k` — receives host-change docs

## Development Workflow

1. **Spec first**: acceptance criteria written in `spec.md` before implementation begins
2. **Track A before Track B**: prove GPU + engine build works before standing up Triton serving
3. **Document as you go**: update `docs/` and `docs/ansible-handoff.md` during, not after
4. **Rust tests gate merges**: `cargo test` green required for Rust client code
5. **Check off acceptance criteria**: the checklist from the prep doc is the definition of done
6. **Capture blockers**: if something doesn't work, write a decision record noting what was tried
   and what path was taken instead

Constitution compliance is verified at the start of each planning session and before
closing the POC. Any agent working in this repo MUST read this file before generating tasks.

## Governance

This constitution supersedes all other practices for the duration of this POC.

- **Amendments**: Any principle change MUST increment the version and record the rationale
  in a `<!-- SYNC IMPACT REPORT -->` comment at the top of this file
- **Versioning**: MAJOR for principle removal/redefinition; MINOR for new principle/section;
  PATCH for wording clarifications
- **Compliance review**: performed at start of plan phase and before POC close-out
- **Retirement**: this constitution governs only the POC repo; learnings that affect the
  long-lived host setup are promoted to ansible-k, not encoded here permanently

**Version**: 1.0.0 | **Ratified**: 2026-05-01 | **Last Amended**: 2026-05-01
