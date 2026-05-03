---

description: "Task list for TRT-LLM POC end-to-end implementation"
---

# Tasks: TensorRT-LLM POC — End-to-End on RTX 5090

**Input**: Design documents from `specs/001-trt-llm-poc-end-to-end/`  
**Prerequisites**: plan.md ✅ spec.md ✅ research.md ✅ data-model.md ✅ contracts/http-api.md ✅ quickstart.md ✅

**Tests**: Routing unit tests (rust-client/tests/routing.rs) are included because FR-010
and SC-005 explicitly require `cargo test` to pass. Integration testing = manual
acceptance checklist (spec.md §Success Criteria).

**Organization**: Tasks grouped by user story. US1 (GPU proof) must complete before US2
(serving), which must complete before US3 (Rust client) can be smoke-tested end-to-end.
US4 (docs handoff) builds throughout and finalizes last.

## Format: `[ID] [P?] [Story?] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: Which user story this task belongs to (US1–US4)
- All file paths are relative to the repository root

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Repo skeleton, environment template, Rust crate scaffolding, docs stub

- [X] T001 [P] Create .gitignore excluding .env, rust-client/target/, and any local trtllm/ path at repo root
- [X] T002 [P] Create .env.example at repo root with TRTLLM_HOME and HF_TOKEN stubs and usage comments; default TRTLLM_HOME to `/ai/models/trtllm` (subdirectory of the shared models root); note that models/, engines/, cache/, and logs/ all live under this root
- [X] T003 [P] Run `cargo new rust-client --bin` to scaffold the crate skeleton, then update rust-client/Cargo.toml with reqwest/serde/serde_json/base64/tokio/tracing/tracing-subscriber/uuid dependencies
- [X] T004 [P] Create docs/ansible-handoff.md with section skeleton (Tools Installed, System Config, Notes) as a running log placeholder

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infra files used by all user stories — both `justfile` and `compose.yaml` are
required before any Track A/B work can proceed.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [X] T005 [P] Create justfile at repo root with all required recipes: `init`, `pull`, `dev`, `compose-up`, `compose-down`, `compose-logs`, `compose-restart`, `run-triton`, `purge`, `gpu`, `ps`; load dotenv; include inline comments per constitution Principle I
- [X] T006 [P] Create compose.yaml at repo root with Triton Track B service (`nvcr.io/nvidia/tritonserver:25.05-trtllm-python-py3`), `restart: unless-stopped`, `runtime: nvidia`, ports 8000/8001/8002, shm_size 2gb, ulimits, volume mount for `${TRTLLM_HOME}`, and inline comments; add a `compose-install` recipe to justfile that copies compose.yaml to `${TRTLLM_HOME}/compose.yaml`

**Checkpoint**: `just --list` works, `compose.yaml` validates with `docker compose config`

---

## Phase 3: User Story 1 — GPU + Engine Proof (Priority: P1) 🎯 MVP

**Goal**: Prove TRT-LLM works end-to-end inside the Track A dev container on the RTX 5090;
`nvidia-smi` shows the GPU, `tensorrt_llm` imports, and a small model generates text.

**Independent Test**: Open `just dev`, run `nvidia-smi`, import `tensorrt_llm`, build a Llama 3.1
8B engine with `trtllm-build`, run inference with `run.py`. No serving or Rust required.

### Implementation for User Story 1

- [X] T007 [P] [US1] Create docs/decisions/001-compose-vs-systemd.md documenting R-004: why Compose was chosen over systemd, with the systemd unit example retained for reference
- [X] T008 [P] [US1] Create docs/decisions/002-openai-compat-vs-triton-native.md documenting R-002: why OpenAI-compatible `/v1/chat/completions` was chosen over Triton native v2
- [X] T009 [P] [US1] Create docs/decisions/003-model-selection.md documenting R-003: model choices (Llama 3.1 8B / Qwen2.5-Coder 7B / LLaVA-1.6 7B), VRAM estimates, and quantization rationale
- [X] T010 [US1] Write docs/setup.md §0–§3: host prerequisites checklist, `$TRTLLM_HOME` init, Track A `just dev` container launch, GPU validation (`nvidia-smi`, `tensorrt_llm` import), model download via `huggingface-cli`, `trtllm-build` command with flags and expected output, inline inference smoke-test

**Checkpoint**: docs/setup.md §0–§3 complete; decision records written; Track A walkthrough
is runnable from a cold start using only docs/setup.md

---

## Phase 4: User Story 2 — Persistent Serving Endpoint (Priority: P2)

**Goal**: Track B Triton container starts automatically after reboot, exposes HTTP on port
8000, and responds to `GET /v1/models` and a minimal inference request.

**Independent Test**: `just compose-up`, wait for Triton ready log, `curl
http://localhost:8000/v1/models` returns model list. Reboot host, confirm container
restarts without manual action.

### Implementation for User Story 2

- [X] T023 [P] [US2] Create triton-model-repo/ at repo root with config.pbtxt templates for all three models (llama3-chat, qwen-coder, llava-caption): each sets name, backend = "tensorrtllm", max_batch_size = 1, and includes inline comments indicating which fields (engine path, instance_count, kv_cache_config) must be updated after engine build; update compose.yaml to bind-mount `${TRTLLM_HOME}/model_repo:/model_repo`
- [X] T011 [US2] Extend docs/setup.md §4–§5: Track B serving setup (Triton model repo layout, `config.pbtxt` name/backend/max_batch_size pattern, `just compose-up` sequence, watching startup logs, `curl /v1/models` health check, port mapping explanation)
- [X] T012 [P] [US2] Populate docs/ansible-handoff.md with all host changes accumulated so far: NVIDIA Container Toolkit install, Docker Compose V2 plugin, `just` tool install, Rust stable toolchain, `$TRTLLM_HOME` directory creation — each with install command, version, and purpose; record actual NGC image digests/tags pulled (output of `docker images --digests` for both Track A and Track B images)

**Checkpoint**: `just compose-up && curl http://localhost:8000/v1/models` succeeds;
docs/ansible-handoff.md has entries for all host tools installed; triton-model-repo/ config.pbtxt templates committed

---

## Phase 5: User Story 3 — Rust Client Exercises Three Capabilities (Priority: P3)

**Goal**: Minimal Rust client crate routes Chat/Code/Caption requests to the correct
model, calls the Triton HTTP endpoint, logs per-request telemetry, and passes `cargo test`.

**Independent Test**: `cargo test` green for routing unit tests; `cargo run -- smoke-test`
returns non-empty coherent responses for all three capabilities against the live endpoint.

### Implementation for User Story 3

- [X] T013 [US3] Create rust-client/src/models.rs with `Message`, `InferenceRequest`, `InferenceResponse` serde structs matching the `contracts/http-api.md` schema; include validation constants for max_tokens and temperature bounds
- [X] T015 [US3] *(TDD stub — write before T014; references `Capability` from T014 and will not compile until T014 is done)* Create rust-client/tests/routing.rs with unit tests covering: image present → Caption, system_hint contains "code" → Code, no image no hint → Chat, image + code hint → Caption (image takes precedence)
- [X] T014 [US3] Create rust-client/src/capability.rs with `Capability` enum (Chat/Code/Caption) and `resolve_capability(req: &InferenceRequest) -> Capability` routing function implementing the three routing rules from data-model.md; implement to make T015 tests compile and pass (TDD green)
- [X] T016 [P] [US3] Create rust-client/src/logging.rs with `log_request(response: &InferenceResponse, success: bool, error: Option<&str>)` emitting `tracing::info!` with all FR-009 fields: request_id, capability, model_id, latency_ms, prompt_tokens, completion_tokens
- [X] T017 [US3] Create rust-client/src/client.rs with `LlmClient` trait (`async fn infer`) and `HttpLlmClient` struct implementing it via reqwest: build OpenAI-compatible JSON body, POST to endpoint URL, deserialize response, measure latency; include `probe_models()` for startup health check against `GET /v1/models`
- [X] T018 [US3] Create rust-client/src/main.rs with `smoke-test` subcommand that builds three `InferenceRequest` values (one per capability), calls `HttpLlmClient::infer` for each, prints the response content, and exits non-zero on any failure; wire up `tracing_subscriber` JSON logging

**Checkpoint**: `cargo test` passes; `cargo build` succeeds; `cargo run -- smoke-test`
(against live Triton) returns responses for chat, code, and caption

---

## Phase 6: User Story 4 — Lifecycle & Documentation Handoff (Priority: P4)

**Goal**: docs/setup.md and docs/ansible-handoff.md together cover the full POC end-to-end
with no undocumented steps — reproducible from scratch by a fresh reader or agent.

**Independent Test**: Read docs/setup.md from §0 to §6 cold; every command referenced must
exist in the repo (justfile, compose.yaml, .env.example); docs/ansible-handoff.md has an
entry for every tool installed on the host during the POC.

### Implementation for User Story 4

- [X] T019 [US4] Complete docs/setup.md §6 troubleshooting section (table of symptoms and first-steps from quickstart.md + any new issues encountered); review entire document §0–§6 for accuracy, completeness, and "why" commentary on every non-obvious step
- [X] T020 [P] [US4] Finalize docs/ansible-handoff.md: review all entries accumulated in T012 and throughout the POC; ensure format matches the constitution II requirement (tool name, install command, version, purpose); add any entries missed during implementation

**Checkpoint**: SC-006 met — docs/setup.md + docs/ansible-handoff.md cover 100% of host
changes; a fresh agent can replicate the environment using only these two files

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Repo presentation and final acceptance validation

- [X] T021 Create README.md at repo root: project title, one-paragraph goal, hardware/stack summary, link to quickstart.md, link to docs/setup.md, and the 6-item acceptance criteria from spec.md SC-001..SC-006
- [X] T022 [P] Validate all acceptance criteria

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately; all T001–T004 run in parallel
- **Foundational (Phase 2)**: Depends on Phase 1 complete; T005 and T006 run in parallel
- **US1 (Phase 3)**: Depends on Phase 2; T007/T008/T009 run in parallel, then T010
- **US2 (Phase 4)**: Depends on Phase 3 (extends docs/setup.md started in T010)
- **US3 (Phase 5)**: Depends on Phase 2 (Cargo.toml from T003); smoke-test requires US2 endpoint live
- **US4 (Phase 6)**: Finalizes docs begun in US1/US2; T019 depends on all prior phases
- **Polish (Phase 7)**: Depends on all phases complete

### User Story Dependencies

- **US1 (P1)**: Can start after Foundational — no dependency on US2/US3/US4
- **US2 (P2)**: Depends on US1 (T011 extends docs/setup.md written in T010)
- **US3 (P3)**: Rust source (T013–T018) can start after Foundational; smoke-test requires US2 live
- **US4 (P4)**: Docs finalization (T019/T020) depends on US1/US2/US3 being complete

### Within US3

```
T013 (models.rs) ──► T015 (test stub, TDD) ──► T014 (capability.rs, TDD green) ─┐
T013 (models.rs) ──► T016 (logging.rs) [P with T014] ──────────────────────────┼──► T017 (client.rs) ──► T018 (main.rs)
```

---

## Parallel Execution Examples

### User Story 1

```bash
# T007, T008, T009 run in parallel (three separate decision doc files):
[ T007: docs/decisions/001-compose-vs-systemd.md    ] ─┐
[ T008: docs/decisions/002-openai-compat-...md      ] ─┼──► T010: docs/setup.md §0–§3
[ T009: docs/decisions/003-model-selection.md       ] ─┘
```

### User Story 3

```bash
# TDD order: T015 stub written before T014 implementation; T016 [P] with T014:
T013: models.rs ──► T015: tests/routing.rs (stub) ──► T014: capability.rs ─┐
T013: models.rs ──► T016: logging.rs [P with T014] ────────────────────────┼──► T017: client.rs ──► T018: main.rs
```

---

## Implementation Strategy

**MVP scope (US1 only)**: After T001–T010, you have proven the stack works — GPU visible,
TRT-LLM imports, engine builds, text generates. This is independently valuable and the
most critical risk-reduction step.

**Increment 1 (+ US2)**: Add T011–T012 for a persistent, auto-restarting serving endpoint.
Now any HTTP client can reach the model.

**Increment 2 (+ US3)**: Add T013–T018 for the Rust integration client. This is the
weekend's capstone deliverable.

**Increment 3 (+ US4 + Polish)**: Add T019–T022 for docs handoff and acceptance sign-off.
Can be interleaved with US3 work.

**Suggested weekend order**: Phase 1 → Phase 2 → US1 (T007–T010) → US2 (T011–T012) →
US3 source files (T013–T018 in dependency order) → US4 + Polish

---

## Task Summary

| Phase | Tasks | Count | Parallel Opportunities |
|-------|-------|-------|----------------------|
| Phase 1: Setup | T001–T004 | 4 | All 4 parallel |
| Phase 2: Foundational | T005–T006 | 2 | Both parallel |
| Phase 3: US1 | T007–T010 | 4 | T007/T008/T009 parallel |
| Phase 4: US2 | T011–T012, T023 | 3 | T012, T023 parallel with T011 |
| Phase 5: US3 | T013–T018 | 6 | T016 [P] with T014; T015 (TDD stub) before T014 |
| Phase 6: US4 | T019–T020 | 2 | T020 parallel with T019 |
| Phase 7: Polish | T021–T022 | 2 | T022 parallel with T021 |
| **Total** | | **23** | |

**Per user story**:

| Story | Tasks | Independent Test |
|-------|-------|-----------------|
| US1 GPU + Engine Proof | T007–T010 (4 tasks + decision docs) | `just dev` → `nvidia-smi` → engine build → inference |
| US2 Persistent Serving | T011–T012, T023 (3 tasks) | `just compose-up` → `curl /v1/models` → reboot test |
| US3 Rust Client | T013–T018 (6 tasks) | `cargo test` + `cargo run -- smoke-test` |
| US4 Docs Handoff | T019–T020 (2 tasks) | Cold read of docs/setup.md + ansible-handoff.md |

**Format validation**: All 23 tasks follow `- [ ] TXXX [P?] [Story?] Description with file path` ✅
