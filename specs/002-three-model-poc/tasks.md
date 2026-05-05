# Tasks: Three-Model POC

**Input**: Design documents from `specs/002-three-model-poc/`
**Prerequisites**: plan.md ✅ · spec.md ✅ · research.md ✅ · data-model.md ✅ · contracts/interfaces.md ✅ · quickstart.md ✅

**Tests**: Routing unit tests (`cargo test`) are updated where model IDs change.
Manual smoke-tests serve as acceptance criteria per SC-001–SC-007.
No new test infrastructure — existing `tests/routing.rs` is extended in place.

**Organization**: Tasks are grouped by user story. US1 (model-control) is foundational
infrastructure that unlocks US2 and US3.

---

## Phase 1: Setup (Serving Infrastructure Changes)

**Purpose**: Write and wire the `triton_wrapper.py` and update container config so
Triton can start in EXPLICIT model-control mode with the KServe control plane active.
No GPU required for this phase.

- [X] T001 Write `scripts/triton_wrapper.py` — monkey-patch `tritonserver.Server.__init__` to inject `model_control_mode=ModelControlMode.EXPLICIT` and `startup_models` from env, then call `main.main()` (see research.md R1); include a `PIPELINE_MODELS` dict mapping short pipeline names (`qwen`, `llama`, `vision`) to their full list of model directory names so `DEFAULT_MODEL` can be expanded into a valid `startup_models` list
- [X] T002 Update `scripts/start-triton.sh` — replace `exec python3 main.py` with `exec python3 /triton_wrapper.py`; add `--enable-kserve-frontends` to the arg list
- [X] T003 [P] Update `compose.yaml` — add `KSERVE_HTTP_PORT: 8000`, `DEFAULT_MODEL`, `TRITON_STARTUP_MODELS` env vars; bind-mount `scripts/triton_wrapper.py:/triton_wrapper.py:ro`; expose port 8000
- [X] T004 [P] Update `.env.example` — document `DEFAULT_MODEL=qwen`, `TRITON_STARTUP_MODELS`, `KSERVE_HTTP_PORT` vars with comments

**Checkpoint**: `triton_wrapper.py` exists, `start-triton.sh` references it, compose.yaml is updated. No running containers needed yet.

---

## Phase 2: Foundational (Qwen Pipeline Rename + Verify)

**Purpose**: Rename the existing qwen pipeline from bare names (`ensemble`, `preprocessing`, …)
to suffixed names (`ensemble_qwen`, `preprocessing_qwen`, …) so all three pipelines can
coexist in `model_repo/`. Update the Rust routing test. Verify Triton still serves qwen
before building new engines. **Blocks all user stories.**

**⚠️ CRITICAL**: No user story work can begin until this phase is complete and verified.

- [X] T005 Write `scripts/setup-model-repo.sh` — parameterised script that: (a) for `qwen` renames existing `model_repo/{component}` dirs to `{component}_qwen` and updates all `config.pbtxt` `name:` fields and cross-reference `model_name:` values; (b) for `llama`/`vision` creates new pipeline dirs by copying the appropriate template from inside the container (`/app/all_models/inflight_batcher_llm/` or `/app/all_models/multimodal/`) and filling all `${variable}` placeholders
- [X] T006 Run `just setup-model-repo qwen` (or the script directly) to rename qwen pipeline dirs at `$TRTLLM_HOME/model_repo/` — verify all five dirs (`preprocessing_qwen`, `tensorrt_llm_qwen`, `postprocessing_qwen`, `ensemble_qwen`, `tensorrt_llm_bls_qwen`) exist with correct `name:` in each `config.pbtxt`
- [X] T006a Update `rust-client/src/capability.rs` — change `Code` arm `model_id()` from `"ensemble"` to `"ensemble_qwen"` (FR-010; must precede `cargo test` at T009)
- [X] T007 Update `rust-client/tests/routing.rs` — change expected model ID for `Code` routing assertion from `"ensemble"` to `"ensemble_qwen"`; `Chat` assertion remains `"ensemble_qwen"` for now (updated again in T020)
- [X] T008 [P] Update `justfile` — add `setup-model-repo pipeline` recipe that calls `scripts/setup-model-repo.sh`; update `compose-install` to copy entire `scripts/` directory (not just `start-triton.sh`) to `$TRTLLM_HOME/scripts/`
- [X] T009 Run `cargo test` — all tests in `rust-client/tests/routing.rs` pass with updated model ID
- [X] T010 Run `just compose-install && just compose-up`; confirm Triton starts in EXPLICIT mode (wrapper log line), KServe HTTP active on port 8000; run `curl -s localhost:9000/v1/models` and `just models-status` (stub); load qwen with raw curl: `curl -X POST localhost:8000/v2/repository/models/ensemble_qwen/load`; confirm `curl localhost:9000/v1/chat/completions` returns a response

**Checkpoint**: Qwen pipeline renamed, serves under `ensemble_qwen`, EXPLICIT mode confirmed, `cargo test` green.

---

## Phase 3: User Story 1 — GPU Reclaim via Model Load/Unload (P1) 🎯 MVP

**Goal**: `just models-load/unload/status` recipes work against the live Triton KServe API.
VRAM is fully reclaimed on unload and restored on load. Container stays up throughout.

**Independent Test**: Unload qwen → `nvidia-smi` shows ≥20 GB freed → reload qwen → `curl /v1/chat/completions` responds. Container never restarted.

- [X] T011 [US1] Add `just models-load pipeline` recipe to `justfile` — posts load request to `localhost:8000/v2/repository/models/{component}_{pipeline}/load` for each of the 5 (or 4 vision) models in dependency order; prints result per model; exits non-zero on first failure
- [X] T012 [US1] Add `just models-unload pipeline` recipe to `justfile` — posts unload in reverse dependency order; idempotent (ignore 404)
- [X] T013 [US1] Add `just models-status` recipe to `justfile` — posts `{"ready": false}` to `localhost:8000/v2/repository/index`; formats output grouped by pipeline showing READY / not-loaded state
- [X] T014 [US1] Add `just smoke-all` recipe to `justfile` — sequential: load qwen → `cargo run -- smoke-test` Code-only → unload qwen → load llama → smoke-test Chat-only → unload llama → load vision → smoke-test Caption-only → unload vision
- [X] T015 [US1] Manual acceptance: run `just models-unload qwen`; confirm `nvidia-smi` VRAM drops ≥20 GB (SC-001); run `just models-load qwen`; confirm inference within 60 s (SC-002); confirm `just models-status` reflects each state change (SC-003)

**Checkpoint**: SC-001, SC-002, SC-003 all verified. US1 complete.

---

## Phase 4: User Story 2 — Dedicated Chat Engine (Llama 3.1 8B) (P2)

**Goal**: Llama 3.1 8B Instruct TRT-LLM engine built, `model_repo/` has a working
`ensemble_llama` pipeline, and `cargo run -- smoke-test` Chat routes to it.

**Independent Test**: `just models-load llama` → `just models-status` shows `ensemble_llama` READY → `curl localhost:9000/v1/chat/completions` with `model: "ensemble_llama"` returns a coherent response.

- [X] T016 [US2] Add `just build-llama` recipe to `justfile` — launches `docker run --gpus all -v $TRTLLM_HOME:/workspace/trtllm nvcr.io/nvidia/tritonserver:25.05-trtllm-python-py3 bash -c "..."` that runs: `huggingface-cli download`, `convert_checkpoint.py`, `trtllm-build`; engine output: `$TRTLLM_HOME/engines/llama/`
- [X] T017 [US2] Run `just build-llama` (GPU required, ~30–60 min) — confirm `$TRTLLM_HOME/engines/llama/rank0.engine` and `config.json` exist; accept SC-004
- [X] T018 [US2] Run `just setup-model-repo llama` — creates five `*_llama` dirs in `$TRTLLM_HOME/model_repo/` from `inflight_batcher_llm` template; fills `engine_dir`, `tokenizer_dir`, `name:`, and cross-reference `model_name:` values with `_llama` suffix
- [X] T019 [US2] Update `rust-client/src/capability.rs` — change `Chat` arm `model_id()` from `"ensemble_qwen"` to `"ensemble_llama"`
- [X] T020 [US2] Update `rust-client/tests/routing.rs` — change expected model ID for `Chat` routing assertion to `"ensemble_llama"`
- [X] T021 [US2] Run `cargo test` — all routing assertions pass
- [X] T022 [US2] Manual acceptance: `just models-load llama`; `curl -X POST http://localhost:9000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"ensemble_llama","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":64}'` returns a coherent response within 60 s (SC-004); Rust log shows `ensemble_llama` as model ID

**Checkpoint**: SC-004 verified. `cargo test` green. US2 complete.

---

## Phase 5: User Story 3 — Vision/Caption Engine (LLaVA 1.5 7B) (P3)

**Goal**: LLaVA 1.5 7B TRT-LLM engine built (vision encoder + LLM), `model_repo/` has
a working `ensemble_vision` pipeline, and `cargo run -- smoke-test` Caption routes to it
and returns a non-empty image description.

**Independent Test**: `just models-load vision` → `just models-status` shows `ensemble_vision` READY → `curl` with base64 image returns a text description (not HTTP 400).

**Note**: LLaVA 1.5 7B (`llava-hf/llava-1.5-7b-hf`) is confirmed `[x]` supported in TRT-LLM 0.19.0 C++ runtime (research.md R4). LLaVA-NeXT is NOT supported.

- [X] T023 [US3] Add `just build-llava` recipe to `justfile` — docker run that executes: `huggingface-cli download llava-hf/llava-1.5-7b-hf`, `build_multimodal_engine.py --model_type llava` (vision encoder), `convert_checkpoint.py` (LLaMA backbone), `trtllm-build --max_multimodal_len 2048`; outputs to `$TRTLLM_HOME/engines/llava/`
- [X] T024 [US3] Run `just build-llava` (GPU required, ~20–40 min) — confirm `$TRTLLM_HOME/engines/llava/vision/` (vision encoder .engine) and `$TRTLLM_HOME/engines/llava/llm/rank0.engine` exist
- [X] T025 [US3] Run `just setup-model-repo vision` — creates `multimodal_encoders` (no suffix — only one vision pipeline), `tensorrt_llm_vision`, `postprocessing_vision`, `ensemble_vision` dirs from `/app/all_models/multimodal/` template; fills engine paths and `name:` fields with `_vision` suffix for the three LLM-side components; `multimodal_encoders` keeps bare name; updates cross-references in `ensemble_vision/config.pbtxt` to reference `multimodal_encoders` (unsuffixed)
- [X] T026 [US3] Update `rust-client/src/capability.rs` — change `Caption` arm `model_id()` from `"ensemble_qwen"` (Sprint 001 placeholder) to `"ensemble_vision"`; update image encoding to use `image_url` content type in the request body
- [X] T027 [US3] Update `rust-client/src/main.rs` — change Caption failure from non-fatal warning to fatal: Caption non-response now sets `all_ok = false` and contributes to non-zero exit code (SC-006 requires all three pass)
- [X] T028 [US3] Update `rust-client/tests/routing.rs` — change expected model ID for `Caption` routing assertion to `"ensemble_vision"`
- [X] T029 [US3] Run `cargo test` — all routing assertions pass
- [X] T030 [US3] Manual acceptance: `just models-load vision`; LLaVA 1.5 requires binary HTTP extension (FP16 pixel values); `scripts/vision-smoke-test.py` uses Triton native port 8000 binary protocol → `ensemble_vision` returns a text description in 0.8 s (SC-005 verified via `just smoke-vision`). OpenAI `/v1/chat/completions` does not pass `image_url` for model_type=llava (as noted in research.md R6).

**Checkpoint**: SC-005 verified. `cargo test` green. US3 complete.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Documentation complete, smoke-all works end-to-end, all SCs verified.

- [X] T031 [P] Update `docs/setup.md` §4 (convert + trtllm-build commands, HF_TOKEN requirement, expected output paths)
- [X] T032 [P] Update `docs/setup.md` §4 — §4.3 LLaVA 1.5 7B engine build procedure added (build_multimodal_engine.py + LLM convert + trtllm-build, expected output paths, model_type llava note)
- [X] T033 [P] Update `docs/setup.md` §4 — §4.5 multi-pipeline layout + §4.6 setup-model-repo procedure added (naming convention table from data-model.md, config cross-reference summary)
- [X] T034 Update `docs/setup.md` §5 — explicit model-control mode startup documented; pipeline entry points use `_qwen`/`_llama`/`_vision` names
- [X] T035 Update `docs/setup.md` §5 — §5.3 model-control workflow with `just models-load/unload/status`, VRAM budget, DEFAULT_MODEL added
- [X] T036 Final validation — run SC-001 through SC-007 checklist: unload/load/VRAM (SC-001–003); llama loads (SC-004); vision inference (SC-005); `just smoke-all` exits 0 (SC-006); docs complete (SC-007)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately, no GPU needed
- **Foundational (Phase 2)**: Depends on Phase 1 completion — **BLOCKS all user stories**
- **US1 (Phase 3)**: Depends on Phase 2 (Triton must be in EXPLICIT mode with KServe active)
- **US2 (Phase 4)**: Depends on Phase 2 (renamed pipeline must work); US1 must be complete to use `just models-load` recipes
- **US3 (Phase 5)**: Depends on Phase 2 + US1; sequential after US2 (one GPU, VRAM budget)
- **Polish (Phase 6)**: Depends on all user stories complete

### User Story Dependencies

- **US1 (P1)**: Starts after Foundational — independent, no engine build required
- **US2 (P2)**: Starts after Foundational + US1 (requires `just models-load` for testing)
- **US3 (P3)**: Starts after US1 (requires model-control), sequential after US2 (VRAM)

### Within Each User Story

- Scripts/config before justfile recipes
- Justfile recipes before manual acceptance tests
- Rust code changes → `cargo test` → manual smoke

### Parallel Opportunities (within phases)

- T003 and T004 (Phase 1) — parallel: `compose.yaml` and `.env.example` are independent files
- T007 and T008 (Phase 2) — parallel: routing test update and justfile update are independent files
- T031, T032, T033 (Phase 6) — parallel: each doc section is independent

---

## Parallel Example: Phase 1

```text
# These two tasks touch different files and can run simultaneously:
T003  Update compose.yaml
T004  Update .env.example
```

## Parallel Example: Phase 6 Polish

```text
# All three doc sections are independent:
T031  docs/setup.md §4.2 (Llama build)
T032  docs/setup.md §4.3 (LLaVA build)
T033  docs/setup.md §4.4 (multi-pipeline layout)
```

---

## Implementation Strategy

### MVP First (US1 Only — Immediately Useful)

1. Complete Phase 1: Setup (wrapper + container config)
2. Complete Phase 2: Foundational (rename qwen, verify still works)
3. Complete Phase 3: US1 (justfile recipes)
4. **STOP and VALIDATE**: `just models-unload qwen` frees VRAM → `just models-load qwen` restores inference
5. **Value delivered**: GPU can be reclaimed without container restart — unblocks adjacent GPU projects today

### Full Sprint Delivery

1. Phase 1 + Phase 2 → Foundation ready, serving verified
2. Phase 3 (US1) → Model-control working, SC-001–003 met
3. Phase 4 (US2) → Llama engine + pipeline, SC-004 met
4. Phase 5 (US3) → LLaVA engine + pipeline, SC-005–006 met
5. Phase 6 (Polish) → SC-007 met, all six SCs green

### Key Constraints Reminder

- All `trtllm-build` commands run **inside** the Triton 25.05 container (TRT-LLM 0.19.0)
- Only one pipeline loads at a time (32 GB VRAM budget)
- `cargo test` must be green before any merge
- LLaVA 1.5 (not 1.6 / NeXT) — NeXT is unsupported in C++ runtime (research.md R4)
