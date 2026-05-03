# Feature Specification: TensorRT-LLM POC — End-to-End on RTX 5090

**Feature Branch**: `001-trt-llm-poc-end-to-end`  
**Created**: 2026-05-01  
**Status**: Draft  
**Input**: [TRT-LLM POC Weekend Spec Prep](/home/ken/obsidian/gratch/Computers%20and%20Services/multiae-viae/TRT-LLM%20POC%20Weekend%20Spec%20Prep.md)

## User Scenarios & Testing

### User Story 1 — GPU + Engine Proof (Priority: P1)

As the developer, I want to confirm that TensorRT-LLM works end-to-end inside a
Docker container on the RTX 5090, so I know the hardware/software stack is functional
before investing time in serving or client code.

**Why this priority**: Everything downstream depends on the GPU being reachable and a
model engine being buildable. If this fails, the POC is blocked.

**Independent Test**: Run the Track A dev container interactively, confirm `nvidia-smi`
shows the RTX 5090, import `tensorrt_llm`, load a small instruct model, and generate
a text response. Delivers standalone proof the stack works.

**Acceptance Scenarios**:

1. **Given** the Track A dev container is running, **When** `nvidia-smi` is executed
   inside the container, **Then** the RTX 5090 is listed with correct memory and driver
   info and no errors
2. **Given** the Track A dev container is running, **When** `python -c "import tensorrt_llm; print(tensorrt_llm.__version__)"` is executed, **Then** a version string is printed
   without errors
3. **Given** a small instruct model (e.g., Llama 3.x 8B Instruct) is available,
   **When** a text generation request is issued inside the container, **Then** a
   coherent response is produced without CPU-fallback warnings or out-of-memory errors

---

### User Story 2 — Persistent Serving Endpoint (Priority: P2)

As the developer, I want the Triton + TRT-LLM serving container to start automatically
after a reboot and expose an HTTP endpoint on the host, so downstream Rust code can
reach it without manual intervention.

**Why this priority**: The Rust client cannot be built or tested until there is a
stable, host-reachable HTTP endpoint. Autostart is required for handoff.

**Independent Test**: Reboot the host (or stop/start Docker), confirm the container
restarts and responds on `localhost:8000`. Delivers a stable target for any HTTP client.

**Acceptance Scenarios**:

1. **Given** `compose.yaml` is in place with `restart: unless-stopped`, **When** the
   host is rebooted, **Then** the Triton container comes back up and responds to an
   HTTP health check on port 8000 without manual intervention
2. **Given** the Triton container is running, **When** a minimal well-formed inference
   request is sent to the HTTP endpoint, **Then** a valid response payload is returned
3. **Given** a `just` recipe is defined for container lifecycle, **When** `just
   compose-up`, `just compose-down`, and `just compose-logs` are executed, **Then** each
   produces the expected Docker Compose result without errors

---

### User Story 3 — Rust Client Exercises Three Capabilities (Priority: P3)

As the developer, I want a minimal Rust client that routes requests to the correct
model for each of three capabilities — general chat, coding assistance, and image
captioning — so I have a working integration skeleton for the future agentic controller.

**Why this priority**: This is the primary deliverable that validates Rust→service
integration. The other stories are prerequisites; this is the proof of end-to-end value.

**Independent Test**: Run `cargo test` and a manual smoke test against the live endpoint
for each of the three capabilities. Delivers a working Rust client crate.

**Acceptance Scenarios**:

1. **Given** the serving endpoint is reachable, **When** a chat request is sent via the
   Rust client, **Then** a plausible conversational response is returned and logged
2. **Given** the serving endpoint is reachable, **When** a coding request is sent via
   the Rust client, **Then** a code snippet response is returned and logged
3. **Given** the serving endpoint is reachable and a test image is available, **When**
   a caption request (with image payload) is sent via the Rust client, **Then** a
   coherent textual description of the image is returned and logged
4. **Given** the Rust client is built, **When** `cargo test` is run, **Then** all tests
   pass and request routing logic (chat/code/caption selection) is covered

---

### User Story 4 — Lifecycle & Documentation Handoff (Priority: P4)

As the developer (and future ansible-k maintainer), I want all host-level setup steps,
Docker run flags, tunable knobs, and `justfile` recipes documented so the POC can be
reproduced from scratch or handed off to automation.

**Why this priority**: Without this, the weekend's work is not reproducible. Low
implementation effort with high long-term value.

**Independent Test**: A fresh reader (or agent) can follow `docs/setup.md` and
`docs/ansible-handoff.md` to replicate the environment without asking questions.

**Acceptance Scenarios**:

1. **Given** the POC is complete, **When** `docs/setup.md` is read cold, **Then** every
   step required to go from a bare host to a running Triton endpoint is documented with
   commands and "why" commentary
2. **Given** any new tool was installed on the host during the POC, **When**
   `docs/ansible-handoff.md` is read, **Then** it lists the tool name, install command,
   version, and purpose in a form suitable for promotion to ansible-k
3. **Given** any non-obvious Docker flag or tunable was used, **When** the relevant
   `justfile` or `compose.yaml` is read, **Then** inline comments explain each setting

---

### Edge Cases

- What happens when GPU memory is exhausted (model too large for 32 GB)?
- What if the NGC image tag does not support SM 12.0 (Blackwell / RTX 5090)?
- What if `restart: unless-stopped` restarts the container before the engine is
  fully loaded, causing a crash loop?
- What if a gated Hugging Face model requires an HF token that is not set in the
  environment?
- What if the image payload for captioning exceeds a reasonable size limit?

## Requirements

### Functional Requirements

- **FR-001**: The Track A dev container MUST make the RTX 5090 GPU visible and usable
  for inference (confirmed via `nvidia-smi` inside the container)
- **FR-002**: The Track A container MUST include a working TensorRT-LLM installation
  at a version compatible with Blackwell (SM 12.0) and FP4
- **FR-003**: The user MUST be able to build a TRT-LLM engine for at least one small
  instruct model inside the Track A container and generate text from it
- **FR-004**: The Track B Triton container MUST expose an HTTP endpoint on the host
  that accepts inference requests and returns responses
- **FR-005**: The serving container MUST restart automatically after a host reboot
  without manual intervention
- **FR-006**: The `justfile` MUST provide recipes for: `pull`, `dev` (Track A shell),
  `compose-up`, `compose-down`, `compose-logs`, `compose-restart`, `purge`, `gpu`, `ps`
- **FR-007**: The Rust client MUST route requests to one of three model capabilities:
  `Chat`, `Code`, `Caption`; routing MUST be deterministic based on request content
- **FR-008**: The Rust client MUST call the serving endpoint over HTTP and return a
  structured response for each capability
- **FR-009**: The Rust client MUST log, per request: request id, latency, model/capability
  chosen, and token counts if available
- **FR-010**: The Rust client MUST include tests for routing logic; `cargo test` MUST
  pass before the story is considered done
- **FR-011**: All host-level changes (tool installs, Docker configuration, systemd or
  compose setup) MUST be documented in `docs/ansible-handoff.md`
- **FR-012**: A `docs/setup.md` file MUST document the end-to-end setup procedure with
  commands and rationale commentary

### Key Entities

- **Capability**: An enumerated inference mode — `Chat`, `Code`, or `Caption` — that
  maps to a specific model and routing rule
- **ModelRoute**: Binding of a `Capability` to a serving endpoint URL and model
  identifier; configured at client startup
- **InferenceRequest**: The payload sent to the serving endpoint — includes messages
  (OpenAI-style role/content array), optional image (base64), capability hint, and
  generation parameters (`max_tokens`, `temperature`)
- **InferenceResponse**: The payload returned — includes response text, capability used,
  latency, and usage statistics
- **Container Lifecycle**: The set of Docker/Compose states (running, stopped, restarting)
  and the `justfile` recipes that manage transitions between them

## Success Criteria

### Measurable Outcomes

- **SC-001**: GPU is confirmed visible inside the Docker container within 30 seconds of
  container start (verified by `nvidia-smi`)
- **SC-002**: A text generation request sent to the Track B Triton serving endpoint
  returns a response within 60 seconds on first inference (cold-start on first model
  load acceptable for POC; measured during `cargo run -- smoke-test`)
- **SC-003**: The serving container is reachable on `localhost:8000` within 2 minutes of
  a host reboot without any manual action
- **SC-004**: All three capabilities (chat, code, caption) return non-empty, coherent
  responses when tested via the Rust client against the live endpoint
- **SC-005**: `cargo test` passes with zero failures for the Rust client crate
- **SC-006**: `docs/setup.md` and `docs/ansible-handoff.md` together cover 100% of the
  host changes made during the POC — no undocumented steps remain

## Assumptions

- The host NVIDIA driver (595.58.03) and CUDA 13.2 stack are already installed and
  functional; `nvidia-smi` works on the host before any Docker work begins
- Docker and the NVIDIA Container Toolkit are already installed on the host and the
  `nvidia` runtime is available to Docker
- The NGC image tags pinned in the prep document (`tensorrt-llm/release:1.3.0rc13`,
  `tritonserver:25.05-trtllm-python-py3`) are available and support Blackwell (SM 12.0)
- The RTX 5090's 32 GB VRAM is sufficient for the selected models with quantization;
  the prep document's model choices (Llama 3.x 8B, Qwen2.5-Coder 7B, LLaVA small) are
  treated as the default targets
- For gated models, an HF token is available in the environment; managing the token
  itself is out of scope
- A single-node, single-GPU setup is assumed; multi-GPU or distributed inference is
  out of scope for this POC
- The Rust client targets the Track B Triton HTTP endpoint; Track A is used only for
  engine validation and is not a long-lived service
- Autostart via Docker Compose `restart: unless-stopped` is the chosen method (over
  systemd); the systemd option from the prep document is documented but not implemented
