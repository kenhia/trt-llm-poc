# Research: TRT-LLM POC End-to-End

**Phase**: 0 — Unknowns resolved before design  
**Feature**: `001-trt-llm-poc-end-to-end`  
**Date**: 2026-05-01

## Research Findings

### R-001: NGC Image Compatibility with Blackwell (SM 12.0)

**Decision**: Use `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13` (Track A) and
`nvcr.io/nvidia/tritonserver:25.05-trtllm-python-py3` (Track B) as pinned in the prep doc.

**Rationale**: TRT-LLM maintainers have confirmed FP4 on GeForce 50 series (SM 12.x) is
supported in recent versions. The NGC 25.05 release cycle targets current CUDA/driver stacks.
The host driver (595.58.03) and CUDA 13.2 are within the support envelope for these images.
The prep doc explicitly notes that older builds may lack SM 12.x support — using pinned
recent tags avoids this risk.

**Alternatives considered**:
- Building TRT-LLM from source: rejected — too much weekend time investment; NGC images
  ship coherent CUDA/TensorRT/TRT-LLM stacks that avoid version mismatch errors
- Using latest-tag images: rejected — tag drift can introduce breaking changes mid-POC;
  pinned tags are reproducible

**Risk**: NGC image tags may not yet be published or may have been pulled. If `1.3.0rc13`
is unavailable, fall back to the latest non-RC `release` tag for Track A, or the latest
`25.xx-trtllm-python-py3` for Track B. Document the actual tag used in `docs/ansible-handoff.md`.

---

### R-002: HTTP API Style — OpenAI-Compatible vs Triton Native v2

**Decision**: Use the **OpenAI-compatible HTTP API** (`POST /v1/chat/completions`,
`POST /v1/completions`) for all three capabilities.

**Rationale**: The prep doc explicitly recommends HTTP for Rust integration because
`reqwest` can call it directly. OpenAI-compatible endpoints (served by
`trtllm-serve` or an OpenAI-compat wrapper inside the Triton container) use a
well-known JSON schema that is simpler than the Triton v2 inference protocol binary/JSON
format. For a POC this eliminates the need to understand Triton's
`/v2/models/{model}/infer` protobuf-adjacent format.

The Triton + TRT-LLM image includes tooling to expose an OpenAI-compatible endpoint via
the `tensorrtllm_backend` example scripts. This is the documented "serving" path in
TRT-LLM community examples.

**Alternatives considered**:
- Triton native HTTP v2 (`/v2/models/{model}/infer`): rejected — more complex payload
  format (input/output tensor schemas per model), harder to type in Rust, no benefit
  for a single-node POC
- gRPC: rejected — adds `tonic`/protobuf dependency to Rust client; HTTP is sufficient
  for POC latency targets

**Contract**: See `contracts/http-api.md` for the request/response schema.

---

### R-003: Model Selection and VRAM Fit (32 GB RTX 5090)

**Decision**: Three models as specified in the prep doc, all with FP8 or INT4/FP4
quantization to ensure comfortable fit on 32 GB.

| Capability | Model | Quant | Est. VRAM |
|------------|-------|-------|-----------|
| Chat | Llama 3.1 8B Instruct | FP8 | ~10 GB |
| Code | Qwen2.5-Coder 7B Instruct | FP8 | ~9 GB |
| Caption | LLaVA-1.6 Mistral-7B (or LLaVA-NeXT 7B) | FP8 | ~12 GB |

All three models simultaneously would require ~31 GB — tight but feasible on 32 GB. For
the POC, load one model at a time unless memory allows concurrent serving.

**Rationale**: 8B-class models balance quality and VRAM footprint. Qwen2.5-Coder is a
widely-used coding benchmark leader at 7B size. LLaVA-1.6 / LLaVA-NeXT are in TRT-LLM's
supported model list and have documented multi-modal pipeline examples.

**Alternatives considered**:
- Larger models (13B+): rejected — risk of OOM on 32 GB without aggressive quantization
- Smaller models (3B-): rejected — quality drop significant for chat/code; not representative
- Running all three concurrently in one Triton instance: defer — evaluate after P1/P2 work;
  start with one model per session for POC simplicity

**Risk**: LLaVA multi-modal pipeline in TRT-LLM requires a separate visual encoder engine
alongside the language model engine. If this proves overly complex, fall back to a
text-only model with a stub image endpoint and document the blocker.

---

### R-004: Autostart Method — Docker Compose vs systemd

**Decision**: Docker Compose with `restart: unless-stopped` as the autostart mechanism.

**Rationale**: Compose is simpler to manage, portable between hosts, and aligns with
the `justfile` recipe model (all lifecycle commands delegate to `docker compose`).
`restart: unless-stopped` provides reboot persistence without needing systemd unit
management. Docker daemon itself starts on boot via systemd (already configured).

**Alternatives considered**:
- systemd unit wrapping `docker run`: documented in `docs/decisions/001-compose-vs-systemd.md`
  and in the prep doc for reference, but not implemented — adds complexity with no POC benefit
- Kubernetes/Docker Swarm: rejected — massively over-engineered for a single-GPU local POC

---

### R-005: Rust Client Architecture

**Decision**: Single Rust binary with a library-style internal structure (no external crate
publication). `LlmClient` as a trait with one concrete implementation backed by `reqwest`.
Routing logic lives in `capability.rs` and is fully unit-testable without a network.

**Rationale**: Keeps the crate simple. The trait boundary allows future swap to a different
backend (e.g., gRPC, different provider) without changing the CLI. Per-request logging via
`tracing` (structured, to stdout) satisfies the observability requirement with minimal code.

**Crates selected**:
- `reqwest` with `json` feature — HTTP client
- `serde` + `serde_json` — payload serialization  
- `base64` — encode image bytes for caption requests
- `tokio` — async runtime
- `tracing` + `tracing-subscriber` — structured per-request logging (preferred over `log`
  for structured fields; `tracing-subscriber` gives JSON output with one line of setup)
- `uuid` — request ID generation

**Alternatives considered**:
- `ureq` (sync HTTP): rejected — less ergonomic with async ecosystem; `reqwest` is the
  dominant choice for async Rust HTTP
- Custom logging: rejected — `tracing` is idiomatic, adds no meaningful complexity

---

### R-006: Environment / Secrets Handling

**Decision**: `.env` file at `$TRTLLM_HOME/.env` loaded by `justfile` via `set dotenv-load`.
`HF_TOKEN` and `TRTLLM_HOME` are the only required secrets/config. A `.env.example` is
committed; `.env` is gitignored.

**Rationale**: Simplest approach that avoids committing credentials. No secret manager
needed for a local POC.

**Alternatives considered**:
- Environment variables set system-wide: works but not reproducible in a fresh shell
- HashiCorp Vault / sops: rejected — over-engineered for a local POC

---

### R-007: Engine Build Workflow (Track A)

**Decision**: Engines are built interactively inside the Track A container using the
`trtllm-build` CLI (provided by the NGC dev image). Built engines are persisted to
`$TRTLLM_HOME/engines/` via the bind mount and reused by Track B (Triton) without rebuilding.

**Rationale**: Engine build is a one-time step per model. Persisting to host filesystem
avoids rebuilding on container restart. Track B Triton container mounts the same directory,
so engines flow directly from Track A → Track B.

**Build command pattern**:
```bash
trtllm-build \
  --checkpoint_dir /workspace/trtllm/models/<model> \
  --output_dir /workspace/trtllm/engines/<model> \
  --gemm_plugin float16 \
  --max_batch_size 1
```
Exact flags depend on the model; `--gemm_plugin` and quantization args are model-specific.
Document actual flags used in `docs/setup.md`.

## Summary of Resolved Unknowns

| Unknown | Resolved By | Decision |
|---------|-------------|----------|
| NGC image SM 12.0 support | R-001 | Use pinned 25.05 tags; document actual tag in handoff |
| HTTP API format for Rust | R-002 | OpenAI-compatible (`/v1/chat/completions`) |
| Model selection + VRAM fit | R-003 | Llama 3.1 8B / Qwen2.5-Coder 7B / LLaVA-1.6 7B at FP8 |
| Autostart mechanism | R-004 | Docker Compose `restart: unless-stopped` |
| Rust client architecture | R-005 | `LlmClient` trait, `reqwest`, `tracing` |
| Secrets/env handling | R-006 | `.env` at `$TRTLLM_HOME`, gitignored |
| Engine build workflow | R-007 | `trtllm-build` in Track A, engines persisted to host |

**All NEEDS CLARIFICATION items resolved. Ready for Phase 1.**
