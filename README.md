# TRT-LLM POC — End-to-End on RTX 5090

Proof-of-concept that runs TensorRT-LLM inference end-to-end on a single RTX 5090
host: build a model engine inside the dev container (Track A), serve it via Triton
(Track B), and call it from a Rust client that routes requests to chat, code, or
caption models based on request content.

## Hardware & Stack

| Layer | Component |
|-------|-----------|
| GPU | NVIDIA RTX 5090, 32 GB VRAM |
| OS | Ubuntu 24.04 LTS |
| Driver | 595.58.03 |
| Inference runtime | TensorRT-LLM `1.3.0rc13` |
| Serving | Triton Server `25.05-trtllm-python-py3` |
| Client | Rust 1.95 (`reqwest` + `tokio`) |
| Orchestration | Docker Compose + `just` |

## Quick Start

See [quickstart.md](specs/001-trt-llm-poc-end-to-end/quickstart.md) for the fast path.

For the full step-by-step guide (toolkit install, engine build, Triton setup):
[docs/setup.md](docs/setup.md).

## Acceptance Criteria

| ID | Criterion |
|----|-----------|
| SC-001 | GPU visible inside Docker container within 30 s of start (`nvidia-smi`) |
| SC-002 | First inference response from Triton in ≤ 60 s (cold-start acceptable) |
| SC-003 | Triton reachable on `localhost:8000` within 2 min of reboot — no manual action |
| SC-004 | All three capabilities (chat, code, caption) return non-empty coherent responses via Rust client |
| SC-005 | `cargo test` passes with zero failures |
| SC-006 | `docs/setup.md` + `docs/ansible-handoff.md` cover 100% of host changes — no undocumented steps |

## Repository Layout

```
.
├── compose.yaml                    # Triton serving stack (Track B)
├── justfile                        # Task runner (just dev, just compose-up, …)
├── rust-client/                    # Rust inference client
│   ├── src/
│   │   ├── capability.rs           # Request routing (Chat / Code / Caption)
│   │   ├── client.rs               # HTTP client (reqwest → Triton)
│   │   ├── logging.rs              # Structured JSON logging (tracing)
│   │   ├── main.rs                 # Binary — `trtllm-client smoke-test`
│   │   ├── models.rs               # Request / response types
│   │   └── lib.rs                  # Library root for integration tests
│   └── tests/routing.rs            # Routing unit tests
├── triton-model-repo/              # Triton model config templates
│   ├── llama3-chat/config.pbtxt
│   ├── qwen-coder/config.pbtxt
│   └── llava-caption/config.pbtxt
├── docs/
│   ├── setup.md                    # Complete setup guide (§0-§6)
│   └── ansible-handoff.md          # All host changes for ansible-k automation
└── specs/001-trt-llm-poc-end-to-end/
    ├── spec.md
    ├── plan.md
    ├── tasks.md
    ├── quickstart.md
    ├── data-model.md
    ├── research.md
    └── decisions/
```

## Key Commands

```sh
just dev          # Enter TRT-LLM dev container (Track A)
just compose-up   # Start Triton serving stack (Track B)
just compose-logs # Tail Triton logs
just ps           # Show running containers
cargo build       # Build Rust client
cargo test        # Run all tests (7 routing unit tests)
cargo run -- smoke-test  # Run live smoke test against Triton
```

## Decisions

- [001 — Container Strategy](specs/001-trt-llm-poc-end-to-end/decisions/001-container-strategy.md)
- [002 — Autostart Method](specs/001-trt-llm-poc-end-to-end/decisions/002-autostart-method.md)
- [003 — Model Selection](specs/001-trt-llm-poc-end-to-end/decisions/003-model-selection.md)
