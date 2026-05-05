# TRT-LLM POC — End-to-End on RTX 5090

Proof-of-concept that runs TensorRT-LLM inference end-to-end on a single RTX 5090
host: build a model engine inside the dev container (Track A), serve it via Triton
(Track B), and call it from a Rust client that routes requests to chat, code, or
caption models based on request content.

> **Note:** This is a POC for learning how to use TensorRT-LLM. It makes
> assumptions about my specific hardware and environment. If you are doing your
> own initial investigation into TRT-LLM, this may give you some pointers, but
> you will almost certainly need to adjust paths and configuration for your setup.

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

## Models

| Pipeline | Model | Capability |
|----------|-------|------------|
| `ensemble_qwen` | Qwen2.5-Coder 7B | Code generation |
| `ensemble_llama` | Llama 3.1 8B Instruct | Chat |
| `ensemble_vision` | LLaVA 1.5 7B | Image captioning |

Only one model pipeline fits in VRAM at a time (~15 GB each on 32 GB).
Use `just models-load <pipeline>` / `just models-unload <pipeline>` to swap.

## Repository Layout

```
.
├── compose.yaml                    # Triton serving stack (Track B)
├── justfile                        # Task runner (just dev, just compose-up, …)
├── scripts/
│   ├── setup-model-repo.sh         # Create pipeline dirs from container templates
│   ├── start-triton.sh             # Triton entrypoint (EXPLICIT model control)
│   ├── triton_wrapper.py           # Wrapper: OpenAI frontend + KServe
│   └── vision-smoke-test.py        # Binary-protocol vision smoke test
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
└── specs/                          # Feature specs (one dir per sprint)
```

## Key Commands

```sh
just dev              # Enter TRT-LLM dev container (Track A)
just compose-up       # Start Triton serving stack (Track B)
just compose-logs     # Tail Triton logs
just models-load qwen # Load a pipeline into GPU memory
just models-unload qwen
just models-status    # Show loaded/unloaded state of all pipelines
just smoke-all        # Sequential load→test→unload for all three pipelines
cargo test            # Run all tests (7 routing unit tests)
```

## Decisions

- [001 — Compose vs Systemd](docs/decisions/001-compose-vs-systemd.md)
- [002 — OpenAI-Compat vs Triton-Native](docs/decisions/002-openai-compat-vs-triton-native.md)
- [003 — Model Selection](docs/decisions/003-model-selection.md)
