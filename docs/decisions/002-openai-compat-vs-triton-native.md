# Decision 002: OpenAI-Compatible API vs Triton Native v2

**Date**: 2026-05-01  
**Status**: Decided  
**Research ref**: R-002 (research.md)

## Context

The Rust client needs to call the Triton serving container over HTTP. Triton exposes two
HTTP API flavors:

1. **Triton native HTTP v2** — `POST /v2/models/{model}/infer`, with input/output tensors
   described by name, shape, and datatype in the JSON body
2. **OpenAI-compatible API** — `POST /v1/chat/completions`, with a messages array and
   standard generation parameters

The Triton + TRT-LLM NGC image includes tooling to expose an OpenAI-compatible endpoint
via the `tensorrtllm_backend` example scripts.

## Options Considered

### Option A: OpenAI-Compatible API — `POST /v1/chat/completions` (chosen)

- Well-known JSON schema; `reqwest` + `serde` can serialize/deserialize with minimal code
- Same endpoint handles all three capabilities (Chat, Code, Caption) — model routing via
  the `model` field
- No per-model tensor descriptor knowledge required
- Community and NVIDIA examples document this path for TRT-LLM serving

### Option B: Triton native HTTP v2 — `POST /v2/models/{model}/infer`

- Lower-level; requires knowing each model's input/output tensor names, shapes, and datatypes
- More control over batching and tensor layout — irrelevant for POC single-request use
- Harder to type in Rust (per-model struct for each request format)
- No benefit for the POC's single-GPU, single-request smoke-test

### Option C: gRPC

- Would require `tonic` + protobuf definitions in the Rust crate
- ~Same capability as native HTTP v2 for this use case, but adds significant dependency weight
- Not worth the complexity for a POC

## Decision

**Option A — OpenAI-compatible API** was chosen because it is the simplest path from
Rust `reqwest` to a working inference call, and the Triton + TRT-LLM image supports it.

See `contracts/http-api.md` for the full request/response schema.
