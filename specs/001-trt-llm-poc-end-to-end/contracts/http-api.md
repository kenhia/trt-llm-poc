# HTTP API Contract: LLM Serving Endpoint

**Feature**: `001-trt-llm-poc-end-to-end`  
**Date**: 2026-05-01  
**Decision**: R-002 — OpenAI-compatible HTTP API (see `research.md`)

## Overview

The Rust client communicates with the Triton + TRT-LLM serving container via an
OpenAI-compatible HTTP API. All three capabilities (Chat, Code, Caption) use the
same endpoint schema. The Triton container exposes this interface on port 8000.

**Base URL**: `http://localhost:8000` (configurable via `TRTLLM_HOST` env var)

---

## Endpoint: `POST /v1/chat/completions`

Used for all three capabilities. The `model` field routes to the correct loaded model.

### Request

**Content-Type**: `application/json`

```json
{
  "model": "<model_id>",
  "messages": [
    {
      "role": "system",
      "content": "<optional system prompt>"
    },
    {
      "role": "user",
      "content": "<user message text>"
    }
  ],
  "max_tokens": 512,
  "temperature": 0.7,
  "stream": false
}
```

#### Caption Request (image input)

For Caption capability, the user message `content` is an array with an image part:

```json
{
  "model": "llava-caption",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,<base64-encoded-bytes>"
          }
        },
        {
          "type": "text",
          "text": "Describe this image."
        }
      ]
    }
  ],
  "max_tokens": 256
}
```

#### Field Definitions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | Yes | Model identifier — one of `llama3-chat`, `qwen-coder`, `llava-caption` |
| `messages` | array | Yes | Non-empty array of `{role, content}` objects |
| `messages[].role` | string | Yes | `"system"`, `"user"`, or `"assistant"` |
| `messages[].content` | string or array | Yes | Text string, or array of content parts (Caption only) |
| `max_tokens` | integer | No | Max tokens to generate; default 512, max 4096 |
| `temperature` | float | No | Sampling temperature [0.0–2.0]; default 0.7 |
| `stream` | boolean | No | Must be `false` for POC; streaming not implemented |

---

### Response

**Content-Type**: `application/json`  
**Status**: `200 OK` on success

```json
{
  "id": "chatcmpl-<uuid>",
  "object": "chat.completion",
  "created": 1746057600,
  "model": "<model_id>",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "<generated text>"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 42,
    "completion_tokens": 128,
    "total_tokens": 170
  }
}
```

#### Field Definitions

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Server-generated completion ID |
| `model` | string | Echoed model identifier |
| `choices[0].message.content` | string | The generated response text |
| `choices[0].finish_reason` | string | `"stop"` (normal), `"length"` (hit max_tokens) |
| `usage.prompt_tokens` | integer | Tokens consumed by input; may be null if not reported |
| `usage.completion_tokens` | integer | Tokens in the generated output; may be null |

---

### Error Responses

| HTTP Status | Meaning | Rust Client Behaviour |
|-------------|---------|----------------------|
| `400 Bad Request` | Malformed request or unsupported model | Log error, return `Err` |
| `503 Service Unavailable` | Model not loaded / Triton not ready | Log error, return `Err` (retry logic out of scope) |
| `500 Internal Server Error` | Inference error | Log error + body, return `Err` |

---

## Endpoint: `GET /v1/models`

Health/discovery check — lists loaded models. Used by the Rust client startup probe.

### Response

```json
{
  "object": "list",
  "data": [
    { "id": "llama3-chat", "object": "model" },
    { "id": "qwen-coder",  "object": "model" },
    { "id": "llava-caption","object": "model" }
  ]
}
```

The Rust client SHOULD call this on startup to confirm the target model is available
before accepting inference requests.

---

## Model Identifiers

| Capability | `model` value | Backing model |
|------------|--------------|---------------|
| `Chat` | `llama3-chat` | Llama 3.1 8B Instruct (FP8 engine) |
| `Code` | `qwen-coder` | Qwen2.5-Coder 7B Instruct (FP8 engine) |
| `Caption` | `llava-caption` | LLaVA-1.6 Mistral-7B (FP8 engine) |

Model identifiers are configured in the Triton model repository (`config.pbtxt` `name`
field) and MUST match the values the Rust client sends in `model`.

---

## Client Configuration

The Rust client reads serving configuration from environment variables at startup:

| Variable | Default | Description |
|----------|---------|-------------|
| `TRTLLM_HOST` | `localhost` | Hostname or IP of the Triton container |
| `TRTLLM_PORT` | `8000` | HTTP port |
| `TRTLLM_CHAT_MODEL` | `llama3-chat` | Model ID for Chat capability |
| `TRTLLM_CODE_MODEL` | `qwen-coder` | Model ID for Code capability |
| `TRTLLM_CAPTION_MODEL` | `llava-caption` | Model ID for Caption capability |

---

## Constraints for This POC

- Streaming (`"stream": true`) is **not implemented** — the client always uses blocking
  completion mode
- Concurrent requests are **not tested** — single-request smoke test only
- Authentication is **not required** — the endpoint is local-only, no auth header needed
- Image format: JPEG or PNG, base64-encoded inline; external URLs are not used
- Image size: no formal limit enforced for POC; if an oversized image causes an inference
  error, document the observed practical limit in `docs/setup.md`
