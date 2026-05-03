# Decision 003: Model Selection

**Date**: 2026-05-01  
**Status**: Decided  
**Research ref**: R-003 (research.md)

## Context

Three inference capabilities are required: general chat, coding assistance, and image
captioning. Models must fit on 32 GB VRAM (RTX 5090) and be supported by TRT-LLM's
engine builder and Triton backend.

## Constraints

- 32 GB VRAM total (RTX 5090)
- TRT-LLM engine builder must support the model architecture
- FP8 or FP4 quantization preferred to reduce VRAM footprint
- Models must be available on HuggingFace (gated ok, HF_TOKEN available)
- POC quality bar: "plausibly good" responses, not SOTA

## Selected Models

| Capability | Model | Quantization | Est. VRAM | Rationale |
|------------|-------|-------------|-----------|-----------|
| Chat | `meta-llama/Meta-Llama-3.1-8B-Instruct` | FP8 | ~10 GB | Strong general instruction following; widely used baseline; TRT-LLM documented support |
| Code | `Qwen/Qwen2.5-Coder-7B-Instruct` | FP8 | ~9 GB | Top-ranked small coding model; fast; TRT-LLM supports Qwen2 architecture |
| Caption | `llava-hf/llava-v1.6-mistral-7b-hf` (or `llava-hf/llava-v1.6-vicuna-7b-hf`) | FP8 | ~12 GB | LLaVA-1.6 listed in TRT-LLM support matrix under multi-modal; Mistral-7B backbone is well-supported |

**Combined estimate**: ~31 GB — tight but fits on 32 GB. For POC, load one model at a
time if concurrent serving causes OOM; document the observed limit.

## Alternatives Considered

- **Larger models (13B+)**: rejected — OOM risk on 32 GB without aggressive quantization
  that may degrade quality below useful threshold
- **Smaller models (1B–3B)**: rejected — quality too low for meaningful chat/code
  demonstrations
- **Running all three concurrently**: deferred — evaluate after smoke tests; POC can
  use sequential loading if memory is tight

## Fallback Plan

If LLaVA-1.6 multi-modal pipeline proves overly complex (separate visual encoder engine,
`config.pbtxt` structure for multi-modal):

1. Document the blocker in `docs/ansible-handoff.md` Notes section
2. Fall back to `Qwen/Qwen2-VL-7B-Instruct` (Qwen2-VL is in TRT-LLM support matrix
   and may have simpler integration) or a text-only stub for the Caption endpoint
3. The Rust client's Caption routing remains unchanged; only the backend model changes

## Triton Model IDs

After engine build, models are registered in Triton with these identifiers:

| Capability | Triton `model` ID | Engine directory |
|------------|------------------|-----------------|
| Chat | `llama3-chat` | `$TRTLLM_HOME/engines/llama3-chat/` |
| Code | `qwen-coder` | `$TRTLLM_HOME/engines/qwen-coder/` |
| Caption | `llava-caption` | `$TRTLLM_HOME/engines/llava-caption/` |
