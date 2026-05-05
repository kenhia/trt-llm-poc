#!/usr/bin/env python3
# scripts/triton_wrapper.py — thin wrapper that injects explicit model-control mode
# before delegating to the bundled OpenAI frontend (main.py).
#
# WHY: main.py constructs tritonserver.Server() without model_control_mode, so all
# models in the repository load at startup (NONE mode).  Explicit mode lets us
# load/unload individual pipelines on demand via the KServe v2 repository API
# (POST /v2/repository/models/{name}/load|unload) without restarting the container.
#
# HOW: We monkey-patch tritonserver.Server.__init__ to intercept the construction
# call made by main.py and inject:
#   model_control_mode = ModelControlMode.EXPLICIT
#   startup_models     = [all 5 model dirs for DEFAULT_MODEL pipeline, if set]
#
# ENV vars consumed here (set in compose.yaml / $TRTLLM_HOME/.env):
#   TRITON_MODEL_CONTROL_MODE  "explicit" (default) | "none"
#   DEFAULT_MODEL              short pipeline name: "qwen" | "llama" | "vision" | ""
#
# All other args (--model-repository, --tokenizer, --openai-port, etc.) are passed
# through unchanged to main.py's argparse.

import os
import sys

# ---------------------------------------------------------------------------
# Pipeline name → ordered list of Triton model directory names.
# Order matters for load (dependency order) and unload (reverse).
# These must exactly match the directory names under $TRTLLM_HOME/model_repo/.
# ---------------------------------------------------------------------------
PIPELINE_MODELS: dict[str, list[str]] = {
    "qwen": [
        "preprocessing_qwen",
        "tensorrt_llm_qwen",
        "postprocessing_qwen",
        "tensorrt_llm_bls_qwen",
        "ensemble_qwen",
    ],
    "llama": [
        "preprocessing_llama",
        "tensorrt_llm_llama",
        "postprocessing_llama",
        "tensorrt_llm_bls_llama",
        "ensemble_llama",
    ],
    "vision": [
        "preprocessing_vision",
        "multimodal_encoders",
        "tensorrt_llm_vision",
        "postprocessing_vision",
        "ensemble_vision",
    ],
}


def _patch_server() -> None:
    """Monkey-patch tritonserver.Server to inject explicit model-control mode."""
    import tritonserver

    control_mode_env = os.environ.get("TRITON_MODEL_CONTROL_MODE", "explicit").lower()
    if control_mode_env == "none":
        # Caller explicitly opted out — let main.py run unmodified.
        print("[triton_wrapper] Model-control mode: NONE (passthrough)", flush=True)
        return

    # Resolve startup_models from DEFAULT_MODEL env var.
    default_model = os.environ.get("DEFAULT_MODEL", "").strip()
    startup_models: list[str] = []
    if default_model:
        startup_models = PIPELINE_MODELS.get(default_model, [])
        if not startup_models:
            print(
                f"[triton_wrapper] WARNING: DEFAULT_MODEL='{default_model}' is not a "
                f"known pipeline name. Known: {list(PIPELINE_MODELS)}. "
                "Starting with no models loaded.",
                flush=True,
            )

    explicit_mode = tritonserver.ModelControlMode.EXPLICIT
    print(
        f"[triton_wrapper] Model-control mode: EXPLICIT | "
        f"startup_models: {startup_models or '(none)'}",
        flush=True,
    )

    _original_init = tritonserver.Server.__init__

    def _patched_init(self, *args, **kwargs):  # type: ignore[override]
        kwargs["model_control_mode"] = explicit_mode
        kwargs["startup_models"] = startup_models
        _original_init(self, *args, **kwargs)

    tritonserver.Server.__init__ = _patched_init  # type: ignore[method-assign]


def _patch_engine_metadata() -> None:
    """Patch TritonLLMEngine._get_model_metadata to skip unloaded models.

    In EXPLICIT model-control mode, self.server.models() returns all models in
    the repository, including those that are not loaded.  Calling model.config()
    on an unloaded model raises tritonserver.NotFoundError, which crashes the
    OpenAI frontend at startup.  This patch silently skips any model whose
    config cannot be retrieved so the frontend starts with only the loaded
    pipeline visible.
    """
    try:
        import tritonserver
        from engine.triton_engine import TritonLLMEngine  # noqa: PLC0415
    except ImportError:
        # Engine module not on path yet — will be importable after sys.path is set.
        # Re-patch after path setup via the deferred call in main().
        return False  # type: ignore[return-value]

    _original_get_metadata = TritonLLMEngine._get_model_metadata

    def _patched_get_metadata(self):  # type: ignore[override]
        model_metadata = {}
        for name, _ in self.server.models().keys():
            model = self.server.model(name)
            try:
                config = model.config()
            except tritonserver.NotFoundError:
                print(
                    f"[triton_wrapper] Skipping unloaded model '{name}' "
                    "(not loaded in EXPLICIT mode)",
                    flush=True,
                )
                continue
            backend = config.get("backend", "")
            if not backend and config.get("platform") == "ensemble":
                backend = "ensemble"
            print(f"Found model: {name=}, {backend=}")
            from engine.triton_engine import TritonModelMetadata, _get_vllm_lora_names  # noqa: PLC0415
            lora_names = None
            if self.backend == "vllm" or backend == "vllm":
                lora_names = _get_vllm_lora_names(
                    self.server.options.model_repository, name, model.version
                )
            metadata = TritonModelMetadata(
                name=name,
                backend=backend,
                model=model,
                tokenizer=self.tokenizer,
                lora_names=lora_names,
                create_time=self.create_time,
                request_converter=self._determine_request_converter(backend),
            )
            model_metadata[name] = metadata
        return model_metadata

    TritonLLMEngine._get_model_metadata = _patched_get_metadata  # type: ignore[method-assign]
    print("[triton_wrapper] Patched TritonLLMEngine._get_model_metadata (EXPLICIT mode)", flush=True)

    # --- Multimodal (image_url) support ---
    # The stock OpenAI frontend only handles text messages; image_url parts raise
    # TypeError in _parse_chat_message_content_parts().  Patch:
    #   1. chat.py → _parse_chat_message_content_parts: handle image_url gracefully
    #   2. triton.py → _create_trtllm_inference_request: inject image_url_input when
    #      the request carries a _image_urls attribute set by our chat wrapper.

    import engine.utils.chat as _chat_utils  # noqa: PLC0415
    import engine.utils.triton as _triton_utils  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    _orig_parse_content_parts = _chat_utils._parse_chat_message_content_parts
    _orig_create_trtllm_req = _triton_utils._create_trtllm_inference_request

    def _patched_parse_content_parts(role, parts):  # type: ignore[override]
        """Accept image_url parts: keep only text, ignore images (text-only conversation)."""
        from engine.utils.chat import ConversationMessage  # noqa: PLC0415
        content = []
        for part in parts:
            p = part.root
            if getattr(p, "type", None) in ("text",):
                content.append({"type": "text", "text": p.text})
            # image_url parts are silently skipped here; the URLs are extracted
            # in _dynamic_chat and injected into the Triton inference request.
        return ConversationMessage(role=role, content=content or None)

    _chat_utils._parse_chat_message_content_parts = _patched_parse_content_parts

    def _patched_create_trtllm_req(model, prompt, request, lora_name):  # type: ignore[override]
        """Add image_url_input when request carries _image_urls (set by _dynamic_chat)."""
        triton_request = _orig_create_trtllm_req(model, prompt, request, lora_name)
        image_urls = getattr(request, "_image_urls", None)
        if image_urls:
            # ensemble_vision maps image_url_input → IMAGE_URL → preprocessing_vision
            img_input = tritonserver.Tensor(
                name="image_url_input",
                data_type=tritonserver.DataType.BYTES,
                data=np.array([[image_urls[0]]], dtype=object),
            )
            triton_request.inputs["image_url_input"] = img_input
            print(
                f"[triton_wrapper] Injected image_url_input: {image_urls[0][:60]}…",
                flush=True,
            )
        return triton_request

    _triton_utils._create_trtllm_inference_request = _patched_create_trtllm_req

    # Re-export so triton_engine's import alias also sees the patched version.
    import engine.triton_engine as _engine_mod  # noqa: PLC0415
    _engine_mod._create_trtllm_inference_request = _patched_create_trtllm_req

    print("[triton_wrapper] Patched frontend for image_url support", flush=True)

    # --- Dynamic model discovery ---
    # The frontend caches model_metadata once at __init__ time.  In EXPLICIT mode,
    # models are loaded/unloaded dynamically via the KServe repository API, so the
    # cache goes stale.  Patch `models()` to always refresh, and patch `chat()` /
    # `completions()` to refresh lazily when the requested model is not in cache.

    _original_models = TritonLLMEngine.models
    _original_chat = TritonLLMEngine.chat
    _original_completion = TritonLLMEngine.completion

    def _dynamic_models(self):  # type: ignore[override]
        self.model_metadata = self._get_model_metadata()
        return _original_models(self)

    def _extract_image_urls(messages) -> list[str]:
        """Pull image_url values out of multimodal chat messages."""
        urls = []
        for msg in messages or []:
            content = getattr(msg.root, "content", None)
            if not isinstance(content, list):
                continue
            for part in content:
                p = part.root
                if getattr(p, "type", None) == "image_url":
                    iu = getattr(p, "image_url", None)
                    if iu:
                        url = getattr(iu, "url", None) or str(iu)
                        urls.append(url)
        return urls

    async def _dynamic_chat(self, request):  # type: ignore[override]
        model_name = getattr(request, "model", None)
        if model_name and model_name not in self.model_metadata:
            print(
                f"[triton_wrapper] Model '{model_name}' not in cache — refreshing metadata",
                flush=True,
            )
            self.model_metadata = self._get_model_metadata()
        # Extract image URLs from multimodal content and attach to request so
        # _patched_create_trtllm_req can inject them into the Triton call.
        image_urls = _extract_image_urls(getattr(request, "messages", None))
        if image_urls:
            request._image_urls = image_urls  # type: ignore[attr-defined]
        return await _original_chat(self, request)

    async def _dynamic_completion(self, request):  # type: ignore[override]
        model_name = getattr(request, "model", None)
        if model_name and model_name not in self.model_metadata:
            print(
                f"[triton_wrapper] Model '{model_name}' not in cache — refreshing metadata",
                flush=True,
            )
            self.model_metadata = self._get_model_metadata()
        return await _original_completion(self, request)

    TritonLLMEngine.models = _dynamic_models  # type: ignore[method-assign]
    TritonLLMEngine.chat = _dynamic_chat  # type: ignore[method-assign]
    TritonLLMEngine.completion = _dynamic_completion  # type: ignore[method-assign]
    print("[triton_wrapper] Patched TritonLLMEngine for dynamic model discovery", flush=True)
    return True  # type: ignore[return-value]


def main() -> None:
    _patch_server()

    # Import main.py from the container's bundled OpenAI frontend.
    # sys.path is adjusted so `import main` resolves to main.py, not this file.
    frontend_dir = "/opt/tritonserver/python/openai/openai_frontend"
    if frontend_dir not in sys.path:
        sys.path.insert(0, frontend_dir)

    # Patch _get_model_metadata now that engine/ is on sys.path.
    _patch_engine_metadata()

    import main as openai_main  # noqa: PLC0415

    openai_main.main()


if __name__ == "__main__":
    main()
