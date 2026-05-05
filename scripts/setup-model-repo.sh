#!/usr/bin/env bash
# scripts/setup-model-repo.sh — create or rename Triton model pipeline directories
#
# Usage:
#   setup-model-repo.sh qwen      # rename bare dirs to *_qwen (first-time only)
#   setup-model-repo.sh llama     # create *_llama pipeline from inflight_batcher_llm template
#   setup-model-repo.sh vision    # create vision pipeline from multimodal template
#
# Idempotent: if target directories already exist the script exits successfully.
#
# REQUIRES (for llama/vision): the Triton 25.05 image to be locally available so
# template files can be copied out.  Run inside `just build-llama` / `just build-llava`
# or after pulling the image with `just pull`.

set -euo pipefail

PIPELINE="${1:-}"
TRTLLM_HOME="${TRTLLM_HOME:-/ai/trtllm-poc}"
MODEL_REPO="${TRTLLM_HOME}/model_repo"
IMAGE_TRITON="nvcr.io/nvidia/tritonserver:25.05-trtllm-python-py3"
# Path prefix as seen by Triton *inside* the container (compose.yaml mounts
# $TRTLLM_HOME → /workspace/trtllm). Engine/tokenizer paths written into
# config.pbtxt must use this prefix so Triton can resolve them at runtime.
CONTAINER_BASE="/workspace/trtllm"

if [[ -z "${PIPELINE}" ]]; then
    echo "Usage: $0 <qwen|llama|vision>" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Replace a string in a file, in-place.
_sed_inplace() {
    local pattern="$1" file="$2"
    sed -i "${pattern}" "${file}"
}

# Update the `name:` field in a config.pbtxt to match the directory name.
_set_name() {
    local config="$1" new_name="$2"
    _sed_inplace "s/^name: \".*\"/name: \"${new_name}\"/" "${config}"
}

# Fill common integer/enum template placeholders left by the container templates.
# These must be concrete values before Triton can parse the protobuf config.
# String-valued params that hold literal "${...}" are tolerated by Triton and left as-is.
_fill_common_templates() {
    local config="$1"
    _sed_inplace 's/\${triton_max_batch_size}/4/g'                         "${config}"
    _sed_inplace 's/\${max_queue_delay_microseconds}/0/g'                   "${config}"
    _sed_inplace 's/\${max_queue_size}/0/g'                                 "${config}"
    _sed_inplace 's/\${decoupled_mode}/false/g'                             "${config}"
    _sed_inplace 's/\${triton_backend}/tensorrtllm/g'                       "${config}"
    _sed_inplace 's/\${logits_datatype}/TYPE_FP32/g'                        "${config}"
    _sed_inplace 's/\${encoder_input_features_data_type}/TYPE_FP16/g'       "${config}"
    _sed_inplace 's/\${skip_special_tokens}/true/g'                         "${config}"
    _sed_inplace 's/\${postprocessing_instance_count}/1/g'                  "${config}"
    _sed_inplace 's/\${preprocessing_instance_count}/1/g'                   "${config}"
    # batching_strategy → gpt_model_type: must be a valid enum value
    _sed_inplace 's/\${batching_strategy}/inflight_fused_batching/g'        "${config}"
    # BLS-specific defaults
    _sed_inplace 's/\${bls_instance_count}/1/g'                             "${config}"
    _sed_inplace 's/\${accumulate_tokens}/false/g'                          "${config}"
    # Draft model / multimodal: empty = disabled
    _sed_inplace 's/\${tensorrt_llm_draft_model_name}//g'                   "${config}"
    _sed_inplace 's/\${multimodal_encoders_name}//g'                        "${config}"
}

# ---------------------------------------------------------------------------
# QWEN — rename bare names → *_qwen
# ---------------------------------------------------------------------------
setup_qwen() {
    local components=("preprocessing" "tensorrt_llm" "postprocessing" "ensemble" "tensorrt_llm_bls")

    echo "[setup-model-repo] Setting up qwen pipeline in ${MODEL_REPO}"

    for comp in "${components[@]}"; do
        local src="${MODEL_REPO}/${comp}"
        local dst="${MODEL_REPO}/${comp}_qwen"

        if [[ ! -d "${src}" && -d "${dst}" ]]; then
            echo "  ${comp}_qwen already exists — ensuring name: field is correct"
            _set_name "${dst}/config.pbtxt" "${comp}_qwen"
            continue
        fi

        if [[ ! -d "${src}" ]]; then
            echo "  ERROR: ${src} does not exist and ${dst} does not exist" >&2
            exit 1
        fi

        echo "  Renaming ${comp} → ${comp}_qwen"
        mv "${src}" "${dst}"

        # Update name: field in config.pbtxt
        _set_name "${dst}/config.pbtxt" "${comp}_qwen"
    done

    # Update ensemble_qwen: model_name refs in ensemble_scheduling
    local ensemble_cfg="${MODEL_REPO}/ensemble_qwen/config.pbtxt"
    echo "  Updating ensemble_qwen cross-references"
    _sed_inplace 's/model_name: "preprocessing"/model_name: "preprocessing_qwen"/' "${ensemble_cfg}"
    _sed_inplace 's/model_name: "tensorrt_llm"/model_name: "tensorrt_llm_qwen"/' "${ensemble_cfg}"
    _sed_inplace 's/model_name: "postprocessing"/model_name: "postprocessing_qwen"/' "${ensemble_cfg}"

    # Update tensorrt_llm_bls_qwen: tensorrt_llm_model_name default
    local bls_cfg="${MODEL_REPO}/tensorrt_llm_bls_qwen/config.pbtxt"
    echo "  Updating tensorrt_llm_bls_qwen config"
    _sed_inplace 's/\${tensorrt_llm_model_name}/tensorrt_llm_qwen/' "${bls_cfg}"

    # Update tensorrt_llm_bls_qwen model.py default args
    local bls_model="${MODEL_REPO}/tensorrt_llm_bls_qwen/1/model.py"
    _sed_inplace "s/default_tensorrt_llm_model_name = 'tensorrt_llm'/default_tensorrt_llm_model_name = 'tensorrt_llm_qwen'/" "${bls_model}"
    _sed_inplace 's/preproc_model_name="preprocessing"/preproc_model_name="preprocessing_qwen"/' "${bls_model}"
    _sed_inplace 's/postproc_model_name="postprocessing"/postproc_model_name="postprocessing_qwen"/' "${bls_model}"

    # Update triton_decoder.py default args
    local decoder="${MODEL_REPO}/tensorrt_llm_bls_qwen/1/lib/triton_decoder.py"
    if [[ -f "${decoder}" ]]; then
        _sed_inplace 's/preproc_model_name="preprocessing"/preproc_model_name="preprocessing_qwen"/' "${decoder}"
        _sed_inplace 's/postproc_model_name="postprocessing"/postproc_model_name="postprocessing_qwen"/' "${decoder}"
        _sed_inplace 's/llm_model_name="tensorrt_llm"/llm_model_name="tensorrt_llm_qwen"/' "${decoder}"
    fi

    echo "[setup-model-repo] qwen pipeline ready"
    sudo chown -R "${USER}:${USER}" "${MODEL_REPO}"
}

# ---------------------------------------------------------------------------
# LLAMA — create *_llama dirs from inflight_batcher_llm template
# ---------------------------------------------------------------------------
setup_llama() {
    local engine_dir="${CONTAINER_BASE}/engines/llama"
    local tokenizer_dir="${CONTAINER_BASE}/models/llama-3.1-8b"
    local components=("preprocessing" "tensorrt_llm" "postprocessing" "ensemble" "tensorrt_llm_bls")

    echo "[setup-model-repo] Setting up llama pipeline in ${MODEL_REPO}"

    # Check at least one dest already exists → idempotent
    if [[ -d "${MODEL_REPO}/ensemble_llama" ]]; then
        echo "  ensemble_llama already exists — skipping (delete manually to recreate)"
        exit 0
    fi

    if [[ ! -d "${TRTLLM_HOME}/engines/llama" ]]; then
        echo "  ERROR: Llama engine dir not found: ${TRTLLM_HOME}/engines/llama" >&2
        echo "  Run 'just build-llama' first." >&2
        exit 1
    fi

    # Copy templates from the container image
    _LLAMA_TMP=$(mktemp -d)
    trap 'sudo rm -rf "${_LLAMA_TMP}"' EXIT

    echo "  Extracting inflight_batcher_llm templates from container..."
    docker run --rm \
        -v "${_LLAMA_TMP}:/out" \
        "${IMAGE_TRITON}" \
        bash -c "cp -r /app/all_models/inflight_batcher_llm/. /out/"

    for comp in "${components[@]}"; do
        local src_dir="${_LLAMA_TMP}/${comp}"
        local dst_dir="${MODEL_REPO}/${comp}_llama"

        if [[ ! -d "${src_dir}" ]]; then
            echo "  WARNING: template dir not found: ${src_dir} — skipping" >&2
            continue
        fi

        echo "  Creating ${comp}_llama"
        cp -r "${src_dir}" "${dst_dir}"

        # Update name: field and fill common template variables
        local cfg="${dst_dir}/config.pbtxt"
        _set_name "${cfg}" "${comp}_llama"
        _fill_common_templates "${cfg}"
        _sed_inplace "s|\${engine_dir}|${engine_dir}|g"       "${cfg}"
        _sed_inplace "s|\${tokenizer_dir}|${tokenizer_dir}|g" "${cfg}"
        # Llama is text-only: no multimodal / vision support
        _sed_inplace "s|\${multimodal_model_path}||g"          "${cfg}"
        _sed_inplace "s|\${max_num_images}|1|g"                "${cfg}"
        _sed_inplace "s|\${add_special_tokens}|true|g"         "${cfg}"
        _sed_inplace 's/\${enable_kv_cache_reuse}/false/g'     "${cfg}"
    done

    # Update ensemble_llama cross-references
    local ensemble_cfg="${MODEL_REPO}/ensemble_llama/config.pbtxt"
    echo "  Updating ensemble_llama cross-references"
    _sed_inplace 's/model_name: "preprocessing"/model_name: "preprocessing_llama"/' "${ensemble_cfg}"
    _sed_inplace 's/model_name: "tensorrt_llm"/model_name: "tensorrt_llm_llama"/' "${ensemble_cfg}"
    _sed_inplace 's/model_name: "postprocessing"/model_name: "postprocessing_llama"/' "${ensemble_cfg}"

    # Update tensorrt_llm_bls_llama config
    local bls_cfg="${MODEL_REPO}/tensorrt_llm_bls_llama/config.pbtxt"
    _sed_inplace 's/\${tensorrt_llm_model_name}/tensorrt_llm_llama/' "${bls_cfg}"

    # Update tensorrt_llm_bls_llama model.py defaults
    local bls_model="${MODEL_REPO}/tensorrt_llm_bls_llama/1/model.py"
    if [[ -f "${bls_model}" ]]; then
        _sed_inplace "s/default_tensorrt_llm_model_name = 'tensorrt_llm'/default_tensorrt_llm_model_name = 'tensorrt_llm_llama'/" "${bls_model}"
        _sed_inplace 's/preproc_model_name="preprocessing"/preproc_model_name="preprocessing_llama"/' "${bls_model}"
        _sed_inplace 's/postproc_model_name="postprocessing"/postproc_model_name="postprocessing_llama"/' "${bls_model}"
    fi

    local decoder="${MODEL_REPO}/tensorrt_llm_bls_llama/1/lib/triton_decoder.py"
    if [[ -f "${decoder}" ]]; then
        _sed_inplace 's/preproc_model_name="preprocessing"/preproc_model_name="preprocessing_llama"/' "${decoder}"
        _sed_inplace 's/postproc_model_name="postprocessing"/postproc_model_name="postprocessing_llama"/' "${decoder}"
        _sed_inplace 's/llm_model_name="tensorrt_llm"/llm_model_name="tensorrt_llm_llama"/' "${decoder}"
    fi

    echo "[setup-model-repo] llama pipeline ready"
    sudo chown -R "${USER}:${USER}" "${MODEL_REPO}"
}

# ---------------------------------------------------------------------------
# VISION — create vision pipeline for LLaVA 1.5 7B (no tensorrt_llm_bls).
# The multimodal ensemble chains 4 components:
#   preprocessing_vision    — from inflight_batcher_llm template
#   multimodal_encoders     — from multimodal template (bare name, no suffix)
#   tensorrt_llm_vision     — from inflight_batcher_llm template
#   postprocessing_vision   — from inflight_batcher_llm template
#   ensemble_vision         — from multimodal template (entry point)
# ---------------------------------------------------------------------------
setup_vision() {
    local llm_engine_dir="${CONTAINER_BASE}/engines/llava/llm"
    local vision_engine_dir="${CONTAINER_BASE}/engines/llava/vision"
    local tokenizer_dir="${CONTAINER_BASE}/models/llava-1.5-7b"

    echo "[setup-model-repo] Setting up vision pipeline in ${MODEL_REPO}"

    if [[ -d "${MODEL_REPO}/ensemble_vision" ]]; then
        echo "  ensemble_vision already exists — ensuring name: field is correct"
        _set_name "${MODEL_REPO}/ensemble_vision/config.pbtxt" "ensemble_vision"
        exit 0
    fi

    if [[ ! -d "${TRTLLM_HOME}/engines/llava/llm" ]]; then
        echo "  ERROR: LLaVA LLM engine dir not found: ${TRTLLM_HOME}/engines/llava/llm" >&2
        echo "  Run 'just build-llava' first." >&2
        exit 1
    fi

    # Use script-level vars so the EXIT trap can see them after function returns.
    _VISION_TMP_MULTI=$(mktemp -d)
    _VISION_TMP_INFLIGHT=$(mktemp -d)
    trap 'sudo rm -rf "${_VISION_TMP_MULTI}" "${_VISION_TMP_INFLIGHT}"' EXIT

    echo "  Extracting multimodal templates from container..."
    docker run --rm \
        -v "${_VISION_TMP_MULTI}:/out" \
        "${IMAGE_TRITON}" \
        bash -c "cp -r /app/all_models/multimodal/. /out/"

    echo "  Extracting inflight_batcher_llm templates from container..."
    docker run --rm \
        -v "${_VISION_TMP_INFLIGHT}:/out" \
        "${IMAGE_TRITON}" \
        bash -c "cp -r /app/all_models/inflight_batcher_llm/. /out/"

    # preprocessing_vision — from inflight_batcher_llm
    local pre_src="${_VISION_TMP_INFLIGHT}/preprocessing"
    if [[ -d "${pre_src}" ]]; then
        echo "  Creating preprocessing_vision"
        cp -r "${pre_src}" "${MODEL_REPO}/preprocessing_vision"
        _set_name "${MODEL_REPO}/preprocessing_vision/config.pbtxt" "preprocessing_vision"
        _fill_common_templates "${MODEL_REPO}/preprocessing_vision/config.pbtxt"
        _sed_inplace "s|\${tokenizer_dir}|${tokenizer_dir}|g" \
            "${MODEL_REPO}/preprocessing_vision/config.pbtxt"
        # multimodal_model_path = vision engine dir (has builder_config/config.json)
        _sed_inplace "s|\${multimodal_model_path}|${vision_engine_dir}|g" \
            "${MODEL_REPO}/preprocessing_vision/config.pbtxt"
        # gpt_model_path = LLM engine dir
        _sed_inplace "s|\${engine_dir}|${llm_engine_dir}|g" \
            "${MODEL_REPO}/preprocessing_vision/config.pbtxt"
    fi

    # multimodal_encoders — from multimodal template, bare name (only one vision pipeline)
    local enc_src="${_VISION_TMP_MULTI}/multimodal_encoders"
    if [[ -d "${enc_src}" ]]; then
        echo "  Creating multimodal_encoders"
        cp -r "${enc_src}" "${MODEL_REPO}/multimodal_encoders"
        _fill_common_templates "${MODEL_REPO}/multimodal_encoders/config.pbtxt"
        _sed_inplace "s|\${vision_engine_dir}|${vision_engine_dir}|g" \
            "${MODEL_REPO}/multimodal_encoders/config.pbtxt"
        _sed_inplace "s|\${tokenizer_dir}|${tokenizer_dir}|g" \
            "${MODEL_REPO}/multimodal_encoders/config.pbtxt"
        _sed_inplace "s|\${multimodal_model_path}|${vision_engine_dir}|g" \
            "${MODEL_REPO}/multimodal_encoders/config.pbtxt"
        _sed_inplace "s|\${hf_model_path}|${tokenizer_dir}|g" \
            "${MODEL_REPO}/multimodal_encoders/config.pbtxt"
    fi

    # tensorrt_llm_vision — from inflight_batcher_llm
    local trt_src="${_VISION_TMP_INFLIGHT}/tensorrt_llm"
    if [[ -d "${trt_src}" ]]; then
        echo "  Creating tensorrt_llm_vision"
        cp -r "${trt_src}" "${MODEL_REPO}/tensorrt_llm_vision"
        _set_name "${MODEL_REPO}/tensorrt_llm_vision/config.pbtxt" "tensorrt_llm_vision"
        _fill_common_templates "${MODEL_REPO}/tensorrt_llm_vision/config.pbtxt"
        _sed_inplace "s|\${engine_dir}|${llm_engine_dir}|g" \
            "${MODEL_REPO}/tensorrt_llm_vision/config.pbtxt"
        _sed_inplace "s|\${tokenizer_dir}|${tokenizer_dir}|g" \
            "${MODEL_REPO}/tensorrt_llm_vision/config.pbtxt"
        # KV cache reuse with prompt table requires extra_ids; disable to avoid assertion.
        _sed_inplace 's/\${enable_kv_cache_reuse}/false/g' \
            "${MODEL_REPO}/tensorrt_llm_vision/config.pbtxt"
    fi

    # postprocessing_vision — from inflight_batcher_llm
    local post_src="${_VISION_TMP_INFLIGHT}/postprocessing"
    if [[ -d "${post_src}" ]]; then
        echo "  Creating postprocessing_vision"
        cp -r "${post_src}" "${MODEL_REPO}/postprocessing_vision"
        _set_name "${MODEL_REPO}/postprocessing_vision/config.pbtxt" "postprocessing_vision"
        _fill_common_templates "${MODEL_REPO}/postprocessing_vision/config.pbtxt"
        _sed_inplace "s|\${tokenizer_dir}|${tokenizer_dir}|g" \
            "${MODEL_REPO}/postprocessing_vision/config.pbtxt"
    fi

    # ensemble_vision — from multimodal template (references all 4 components above)
    local ens_src="${_VISION_TMP_MULTI}/ensemble"
    if [[ -d "${ens_src}" ]]; then
        echo "  Creating ensemble_vision"
        cp -r "${ens_src}" "${MODEL_REPO}/ensemble_vision"
        # Ensure version directory exists (Triton requires at least one version)
        mkdir -p "${MODEL_REPO}/ensemble_vision/1"
        _set_name "${MODEL_REPO}/ensemble_vision/config.pbtxt" "ensemble_vision"
        _fill_common_templates "${MODEL_REPO}/ensemble_vision/config.pbtxt"
        # Update cross-references: bare names → suffixed names
        # multimodal_encoders stays bare (no suffix)
        _sed_inplace 's/model_name: "preprocessing"/model_name: "preprocessing_vision"/' \
            "${MODEL_REPO}/ensemble_vision/config.pbtxt"
        _sed_inplace 's/model_name: "tensorrt_llm"/model_name: "tensorrt_llm_vision"/' \
            "${MODEL_REPO}/ensemble_vision/config.pbtxt"
        _sed_inplace 's/model_name: "postprocessing"/model_name: "postprocessing_vision"/' \
            "${MODEL_REPO}/ensemble_vision/config.pbtxt"
    fi

    echo "[setup-model-repo] vision pipeline ready"
    sudo chown -R "${USER}:${USER}" "${MODEL_REPO}"
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
case "${PIPELINE}" in
    qwen)   setup_qwen ;;
    llama)  setup_llama ;;
    vision) setup_vision ;;
    *)
        echo "ERROR: unknown pipeline '${PIPELINE}'. Valid: qwen | llama | vision" >&2
        exit 1
        ;;
esac
