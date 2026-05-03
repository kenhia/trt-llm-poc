use crate::capability::Capability;
use crate::models::InferenceResponse;

/// Emit a structured tracing event for a completed inference request (FR-009).
///
/// All fields required by FR-009 are included:
///   request_id, capability, model_id, latency_ms,
///   prompt_tokens, completion_tokens, success, error
pub fn log_request(response: &InferenceResponse, success: bool, error: Option<&str>) {
    tracing::info!(
        request_id = %response.request_id,
        capability = %response.capability,
        model_id = %response.model_id,
        latency_ms = response.latency_ms,
        prompt_tokens = response.prompt_tokens,
        completion_tokens = response.completion_tokens,
        success = success,
        error = error,
        "inference request completed"
    );
}

/// Emit a structured tracing event for a failed inference request (no response available).
pub fn log_request_error(request_id: &str, capability: Capability, latency_ms: u64, error: &str) {
    tracing::error!(
        request_id = %request_id,
        capability = %capability,
        latency_ms = latency_ms,
        success = false,
        error = %error,
        "inference request failed"
    );
}
