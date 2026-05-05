use crate::models::InferenceRequest;

/// The three inference capabilities supported by this POC.
/// Determines which Triton model and model_id are used for a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Capability {
    /// General conversational LLM → model_id "ensemble_llama"
    Chat,
    /// Coding-focused LLM → model_id "ensemble_qwen"
    Code,
    /// Vision-language model for image description → model_id "ensemble_vision"
    Caption,
}

impl Capability {
    /// Returns the Triton model identifier for this capability.
    /// Each capability routes to its dedicated pipeline entry point.
    /// See data-model.md for the full pipeline naming convention.
    pub fn model_id(&self) -> &'static str {
        match self {
            Capability::Chat => "ensemble_llama",
            Capability::Code => "ensemble_qwen",
            Capability::Caption => "ensemble_vision",
        }
    }
}

impl std::fmt::Display for Capability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Capability::Chat => write!(f, "Chat"),
            Capability::Code => write!(f, "Code"),
            Capability::Caption => write!(f, "Caption"),
        }
    }
}

/// Routing rules from data-model.md:
/// 1. If `image` is Some(_) → Caption  (image presence always wins)
/// 2. Else if `system_hint` contains "code" (case-insensitive) → Code
/// 3. Else → Chat
pub fn resolve_capability(req: &InferenceRequest) -> Capability {
    if req.image.is_some() {
        return Capability::Caption;
    }
    if let Some(hint) = &req.system_hint
        && hint.to_lowercase().contains("code")
    {
        return Capability::Code;
    }
    Capability::Chat
}
