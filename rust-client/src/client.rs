use std::time::Instant;

use reqwest::Client;

use crate::capability::{Capability, resolve_capability};
use crate::models::{
    ChatCompletionRequest, ChatCompletionResponse, ContentPart, ImageUrl, InferenceRequest,
    InferenceResponse, MAX_TOKENS_DEFAULT, Message, MessageContent, ModelsResponse,
    TEMPERATURE_DEFAULT,
};

/// Trait for sending inference requests. Enables testing with mock implementations.
#[async_trait::async_trait]
pub trait LlmClient {
    async fn infer(&self, req: InferenceRequest) -> Result<InferenceResponse, String>;
}

/// HTTP implementation of `LlmClient` using Triton's OpenAI-compatible endpoint.
pub struct HttpLlmClient {
    client: Client,
    /// Base URL for the Triton endpoint, e.g., "http://localhost:8000"
    base_url: String,
}

impl HttpLlmClient {
    pub fn new(base_url: String) -> Self {
        Self {
            client: Client::new(),
            base_url,
        }
    }

    /// Build base_url from TRTLLM_HOST / TRTLLM_PORT env vars.
    /// Defaults: host = "localhost", port = "9000" (OpenAI-compat frontend)
    pub fn from_env() -> Self {
        let host = std::env::var("TRTLLM_HOST").unwrap_or_else(|_| "localhost".to_string());
        let port = std::env::var("TRTLLM_PORT").unwrap_or_else(|_| "9000".to_string());
        Self::new(format!("http://{}:{}", host, port))
    }

    /// Probe GET /v1/models — returns list of loaded model IDs.
    /// Used at startup to confirm the target model is available before sending requests.
    pub async fn probe_models(&self) -> Result<Vec<String>, String> {
        let url = format!("{}/v1/models", self.base_url);
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| format!("GET /v1/models failed: {}", e))?;

        if !resp.status().is_success() {
            return Err(format!("GET /v1/models returned HTTP {}", resp.status()));
        }

        let body: ModelsResponse = resp
            .json()
            .await
            .map_err(|e| format!("Failed to parse /v1/models response: {}", e))?;

        Ok(body.data.into_iter().map(|m| m.id).collect())
    }

    /// Build the wire-format ChatCompletionRequest from an InferenceRequest.
    fn build_wire_request(req: &InferenceRequest, capability: Capability) -> ChatCompletionRequest {
        let model_id = capability.model_id().to_string();
        let max_tokens = req.max_tokens.unwrap_or(MAX_TOKENS_DEFAULT);
        let temperature = req.temperature.unwrap_or(TEMPERATURE_DEFAULT);

        // For Caption, inject the image as a multimodal content part in the last user message.
        let messages = if capability == Capability::Caption {
            if let Some(image_b64) = &req.image {
                let data_url = format!("data:image/jpeg;base64,{}", image_b64);
                // Build a multimodal message: image part first, then any text from the
                // last user message (or a default caption prompt).
                let text_content = req
                    .messages
                    .iter()
                    .rev()
                    .find(|m| m.role == "user")
                    .map(|m| match &m.content {
                        MessageContent::Text(t) => t.clone(),
                        MessageContent::Parts(_) => "Describe this image.".to_string(),
                    })
                    .unwrap_or_else(|| "Describe this image.".to_string());

                let multimodal_msg = Message {
                    role: "user".to_string(),
                    content: MessageContent::Parts(vec![
                        ContentPart::ImageUrl {
                            image_url: ImageUrl { url: data_url },
                        },
                        ContentPart::Text { text: text_content },
                    ]),
                };
                // Keep any prior system messages, replace last user message with multimodal.
                let mut msgs: Vec<Message> = req
                    .messages
                    .iter()
                    .filter(|m| m.role == "system")
                    .cloned()
                    .collect();
                msgs.push(multimodal_msg);
                msgs
            } else {
                req.messages.clone()
            }
        } else {
            req.messages.clone()
        };

        ChatCompletionRequest {
            model: model_id,
            messages,
            max_tokens,
            temperature,
            stream: false,
        }
    }
}

#[async_trait::async_trait]
impl LlmClient for HttpLlmClient {
    async fn infer(&self, req: InferenceRequest) -> Result<InferenceResponse, String> {
        let capability = resolve_capability(&req);
        let wire_req = Self::build_wire_request(&req, capability);
        let url = format!("{}/v1/chat/completions", self.base_url);

        let start = Instant::now();
        let http_resp = self
            .client
            .post(&url)
            .json(&wire_req)
            .send()
            .await
            .map_err(|e| format!("POST /v1/chat/completions failed: {}", e))?;
        let latency_ms = start.elapsed().as_millis() as u64;

        let status = http_resp.status();
        if !status.is_success() {
            let body = http_resp.text().await.unwrap_or_default();
            return Err(format!("HTTP {} from endpoint: {}", status, body));
        }

        let body: ChatCompletionResponse = http_resp
            .json()
            .await
            .map_err(|e| format!("Failed to parse completion response: {}", e))?;

        let content = body
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .unwrap_or_default();

        Ok(InferenceResponse {
            request_id: req.id,
            capability,
            content,
            model_id: body.model,
            prompt_tokens: body.usage.as_ref().and_then(|u| u.prompt_tokens),
            completion_tokens: body.usage.as_ref().and_then(|u| u.completion_tokens),
            latency_ms,
        })
    }
}
