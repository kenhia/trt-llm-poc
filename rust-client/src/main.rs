use std::process;

use tracing_subscriber::{EnvFilter, fmt};
use uuid::Uuid;

use rust_client::capability::Capability;
use rust_client::client::{HttpLlmClient, LlmClient};
use rust_client::logging::{log_request, log_request_error};
use rust_client::models::{InferenceRequest, Message, MessageContent};

#[tokio::main]
async fn main() {
    // Initialise JSON structured logging to stdout.
    // Log level is read from RUST_LOG env var; default to "info".
    fmt()
        .json()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_current_span(false)
        .init();

    let args: Vec<String> = std::env::args().collect();
    let subcommand = args.get(1).map(String::as_str);

    match subcommand {
        Some("smoke-test") => {
            if let Err(e) = run_smoke_test().await {
                tracing::error!(error = %e, "smoke-test failed");
                process::exit(1);
            }
        }
        _ => {
            eprintln!("Usage: trtllm-client <subcommand>");
            eprintln!("  smoke-test    Run inference against all three capabilities");
            process::exit(1);
        }
    }
}

async fn run_smoke_test() -> Result<(), String> {
    let client = HttpLlmClient::from_env();

    // Probe /v1/models before sending any inference requests.
    tracing::info!("Probing /v1/models...");
    let loaded = client
        .probe_models()
        .await
        .map_err(|e| format!("Startup probe failed: {}", e))?;
    tracing::info!(models = ?loaded, "Models available");

    // Only test capabilities whose ensemble model is currently loaded.
    // This allows smoke-all to load one pipeline at a time and test it.
    let is_loaded = |cap: Capability| loaded.contains(&cap.model_id().to_string());

    let mut all_ok = true;
    let mut tests_run = 0u32;

    // --- Chat ---
    if is_loaded(Capability::Chat) {
        tests_run += 1;
        all_ok &= run_request(
            &client,
            make_request(None, None, "Hello! What can you help me with today?"),
            Capability::Chat,
        )
        .await;
    } else {
        tracing::info!("Skipping Chat — {} not loaded", Capability::Chat.model_id());
    }

    // --- Code ---
    if is_loaded(Capability::Code) {
        tests_run += 1;
        all_ok &= run_request(
            &client,
            make_request(
                Some("code"),
                None,
                "Write a Rust function that returns the nth Fibonacci number.",
            ),
            Capability::Code,
        )
        .await;
    } else {
        tracing::info!("Skipping Code — {} not loaded", Capability::Code.model_id());
    }

    // --- Caption ---
    // Caption (LLaVA 1.5) requires pre-processed FP16 pixel values via the Triton
    // binary HTTP extension.  The OpenAI /v1/chat/completions endpoint does not
    // support image_url for this model_type, so Caption is tested separately by
    // `just smoke-vision` (scripts/vision-smoke-test.py).
    if is_loaded(Capability::Caption) {
        tracing::info!(
            "Skipping Caption in Rust smoke-test — use `just smoke-vision` for vision testing"
        );
    } else {
        tracing::info!(
            "Skipping Caption — {} not loaded",
            Capability::Caption.model_id()
        );
    }

    if tests_run == 0 {
        // Caption is tested by smoke-vision.py; if only vision is loaded that's fine.
        let has_vision = loaded.contains(&Capability::Caption.model_id().to_string());
        if has_vision {
            tracing::info!(
                "No text capabilities loaded — vision pipeline present, delegating to smoke-vision"
            );
            return Ok(());
        }
        return Err(format!(
            "smoke-test FAILED — no capabilities loaded. Expected one of: {}, {}, {}",
            Capability::Chat.model_id(),
            Capability::Code.model_id(),
            Capability::Caption.model_id(),
        ));
    }

    if all_ok {
        tracing::info!(
            tests_run,
            "smoke-test PASSED — all loaded capabilities responded"
        );
        Ok(())
    } else {
        Err("smoke-test FAILED — one or more loaded capabilities did not respond".to_string())
    }
}

/// Send one request, print the response, log it, and return success/fail.
async fn run_request(
    client: &HttpLlmClient,
    req: InferenceRequest,
    expected_capability: Capability,
) -> bool {
    let capability_label = expected_capability.to_string();
    tracing::info!(capability = %capability_label, "sending request");

    match client.infer(req).await {
        Ok(resp) => {
            println!("[{}] {}", capability_label, resp.content);
            log_request(&resp, true, None);
            true
        }
        Err(e) => {
            log_request_error("unknown", expected_capability, 0, &e);
            eprintln!("[{}] ERROR: {}", capability_label, e);
            false
        }
    }
}

fn make_request(
    system_hint: Option<&str>,
    image: Option<&str>,
    user_text: &str,
) -> InferenceRequest {
    InferenceRequest {
        id: Uuid::new_v4().to_string(),
        messages: vec![Message {
            role: "user".to_string(),
            content: MessageContent::Text(user_text.to_string()),
        }],
        image: image.map(str::to_string),
        system_hint: system_hint.map(str::to_string),
        max_tokens: Some(256),
        temperature: Some(0.7),
    }
}

#[allow(dead_code)] // kept for future vision-via-OpenAI testing
fn make_image_request(image_b64: &str, prompt: &str) -> InferenceRequest {
    InferenceRequest {
        id: Uuid::new_v4().to_string(),
        messages: vec![Message {
            role: "user".to_string(),
            content: MessageContent::Text(prompt.to_string()),
        }],
        image: Some(image_b64.to_string()),
        system_hint: None,
        max_tokens: Some(128),
        temperature: Some(0.3),
    }
}
