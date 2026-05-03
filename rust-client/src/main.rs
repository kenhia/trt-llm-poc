use std::process;

use tracing_subscriber::{fmt, EnvFilter};
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

    let mut all_ok = true;

    // --- Chat ---
    all_ok &= run_request(
        &client,
        make_request(None, None, "Hello! What can you help me with today?"),
        Capability::Chat,
    )
    .await;

    // --- Code ---
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

    // --- Caption ---
    // Use a minimal 1x1 white JPEG base64 as a stand-in image for the smoke test.
    // Replace with a real image path/URL for meaningful output.
    let stub_image = "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8U\
                      HRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAARC\
                      AABAAEDASIA2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoH\
                      BwYIDAoMCwsKCwsNCxAQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/\
                      AABEIAAIBAAMB/8QAFgABAQEAAAAAAAAAAAAAAAAABgUE/8QAIhAAAQMEAgMB\
                      AAAAAAAAAAAAAQIDBAUREiExQWH/xAAUAQEAAAAAAAAAAAAAAAAAAAAA/8QAFBEB\
                      AAAAAAAAAAAAAAAAAAAAAP/aAAwDAQACEQMRAD8Amo3DcLkuirqKuequmpZal\
                      5lTM8pOajqPUkkkkkkn/2Q==";
    all_ok &= run_request(
        &client,
        make_image_request(stub_image, "Describe this image briefly."),
        Capability::Caption,
    )
    .await;

    if all_ok {
        tracing::info!("smoke-test PASSED — all three capabilities responded");
        Ok(())
    } else {
        Err("smoke-test FAILED — one or more capabilities did not respond".to_string())
    }
}

/// Send one request, print the response, log it, and return success/fail.
async fn run_request(client: &HttpLlmClient, req: InferenceRequest, expected_capability: Capability) -> bool {
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

fn make_request(system_hint: Option<&str>, image: Option<&str>, user_text: &str) -> InferenceRequest {
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
