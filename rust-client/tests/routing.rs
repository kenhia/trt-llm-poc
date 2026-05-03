// TDD stub — written before capability.rs (T014).
// This file will not compile until T014 provides `Capability` and `resolve_capability`.
// Once T014 is implemented these tests must all pass: `cargo test --test routing`

use rust_client::capability::{Capability, resolve_capability};
use rust_client::models::{InferenceRequest, Message, MessageContent};
use uuid::Uuid;

fn req(system_hint: Option<&str>, image: Option<&str>) -> InferenceRequest {
    InferenceRequest {
        id: Uuid::new_v4().to_string(),
        messages: vec![Message {
            role: "user".to_string(),
            content: MessageContent::Text("test".to_string()),
        }],
        image: image.map(|s| s.to_string()),
        system_hint: system_hint.map(|s| s.to_string()),
        max_tokens: None,
        temperature: None,
    }
}

#[test]
fn image_present_routes_to_caption() {
    let r = req(None, Some("base64imagedata"));
    assert!(matches!(resolve_capability(&r), Capability::Caption));
}

#[test]
fn system_hint_code_routes_to_code() {
    let r = req(Some("code"), None);
    assert!(matches!(resolve_capability(&r), Capability::Code));
}

#[test]
fn system_hint_code_uppercase_routes_to_code() {
    let r = req(Some("Please use CODE style"), None);
    assert!(matches!(resolve_capability(&r), Capability::Code));
}

#[test]
fn no_image_no_hint_routes_to_chat() {
    let r = req(None, None);
    assert!(matches!(resolve_capability(&r), Capability::Chat));
}

#[test]
fn empty_hint_routes_to_chat() {
    let r = req(Some(""), None);
    assert!(matches!(resolve_capability(&r), Capability::Chat));
}

#[test]
fn image_plus_code_hint_routes_to_caption_image_wins() {
    // Image presence takes precedence over system_hint per data-model.md routing rules.
    let r = req(Some("code"), Some("base64imagedata"));
    assert!(matches!(resolve_capability(&r), Capability::Caption));
}

#[test]
fn hint_without_code_routes_to_chat() {
    // A system_hint that doesn't contain "code" should not trigger Code routing.
    let r = req(Some("be helpful and concise"), None);
    assert!(matches!(resolve_capability(&r), Capability::Chat));
}
