//! Ollama wire protocol over the shared catalog, admission and Chat worker.
mod metadata;
pub(super) mod request;
pub(super) mod response;

use super::*;
use axum::{
    Router,
    body::Body,
    extract::State,
    http::HeaderMap,
    response::Response,
    routing::{delete, get, post},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    Chat,
    Generate,
}

pub(super) async fn resolve(manager: &Manager, name: &str) -> String {
    let records = manager.records.lock().await;
    if records.contains_key(name) {
        return name.into();
    }
    name.strip_suffix(":latest")
        .filter(|s| records.contains_key(*s))
        .unwrap_or(name)
        .into()
}

async fn chat(State(manager): State<Arc<Manager>>, headers: HeaderMap, body: Body) -> Response {
    infer(manager, headers, body, Kind::Chat).await
}
async fn generate(State(manager): State<Arc<Manager>>, headers: HeaderMap, body: Body) -> Response {
    infer(manager, headers, body, Kind::Generate).await
}
async fn infer(manager: Arc<Manager>, headers: HeaderMap, body: Body, kind: Kind) -> Response {
    // Adapt the streamed body itself, retaining the gateway's lease and permits
    // until the native client consumes the response or disconnects.
    let started = Instant::now();
    let upstream = routes::gateway(manager, headers, body, Some(kind)).await;
    response::adapt(upstream, kind, started).await
}

pub(super) fn router() -> Router<Arc<Manager>> {
    Router::new()
        .route("/", get(|| async { "Orinfer is running" }))
        .route("/api/version", get(metadata::version))
        .route("/api/tags", get(metadata::tags))
        .route("/api/show", post(metadata::show))
        .route("/api/ps", get(metadata::ps))
        .route("/api/copy", post(metadata::copy))
        .route("/api/delete", delete(metadata::delete))
        .route("/api/chat", post(chat))
        .route("/api/generate", post(generate))
        .route("/api/{*path}", axum::routing::any(metadata::unsupported))
}
