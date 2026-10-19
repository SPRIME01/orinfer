use super::*;
use axum::{
    Json,
    extract::{State, rejection::JsonRejection},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use std::time::{SystemTime, UNIX_EPOCH};

fn denied(headers: &HeaderMap) -> Option<Response> {
    (!routes::authorized(headers))
        .then(|| response::error(StatusCode::UNAUTHORIZED, "Invalid API key"))
}
pub(super) async fn version(headers: HeaderMap) -> Response {
    if let Some(e) = denied(&headers) {
        return e;
    }
    Json(json!({"version":env!("CARGO_PKG_VERSION"),"server":"orinfer"})).into_response()
}
pub(super) async fn unsupported(headers: HeaderMap) -> Response {
    if let Some(e) = denied(&headers) {
        return e;
    }
    response::error(
        StatusCode::NOT_IMPLEMENTED,
        "This Ollama operation is not supported; import native orinfer models with the CLI",
    )
}
fn modified(record: &ModelRecord) -> String {
    response::timestamp(
        fs::metadata(record.path.join("cache/model.json"))
            .and_then(|m| m.modified())
            .unwrap_or(UNIX_EPOCH),
    )
}
fn details(record: &ModelRecord) -> Value {
    json!({"format":"safetensors","family":record.architecture,"families":[record.architecture]})
}
pub(super) async fn tags(State(manager): State<Arc<Manager>>, headers: HeaderMap) -> Response {
    if let Some(e) = denied(&headers) {
        return e;
    }
    let models:Vec<_>=manager.records.lock().await.values().map(|r| json!({
        "name":r.name,"model":r.name,"modified_at":modified(r),"size":r.weight_bytes,"digest":r.identity,"details":details(r)
    })).collect();
    Json(json!({"models":models})).into_response()
}
fn input(
    input: std::result::Result<Json<Value>, JsonRejection>,
) -> std::result::Result<Value, (StatusCode, String)> {
    input
        .map(|Json(value)| value)
        .map_err(|e| (e.status(), e.body_text()))
}
pub(super) async fn show(
    State(manager): State<Arc<Manager>>,
    headers: HeaderMap,
    body: std::result::Result<Json<Value>, JsonRejection>,
) -> Response {
    if let Some(e) = denied(&headers) {
        return e;
    }
    let value = match input(body) {
        Ok(v) => v,
        Err((status, message)) => return response::error(status, message),
    };
    let Some(name) = value
        .get("model")
        .or_else(|| value.get("name"))
        .and_then(Value::as_str)
    else {
        return response::error(StatusCode::BAD_REQUEST, "model is required");
    };
    let name = resolve(&manager, name).await;
    let Some(record) = manager.records.lock().await.get(&name).cloned() else {
        return response::error(StatusCode::NOT_FOUND, format!("Unknown model: {name}"));
    };
    match tokio::task::spawn_blocking(move || show_record(&record)).await {
        Ok(Ok(value)) => Json(value).into_response(),
        Ok(Err(e)) => response::error(StatusCode::INTERNAL_SERVER_ERROR, e),
        Err(e) => response::error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}
fn read_json(path: PathBuf) -> Result<Value> {
    use std::io::Read;
    let mut bytes = Vec::new();
    File::open(path)
        .map_err(|e| e.to_string())?
        .take((8 << 20) + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() > 8 << 20 {
        return Err("Model metadata exceeds 8 MiB".into());
    }
    serde_json::from_slice(&bytes).map_err(|e| e.to_string())
}
fn show_record(record: &ModelRecord) -> Result<Value> {
    let config = read_json(record.path.join("config.json"))?;
    let tokenizer = read_json(record.path.join("tokenizer_config.json"))?;
    let text = config.get("text_config").unwrap_or(&config);
    let mut info = json!({"general.architecture":record.architecture});
    for (target, source) in [
        ("context_length", "max_position_embeddings"),
        ("embedding_length", "hidden_size"),
        ("block_count", "num_hidden_layers"),
        ("attention.head_count", "num_attention_heads"),
        ("attention.head_count_kv", "num_key_value_heads"),
    ] {
        if let Some(value) = text.get(source) {
            info[format!("{}.{target}", record.architecture)] = value.clone();
        }
    }
    // The registered execution contract, rather than a source checkpoint's
    // theoretical window, determines the usable context.
    info[format!("{}.context_length", record.architecture)] = json!(record.max_context);
    let mut capabilities = vec!["completion", "tools", "thinking"];
    if config.get("vision_config").is_some_and(|v| !v.is_null()) {
        capabilities.push("vision");
    }
    Ok(
        json!({"modified_at":modified(record),"details":details(record),"model_info":info,
        "template":tokenizer.get("chat_template").and_then(Value::as_str).unwrap_or(""),
        "parameters":"","capabilities":capabilities,
        "orinfer":{"execution_package":record.execution_package,"quantization":config.get("quantization_config"),"default_request_params":record.options.default_request_params}}),
    )
}
pub(super) async fn ps(State(manager): State<Arc<Manager>>, headers: HeaderMap) -> Response {
    if let Some(e) = denied(&headers) {
        return e;
    }
    let runners: Vec<_> = manager.runners.lock().await.values().cloned().collect();
    let records = manager.records.lock().await;
    let models: Vec<_> = runners.iter().filter_map(|runner| {
        let record = records.values().find(|r| runner_key(r) == runner.key)?;
        let keep_alive = runner.keep_alive_s.load(Ordering::Relaxed);
        let idle = runner.last_used.lock().unwrap_or_else(|e| e.into_inner()).elapsed().as_secs();
        let remaining = if runner.active.load(Ordering::Acquire) > 0 {
            keep_alive
        } else {
            keep_alive.saturating_sub(idle)
        };
        let expires = SystemTime::now().checked_add(Duration::from_secs(remaining))
            .filter(|time| keep_alive != u64::MAX && *time <= UNIX_EPOCH + Duration::from_secs(253402300799))
            .map(response::timestamp).unwrap_or_else(|| "9999-12-31T23:59:59Z".into());
        Some(json!({"name":runner.model,"model":runner.model,"size":record.weight_bytes,"digest":record.identity,
            "details":details(record),"expires_at":expires,"context_length":record.max_context,
            "estimated_load_bytes":record.load_bytes,"active_requests":runner.active.load(Ordering::Acquire)}))
    }).collect();
    Json(json!({"models":models})).into_response()
}

pub(super) async fn copy(
    State(manager): State<Arc<Manager>>,
    headers: HeaderMap,
    body: std::result::Result<Json<Value>, JsonRejection>,
) -> Response {
    if let Some(e) = denied(&headers) {
        return e;
    }
    let value = match input(body) {
        Ok(v) => v,
        Err((status, message)) => return response::error(status, message),
    };
    let (Some(source), Some(destination)) =
        (value["source"].as_str(), value["destination"].as_str())
    else {
        return response::error(
            StatusCode::BAD_REQUEST,
            "source and destination are required",
        );
    };
    if let Err(e) = registry::validate_name(destination) {
        return response::error(StatusCode::BAD_REQUEST, e);
    }
    let source = resolve(&manager, source).await;
    let mut records = manager.records.lock().await;
    let Some(mut record) = records.get(&source).cloned() else {
        return response::error(StatusCode::NOT_FOUND, format!("Unknown model: {source}"));
    };
    record.name = destination.into();
    let mut updated = records.clone();
    updated.insert(destination.into(), record);
    if let Err(e) = registry::write(&manager.config.data_dir, &updated) {
        return response::error(StatusCode::INTERNAL_SERVER_ERROR, e);
    }
    *records = updated;
    StatusCode::OK.into_response()
}
pub(super) async fn delete(
    State(manager): State<Arc<Manager>>,
    headers: HeaderMap,
    body: std::result::Result<Json<Value>, JsonRejection>,
) -> Response {
    if let Some(e) = denied(&headers) {
        return e;
    }
    let value = match input(body) {
        Ok(v) => v,
        Err((status, message)) => return response::error(status, message),
    };
    let Some(name) = value
        .get("model")
        .or_else(|| value.get("name"))
        .and_then(Value::as_str)
    else {
        return response::error(StatusCode::BAD_REQUEST, "model is required");
    };
    let name = resolve(&manager, name).await;
    if !manager.records.lock().await.contains_key(&name) {
        return response::error(StatusCode::NOT_FOUND, format!("Unknown model: {name}"));
    }
    if let Err(e) = manager.stop(&name, false).await {
        return response::error(StatusCode::SERVICE_UNAVAILABLE, e);
    }
    let mut records = manager.records.lock().await;
    let mut updated = records.clone();
    updated.remove(&name);
    if let Err(e) = registry::write(&manager.config.data_dir, &updated) {
        return response::error(StatusCode::INTERNAL_SERVER_ERROR, e);
    }
    *records = updated;
    StatusCode::OK.into_response()
}
