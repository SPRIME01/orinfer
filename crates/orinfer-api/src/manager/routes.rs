use super::*;
use axum::{
    Json, Router,
    body::Body,
    extract::{Query, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use futures_util::StreamExt;
use serde::Deserialize;
use std::future::IntoFuture;

fn error(status: StatusCode, message: impl Into<String>) -> Response {
    crate::server::error(status, message)
}
fn result(value: Result<Value>) -> Response {
    match value {
        Ok(value) => Json(value).into_response(),
        Err(e) => error(StatusCode::BAD_REQUEST, e),
    }
}
fn authorized(headers: &HeaderMap) -> bool {
    std::env::var("ORINFER_API_KEY")
        .ok()
        .filter(|s| !s.is_empty())
        .is_none_or(|key| {
            headers
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.strip_prefix("Bearer "))
                == Some(key.as_str())
        })
}
async fn models(State(manager): State<Arc<Manager>>, headers: HeaderMap) -> Response {
    if !authorized(&headers) {
        return error(StatusCode::UNAUTHORIZED, "Invalid API key");
    }
    Json(json!({"object":"list","data":manager.records.lock().await.keys().map(|name| json!({"id":name,"object":"model","owned_by":"orinfer","created":0})).collect::<Vec<_>>()})).into_response()
}
async fn completions(
    State(manager): State<Arc<Manager>>,
    headers: HeaderMap,
    body: Body,
) -> Response {
    if !authorized(&headers) {
        return error(StatusCode::UNAUTHORIZED, "Invalid API key");
    }
    let Ok(slot) = Arc::clone(&manager.requests).try_acquire_owned() else {
        return error(
            StatusCode::TOO_MANY_REQUESTS,
            "Global request queue is full",
        );
    };
    let Ok(mut memory) = Arc::clone(&manager.body_memory)
        .acquire_many_owned(96)
        .await
    else {
        return error(StatusCode::SERVICE_UNAVAILABLE, "Service is stopping");
    };
    let bytes = match axum::body::to_bytes(body, 32 << 20).await {
        Ok(b) => b,
        Err(e) => return error(StatusCode::PAYLOAD_TOO_LARGE, e.to_string()),
    };
    let charge = (bytes.len().saturating_mul(3).div_ceil(1 << 20)).clamp(1, 96);
    let input_bytes = bytes.len();
    drop(memory.split(96 - charge));
    let mut request: Value = match serde_json::from_slice(&bytes) {
        Ok(v) => v,
        Err(e) => return error(StatusCode::BAD_REQUEST, e.to_string()),
    };
    drop(bytes);
    let Some(name) = request
        .get("model")
        .and_then(Value::as_str)
        .map(str::to_owned)
    else {
        return error(StatusCode::BAD_REQUEST, "model is required");
    };
    if !manager.records.lock().await.contains_key(&name) {
        return error(StatusCode::NOT_FOUND, format!("Unknown model: {name}"));
    }
    let (mut lease, record) = match manager.acquire(&name).await {
        Ok(value) => value,
        Err(e) => return error(StatusCode::SERVICE_UNAVAILABLE, e),
    };
    if let Err(message) =
        crate::server::defaults::merge(&mut request, &record.options.default_request_params)
    {
        return error(StatusCode::BAD_REQUEST, message);
    }
    request["model"] = json!(lease.0.model);
    let started = Instant::now();
    loop {
        match manager.reserve_request(&record, &request, input_bytes) {
            Ok(budget) => {
                lease.1 = Some(budget);
                break;
            }
            Err(message) => {
                if manager.stopping.load(Ordering::Acquire)
                    || started.elapsed() >= Duration::from_secs(manager.config.load_timeout_s)
                {
                    return error(StatusCode::SERVICE_UNAVAILABLE, message);
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }
    }
    let mut upstream = lease
        .0
        .client
        .post("http://localhost/v1/chat/completions")
        .json(&request);
    for key in ["authorization", "x-request-id"] {
        if let Some(value) = headers.get(key) {
            upstream = upstream.header(key, value);
        }
    }
    let response = match upstream.send().await {
        Ok(r) => r,
        Err(e) => return error(StatusCode::BAD_GATEWAY, e.to_string()),
    };
    let status = response.status();
    let response_headers = response.headers().clone();
    // The response stream retains the model lease and memory permits. Dropping
    // it cancels the upstream request and propagates backpressure without buffering SSE.
    let stream = futures_util::stream::unfold(
        (Box::pin(response.bytes_stream()), lease, slot, memory),
        |(mut stream, lease, slot, memory)| async {
            stream
                .next()
                .await
                .map(|chunk| (chunk, (stream, lease, slot, memory)))
        },
    );
    let mut output = Response::new(Body::from_stream(stream));
    *output.status_mut() = status;
    for name in ["content-type", "cache-control", "x-request-id"] {
        if let Some(value) = response_headers.get(name) {
            output.headers_mut().insert(name, value.clone());
        }
    }
    output
}

#[derive(Deserialize)]
struct Import {
    name: String,
    path: PathBuf,
    #[serde(default)]
    options: Option<WorkerOptions>,
}
async fn import(State(manager): State<Arc<Manager>>, Json(input): Json<Import>) -> Response {
    let options = input
        .options
        .unwrap_or_else(|| manager.config.worker_options.clone());
    let record = match tokio::task::spawn_blocking(move || {
        registry::inspect(&input.name, &input.path, options)
    })
    .await
    {
        Ok(Ok(record)) => record,
        Ok(Err(e)) => return result(Err(e)),
        Err(e) => return result(Err(e.to_string())),
    };
    let mut records = manager.records.lock().await;
    let mut updated = records.clone();
    updated.insert(record.name.clone(), record.clone());
    if let Err(e) = registry::write(&manager.config.data_dir, &updated) {
        return result(Err(e));
    }
    *records = updated;
    result(Ok(json!(record)))
}
async fn pull_model(
    State(manager): State<Arc<Manager>>,
    Json(mut input): Json<super::pull::Pull>,
) -> Response {
    if input.options.is_none() {
        input.options = Some(manager.config.worker_options.clone());
    }
    let data = manager.config.data_dir.clone();
    let record = match tokio::task::spawn_blocking(move || super::pull::pull(&data, input)).await {
        Ok(Ok(record)) => record,
        Ok(Err(error)) => return result(Err(error)),
        Err(error) => return result(Err(error.to_string())),
    };
    let mut records = manager.records.lock().await;
    let mut updated = records.clone();
    updated.insert(record.name.clone(), record.clone());
    if let Err(error) = registry::write(&manager.config.data_dir, &updated) {
        return result(Err(error));
    }
    *records = updated;
    result(Ok(json!(record)))
}
#[derive(Deserialize)]
struct Name {
    name: String,
}
#[derive(Deserialize)]
struct Load {
    name: String,
    keep_alive_s: Option<u64>,
}
async fn load(State(manager): State<Arc<Manager>>, Json(input): Json<Load>) -> Response {
    match manager.acquire(&input.name).await {
        Ok((lease, _)) => {
            if let Some(seconds) = input.keep_alive_s {
                lease.0.keep_alive_s.store(seconds, Ordering::Relaxed);
            }
            result(Ok(
                json!({"model":input.name,"pid":lease.0.pid,"state":"ready"}),
            ))
        }
        Err(e) => result(Err(e)),
    }
}
#[derive(Deserialize)]
struct Stop {
    name: String,
    #[serde(default)]
    force: bool,
}
async fn stop(State(manager): State<Arc<Manager>>, Json(input): Json<Stop>) -> Response {
    result(
        manager
            .stop(&input.name, input.force)
            .await
            .map(|_| json!({"model":input.name,"state":"unloaded"})),
    )
}
async fn remove(State(manager): State<Arc<Manager>>, Json(input): Json<Name>) -> Response {
    if let Err(e) = manager.stop(&input.name, false).await {
        return result(Err(e));
    }
    let mut records = manager.records.lock().await;
    let mut updated = records.clone();
    updated.remove(&input.name);
    if let Err(e) = registry::write(&manager.config.data_dir, &updated) {
        return result(Err(e));
    }
    *records = updated;
    result(Ok(json!({"removed":input.name})))
}
async fn show(State(manager): State<Arc<Manager>>, Query(input): Query<Name>) -> Response {
    result(
        manager
            .records
            .lock()
            .await
            .get(&input.name)
            .map(|r| json!(r))
            .ok_or_else(|| format!("Unknown model: {}", input.name)),
    )
}
async fn list(State(manager): State<Arc<Manager>>) -> Json<Value> {
    Json(json!(
        manager.records.lock().await.values().collect::<Vec<_>>()
    ))
}
async fn ps(State(manager): State<Arc<Manager>>) -> Json<Value> {
    Json(manager.status().await)
}

#[derive(Deserialize)]
struct Logs {
    name: String,
    #[serde(default = "log_lines")]
    tail: usize,
}
fn log_lines() -> usize {
    100
}
async fn logs(State(manager): State<Arc<Manager>>, Json(input): Json<Logs>) -> Response {
    use std::io::{Read, Seek, SeekFrom};
    let record = manager.records.lock().await.get(&input.name).cloned();
    let Some(record) = record else {
        return result(Err("Unknown model".into()));
    };
    let file = manager
        .directory
        .join(&runner_key(&record)[..16])
        .join("worker.log");
    result((|| {
        let mut file = fs::File::open(file).map_err(|e| e.to_string())?;
        let length = file.metadata().map_err(|e| e.to_string())?.len();
        file.seek(SeekFrom::Start(length.saturating_sub(1 << 20)))
            .map_err(|e| e.to_string())?;
        let mut bytes = Vec::new();
        file.take(1 << 20)
            .read_to_end(&mut bytes)
            .map_err(|e| e.to_string())?;
        let text = String::from_utf8_lossy(&bytes);
        let lines: Vec<_> = text.lines().rev().take(input.tail.min(10000)).collect();
        Ok(json!({"text":lines.into_iter().rev().collect::<Vec<_>>().join("\n")+"\n"}))
    })())
}

#[derive(Deserialize)]
struct Batch {
    name: String,
    input: PathBuf,
    output: PathBuf,
    #[serde(default)]
    resume: bool,
    #[serde(default = "batch_order")]
    order: String,
    #[serde(default)]
    options: serde_json::Map<String, Value>,
    #[serde(default = "throughput")]
    scheduler_policy: String,
}
fn throughput() -> String {
    "throughput".into()
}
fn batch_order() -> String {
    "auto".into()
}
async fn batch(State(manager): State<Arc<Manager>>, Json(input): Json<Batch>) -> Response {
    if input
        .scheduler_policy
        .parse::<orinfer_engine::scheduler::Policy>()
        .is_err()
    {
        return result(Err("Invalid scheduler policy".into()));
    }
    if !input.input.is_absolute()
        || !input.output.is_absolute()
        || !matches!(input.order.as_str(), "auto" | "input")
    {
        return result(Err(
            "Batch requires absolute file paths and auto/input order".into(),
        ));
    }
    if let Err(e) = manager.stop(&input.name, false).await {
        return result(Err(e));
    }
    let prepared = async {
        let _transition = manager
            .transition
            .acquire()
            .await
            .map_err(|_| "Service is stopping")?;
        if manager.stopping.load(Ordering::Acquire) {
            return Err("Service is stopping".into());
        }
        let record = manager
            .records
            .lock()
            .await
            .get(&input.name)
            .cloned()
            .ok_or("Unknown model")?;
        let key = runner_key(&record);
        let mut values = serde_json::to_value(&record.options).map_err(|e| e.to_string())?;
        values
            .as_object_mut()
            .ok_or("Invalid model options")?
            .extend(input.options.clone());
        let options: WorkerOptions = serde_json::from_value(values).map_err(|e| e.to_string())?;
        options.settings(&record.name, &record.path)?.validate()?;
        if manager.runners.lock().await.contains_key(&key) {
            return Err("Model acquired by another request; retry the batch job".into());
        }
        manager.reserve_space(&record).await?;
        let directory = manager.directory.join(&key[..16]);
        private_directory(&directory)?;
        let log = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(directory.join("worker.log"))
            .map_err(|e| e.to_string())?;
        let report =
            fs::File::create(directory.join("batch-report.json")).map_err(|e| e.to_string())?;
        let mut command = manager.command();
        command
            .arg("batch")
            .arg(&record.path)
            .arg("--model")
            .arg(&record.name)
            .arg("--input")
            .arg(&input.input)
            .arg("--output")
            .arg(&input.output)
            .arg("--batch-order")
            .arg(&input.order)
            .arg("--scheduler-policy")
            .arg(&input.scheduler_policy)
            .arg("--gpu-lock")
            .arg(directory.join("gpu.lock"))
            .args(options.arguments())
            .stdin(std::process::Stdio::null())
            .stdout(report)
            .stderr(log);
        if input.resume {
            command.arg("--resume");
        }
        let child = command.spawn().map_err(|e| e.to_string())?;
        let pid = child.id().ok_or("Batch worker did not start")?;
        let runner = Arc::new(Runner {
            key: key.clone(),
            model: record.name,
            child: Mutex::new(child),
            client: reqwest::Client::new(),
            active: AtomicUsize::new(1),
            last_used: SyncMutex::new(Instant::now()),
            keep_alive_s: AtomicU64::new(0),
            draining: AtomicBool::new(false),
            directory,
            pid,
            batch: true,
        });
        manager
            .runners
            .lock()
            .await
            .insert(key, Arc::clone(&runner));
        Ok(Lease(runner, None))
    }
    .await;
    let lease = match prepared {
        Ok(lease) => lease,
        Err(e) => {
            manager.release_gpu_if_idle().await;
            return result(Err(e));
        }
    };
    let status = loop {
        match lease.0.child.lock().await.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {}
            Err(e) => return result(Err(e.to_string())),
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    };
    let report = fs::read(lease.0.directory.join("batch-report.json"))
        .map_err(|e| e.to_string())
        .and_then(|raw| serde_json::from_slice::<Value>(&raw).map_err(|e| e.to_string()));
    let _transition = manager.transition.acquire().await;
    if let Err(e) = manager.remove_runner(&lease.0).await {
        return result(Err(e));
    }
    match report {
        Ok(report) => Json(json!({"success":status.success(),"report":report})).into_response(),
        Err(e) => result(Err(format!("Batch worker {status}: {e}"))),
    }
}

pub(super) fn public_router(manager: Arc<Manager>) -> Router {
    Router::new()
        .route(
            "/health",
            get(|| async {
                Json(json!({"status":"ready","service":"orinfer","lazy_loading":true}))
            }),
        )
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(completions))
        .with_state(manager)
}

pub(super) async fn serve(manager: Arc<Manager>) -> Result<()> {
    let socket = manager.directory.join("control.sock");
    if socket.exists() {
        fs::remove_file(&socket).map_err(|e| e.to_string())?;
    }
    let local = tokio::net::UnixListener::bind(&socket).map_err(|e| e.to_string())?;
    let public = tokio::net::TcpListener::bind(&manager.config.listen)
        .await
        .map_err(|e| e.to_string())?;
    let api = public_router(Arc::clone(&manager));
    let control = Router::new()
        .route("/models", get(list))
        .route("/ps", get(ps))
        .route("/show", get(show))
        .route("/v1/chat/completions", post(completions))
        .route("/import", post(import))
        .route("/pull", post(pull_model))
        .route("/load", post(load))
        .route("/stop", post(stop))
        .route("/remove", post(remove))
        .route("/logs", post(logs))
        .route("/batch", post(batch))
        .with_state(Arc::clone(&manager));
    let (shutdown, wait) = tokio::sync::watch::channel(false);
    let mut public_wait = wait.clone();
    let mut local_wait = wait;
    let mut public_task = tokio::spawn(
        axum::serve(public, api)
            .with_graceful_shutdown(async move {
                let _ = public_wait.changed().await;
            })
            .into_future(),
    );
    let mut local_task = tokio::spawn(
        axum::serve(local, control)
            .with_graceful_shutdown(async move {
                let _ = local_wait.changed().await;
            })
            .into_future(),
    );
    let sweep_manager = Arc::clone(&manager);
    let sweeper = tokio::spawn(async move {
        while !sweep_manager.stopping.load(Ordering::Acquire) {
            tokio::time::sleep(Duration::from_secs(1)).await;
            sweep_manager.sweep().await;
        }
    });
    eprintln!(
        "API READY http://{}/v1; lazy_loading=true; control={}",
        manager.config.listen,
        socket.display()
    );
    let outcome = tokio::select! {
        _ = crate::server::shutdown_signal() => Ok(()),
        result = &mut public_task => Err(format!("Public listener exited: {result:?}")),
        result = &mut local_task => Err(format!("Control listener exited: {result:?}")),
    };
    manager.stopping.store(true, Ordering::Release);
    manager.requests.close();
    manager.body_memory.close();
    let _ = shutdown.send(true);
    sweeper.abort();
    // Readers can finish while draining; model processes remain alive until then.
    let drained = async {
        if !public_task.is_finished() {
            let _ = (&mut public_task).await;
        }
        if !local_task.is_finished() {
            let _ = (&mut local_task).await;
        }
    };
    if tokio::time::timeout(Duration::from_secs(30), drained)
        .await
        .is_err()
    {
        public_task.abort();
        local_task.abort();
    }
    let _transition = manager
        .transition
        .acquire()
        .await
        .map_err(|_| "Manager stopped")?;
    let runners: Vec<_> = manager.runners.lock().await.values().cloned().collect();
    for runner in runners {
        manager.remove_runner(&runner).await?;
    }
    let _ = fs::remove_file(socket);
    outcome
}
