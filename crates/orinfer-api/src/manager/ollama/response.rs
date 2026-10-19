use super::*;
use axum::{
    Json,
    body::{Body, Bytes},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use futures_util::{Stream, StreamExt};
use std::{collections::VecDeque, pin::Pin, time::SystemTime};

const LIMIT: usize = 1 << 20;
#[derive(Clone)]
pub(in crate::manager) struct Context {
    pub model: String,
}
#[derive(Clone)]
struct NativeResponse;

pub(super) fn timestamp(time: SystemTime) -> String {
    humantime::format_rfc3339_nanos(time).to_string()
}
pub(super) fn error(status: StatusCode, message: impl Into<String>) -> Response {
    (status, Json(json!({"error":message.into()}))).into_response()
}
fn base(kind: Kind, model: &str, done: bool) -> Value {
    let mut value = json!({"model":model,"created_at":timestamp(SystemTime::now()),"done":done});
    match kind {
        Kind::Chat => value["message"] = json!({"role":"assistant","content":""}),
        Kind::Generate => value["response"] = json!(""),
    }
    value
}
fn encoded(value: &Value) -> Bytes {
    let mut bytes = serde_json::to_vec(value).expect("JSON value serializes");
    bytes.push(b'\n');
    Bytes::from(bytes)
}
pub(in crate::manager) fn lifecycle(
    kind: Kind,
    model: &str,
    reason: &str,
    stream: bool,
) -> Response {
    let mut value = base(kind, model, true);
    value["done_reason"] = json!(reason);
    let mut response = if stream {
        ([("content-type", "application/x-ndjson")], encoded(&value)).into_response()
    } else {
        Json(value).into_response()
    };
    response.extensions_mut().insert(NativeResponse);
    response
}
fn metrics(output: &mut Value, usage: &Value, started: Instant) {
    output["total_duration"] = json!(started.elapsed().as_nanos().min(u64::MAX as u128) as u64);
    for (target, source) in [
        ("prompt_eval_count", "prompt_tokens"),
        ("eval_count", "completion_tokens"),
    ] {
        if let Some(value) = usage.get(source) {
            output[target] = value.clone();
        }
    }
    if let Some(cached) = usage.pointer("/prompt_tokens_details/cached_tokens") {
        output["prompt_eval_cached_count"] = cached.clone();
    }
}
fn calls(input: &Value) -> Result<Value> {
    let mut result = Vec::new();
    for (index, call) in input
        .as_array()
        .ok_or("Invalid upstream tool_calls")?
        .iter()
        .enumerate()
    {
        let args: Value = serde_json::from_str(
            call["function"]["arguments"]
                .as_str()
                .ok_or("Missing upstream tool arguments")?,
        )
        .map_err(|e| format!("Incomplete upstream tool arguments: {e}"))?;
        if !args.is_object() || call["function"]["name"].as_str().is_none_or(str::is_empty) {
            return Err("Invalid upstream tool function".into());
        }
        result.push(json!({"id":call["id"],"function":{"index":index,"name":call["function"]["name"],"arguments":args}}));
    }
    Ok(json!(result))
}
fn message(output: &mut Value, input: &Value, kind: Kind) -> Result<()> {
    let target = match kind {
        Kind::Chat => &mut output["message"],
        Kind::Generate => &mut *output,
    };
    if let Some(text) = input["content"].as_str() {
        target[if kind == Kind::Chat {
            "content"
        } else {
            "response"
        }] = json!(text);
    }
    if let Some(thinking) = input["reasoning_content"].as_str() {
        target["thinking"] = json!(thinking);
    }
    if let Some(tools) = input.get("tool_calls") {
        target["tool_calls"] = calls(tools)?;
    }
    Ok(())
}
fn reason(value: &str) -> &str {
    if value == "tool_calls" { "stop" } else { value }
}
fn upstream_error(value: &Value) -> String {
    value
        .pointer("/error/message")
        .or_else(|| value.get("error"))
        .and_then(Value::as_str)
        .unwrap_or("Upstream inference failed")
        .into()
}
pub(super) async fn adapt(mut upstream: Response, kind: Kind, started: Instant) -> Response {
    if upstream.extensions().get::<NativeResponse>().is_some() {
        return upstream;
    }
    let context = upstream.extensions_mut().remove::<Context>();
    let status = upstream.status();
    let headers = upstream.headers().clone();
    if !status.is_success() {
        let body = axum::body::to_bytes(upstream.into_body(), LIMIT).await;
        let message = body
            .ok()
            .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
            .map(|v| upstream_error(&v))
            .unwrap_or_else(|| format!("Upstream HTTP {status}"));
        return error(status, message);
    }
    let model = context.map(|c| c.model).unwrap_or_default();
    let streaming = headers
        .get("content-type")
        .and_then(|h| h.to_str().ok())
        .is_some_and(|s| s.starts_with("text/event-stream"));
    let mut response = if streaming {
        let state = StreamState {
            source: Box::pin(upstream.into_body().into_data_stream()),
            buffer: Vec::new(),
            data: String::new(),
            queue: VecDeque::new(),
            finished: false,
            kind,
            model,
            started,
            usage: Value::Null,
            finish: None,
            tools: BTreeMap::new(),
            tool_bytes: 0,
        };
        let stream = futures_util::stream::unfold(state, |mut state| async move {
            loop {
                if let Some(value) = state.queue.pop_front() {
                    return Some((Ok::<_, std::convert::Infallible>(encoded(&value)), state));
                }
                if state.finished {
                    return None;
                }
                match state.source.next().await {
                    Some(Ok(bytes)) => {
                        if let Err(e) = state.push(&bytes) {
                            state.fail(e);
                        }
                    }
                    Some(Err(e)) => state.fail(e.to_string()),
                    None => state.fail("Upstream stream ended before [DONE]".into()),
                }
            }
        });
        (
            [
                ("content-type", "application/x-ndjson"),
                ("cache-control", "no-cache"),
            ],
            Body::from_stream(stream),
        )
            .into_response()
    } else {
        let parsed = axum::body::to_bytes(upstream.into_body(), 32 << 20)
            .await
            .map_err(|e| e.to_string())
            .and_then(|b| serde_json::from_slice::<Value>(&b).map_err(|e| e.to_string()));
        match parsed.and_then(|v| {
            if v.get("error").is_some() {
                return Err(upstream_error(&v));
            }
            let choice = v
                .pointer("/choices/0")
                .ok_or("Missing upstream completion")?;
            let finish = choice["finish_reason"]
                .as_str()
                .ok_or("Missing upstream finish reason")?;
            let mut output = base(kind, &model, true);
            output["done_reason"] = json!(reason(finish));
            message(&mut output, &choice["message"], kind)?;
            metrics(&mut output, &v["usage"], started);
            if let Some(logprobs) = choice.pointer("/logprobs/content") {
                output["logprobs"] = logprobs.clone();
            }
            Ok(output)
        }) {
            Ok(output) => Json(output).into_response(),
            Err(e) => error(StatusCode::BAD_GATEWAY, e),
        }
    };
    if let Some(id) = headers.get("x-request-id") {
        response.headers_mut().insert("x-request-id", id.clone());
    }
    response
}

type Source = Pin<Box<dyn Stream<Item = std::result::Result<Bytes, axum::Error>> + Send>>;
struct StreamState {
    source: Source,
    buffer: Vec<u8>,
    data: String,
    queue: VecDeque<Value>,
    finished: bool,
    kind: Kind,
    model: String,
    started: Instant,
    usage: Value,
    finish: Option<String>,
    tools: BTreeMap<usize, Value>,
    tool_bytes: usize,
}
impl StreamState {
    fn fail(&mut self, error: String) {
        self.queue.push_back(json!({"error":error}));
        self.finished = true;
    }
    fn push(&mut self, bytes: &[u8]) -> Result<()> {
        // Parse complete lines, allowing arbitrary UTF-8 and CRLF chunk splits.
        for part in bytes.split_inclusive(|b| *b == b'\n') {
            if self.finished {
                break;
            }
            if self.buffer.len() + part.len() + self.data.len() > LIMIT {
                return Err("Upstream SSE event exceeds 1 MiB".into());
            }
            self.buffer.extend_from_slice(part);
            if !part.ends_with(b"\n") {
                continue;
            }
            let line = std::str::from_utf8(&self.buffer)
                .map_err(|e| e.to_string())?
                .trim_end_matches(['\r', '\n'])
                .to_owned();
            self.buffer.clear();
            if line.is_empty() {
                if !self.data.is_empty() {
                    let data = std::mem::take(&mut self.data);
                    self.event(data.trim_end_matches('\n'))?;
                }
            } else if let Some(data) = line.strip_prefix("data:") {
                self.data.push_str(data.strip_prefix(' ').unwrap_or(data));
                self.data.push('\n');
            }
        }
        Ok(())
    }
    fn event(&mut self, data: &str) -> Result<()> {
        if data == "[DONE]" {
            let finish = self
                .finish
                .as_deref()
                .ok_or("Upstream stream omitted finish_reason")?;
            let mut output = base(self.kind, &self.model, true);
            output["done_reason"] = json!(reason(finish));
            if !self.tools.is_empty() {
                let tools: Vec<_> = self.tools.values().cloned().collect();
                let mut delta = base(self.kind, &self.model, false);
                delta["message"]["tool_calls"] = calls(&json!(tools))?;
                self.queue.push_back(delta);
            }
            metrics(&mut output, &self.usage, self.started);
            self.queue.push_back(output);
            self.finished = true;
            return Ok(());
        }
        let value: Value = serde_json::from_str(data).map_err(|e| e.to_string())?;
        if value.get("error").is_some() {
            return Err(upstream_error(&value));
        }
        if let Some(usage) = value.get("usage").filter(|v| !v.is_null()) {
            self.usage = usage.clone();
        }
        let Some(choice) = value.pointer("/choices/0") else {
            return Ok(());
        };
        if let Some(finish) = choice["finish_reason"].as_str() {
            self.finish = Some(finish.into());
        }
        let delta = &choice["delta"];
        let mut output = base(self.kind, &self.model, false);
        if let Some(tools) = delta["tool_calls"].as_array() {
            for tool in tools {
                let index = tool["index"]
                    .as_u64()
                    .ok_or("Missing streamed tool index")? as usize;
                self.tool_bytes = self.tool_bytes.saturating_add(tool.to_string().len());
                if self.tool_bytes > LIMIT {
                    return Err("Streamed tool calls exceed 1 MiB".into());
                }
                let entry = self
                    .tools
                    .entry(index)
                    .or_insert_with(|| json!({"id":"","function":{"name":"","arguments":""}}));
                for (path, field) in [
                    ("/id", "id"),
                    ("/function/name", "name"),
                    ("/function/arguments", "arguments"),
                ] {
                    if let Some(text) = tool.pointer(path).and_then(Value::as_str) {
                        let target = if field == "id" {
                            &mut entry[field]
                        } else {
                            &mut entry["function"][field]
                        };
                        if let Value::String(value) = target {
                            value.push_str(text);
                        }
                    }
                }
            }
        }
        message(
            &mut output,
            &json!({"content":delta["content"],"reasoning_content":delta["reasoning_content"]}),
            self.kind,
        )?;
        if let Some(logprobs) = choice.pointer("/logprobs/content") {
            output["logprobs"] = logprobs.clone();
        }
        if delta["content"].as_str().is_some_and(|s| !s.is_empty())
            || delta["reasoning_content"]
                .as_str()
                .is_some_and(|s| !s.is_empty())
            || output.get("logprobs").is_some()
        {
            self.queue.push_back(output);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn state() -> StreamState {
        StreamState {
            source: Box::pin(futures_util::stream::empty()),
            buffer: Vec::new(),
            data: String::new(),
            queue: VecDeque::new(),
            finished: false,
            kind: Kind::Chat,
            model: "alias".into(),
            started: Instant::now(),
            usage: Value::Null,
            finish: None,
            tools: BTreeMap::new(),
            tool_bytes: 0,
        }
    }
    #[test]
    fn every_byte_split_preserves_unicode_crlf_reasoning_tools_and_counts() {
        let events = [
            json!({"choices":[{"delta":{"reasoning_content":"推理"}}]}),
            json!({"choices":[{"delta":{"content":"你好"}}]}),
            json!({"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"lookup","arguments":"{\"key\":"}}]}}]}),
            json!({"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"你好\"}"}}]},"finish_reason":"tool_calls"}]}),
            json!({"choices":[],"usage":{"prompt_tokens":8,"completion_tokens":3}}),
        ];
        let mut data = String::from(": heartbeat\r\n\r\n");
        for event in events {
            data.push_str(&format!("data: {event}\r\n\r\n"));
        }
        data.push_str("data: [DONE]\r\n\r\n");
        for split in 0..=data.len() {
            let mut state = state();
            state.push(&data.as_bytes()[..split]).unwrap();
            state.push(&data.as_bytes()[split..]).unwrap();
            assert!(state.finished);
            let rows = state.queue;
            assert_eq!(rows[0]["message"]["thinking"], "推理");
            assert_eq!(rows[1]["message"]["content"], "你好");
            let done = rows.back().unwrap();
            assert_eq!(
                rows[2]["message"]["tool_calls"][0]["function"]["arguments"],
                json!({"key":"你好"})
            );
            assert_eq!(rows[2]["message"]["tool_calls"][0]["id"], "call_1");
            assert_eq!(done["done_reason"], "stop");
            assert_eq!(done["eval_count"], 3);
        }
    }
    #[test]
    fn invalid_or_oversized_tool_calls_and_unfinished_streams_cannot_succeed() {
        let mut state = state();
        assert!(state.push(b"data: [DONE]\n\n").is_err());
        assert!(state.push(&vec![b'a'; LIMIT + 1]).is_err());
        let mut state = super::tests::state();
        state.event(&json!({"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"f","arguments":"{"}}]},"finish_reason":"length"}]}).to_string()).unwrap();
        assert!(state.event("[DONE]").is_err());
        assert!(!state.finished);
    }
}
