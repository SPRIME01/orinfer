use super::protocol::Http;
use super::*;
use futures_util::StreamExt;

async fn post(http: &Http, path: &str, body: Value) -> reqwest::Response {
    http.client
        .post(format!("{}{path}", http.url))
        .json(&body)
        .send()
        .await
        .unwrap()
}
fn spec(fixture: &Fixture) {
    let usage = json!({"prompt_tokens":8,"completion_tokens":3,"prompt_tokens_details":{"cached_tokens":4}});
    fs::write(fixture.root.join("response.json"),json!({
        "body":{"choices":[{"message":{"role":"assistant","content":"你好","reasoning_content":"思考"},"finish_reason":"stop"}],"usage":usage},
        "events":[{"choices":[{"delta":{"reasoning_content":"思考"}}]},
            {"choices":[{"delta":{"content":"你好"}}]},
            {"choices":[{"delta":{},"finish_reason":"stop"}]},
            {"choices":[],"usage":usage}]
    }).to_string()).unwrap();
}

#[tokio::test]
async fn discovery_is_cpu_only_and_unknown_operations_use_native_errors() {
    let fixture = Fixture::new();
    fs::write(
        fixture.root.join("config.json"),
        json!({"text_config":{"hidden_size":256,"num_hidden_layers":4},"vision_config":{}})
            .to_string(),
    )
    .unwrap();
    fs::write(
        fixture.root.join("tokenizer_config.json"),
        json!({"chat_template":"native jinja"}).to_string(),
    )
    .unwrap();
    let http = Http::new(Arc::clone(&fixture.manager)).await;
    for route in ["/", "/api/version", "/api/tags", "/api/ps"] {
        let response = http
            .client
            .get(format!("{}{route}", http.url))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200, "{route}");
    }
    let tags: Value = http
        .client
        .get(format!("{}/api/tags", http.url))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(tags["models"][0]["name"], "fixture:2b");
    let shown: Value = post(&http, "/api/show", json!({"model":"fixture:2b"}))
        .await
        .json()
        .await
        .unwrap();
    assert_eq!(shown["model_info"]["test.context_length"], 8192);
    assert_eq!(shown["template"], "native jinja");
    assert!(
        shown["capabilities"]
            .as_array()
            .unwrap()
            .contains(&json!("vision"))
    );
    assert_eq!(fixture.starts(), 0);
    assert_eq!(
        post(&http, "/api/show", json!({"model":"absent"}))
            .await
            .status(),
        404
    );
    let response = post(&http, "/api/embed", json!({})).await;
    assert_eq!(response.status(), 501);
    assert!(response.json::<Value>().await.unwrap()["error"].is_string());
    let malformed = http
        .client
        .post(format!("{}/api/chat", http.url))
        .body("{")
        .send()
        .await
        .unwrap();
    assert_eq!(malformed.status(), 400);
    assert!(malformed.json::<Value>().await.unwrap()["error"].is_string());
}

#[tokio::test]
async fn native_chat_and_generate_preserve_defaults_thinking_usage_and_streaming() {
    let fixture = Fixture::new();
    spec(&fixture);
    fixture
        .alias("alias", |r| {
            r.options.default_request_params =
                json!({"temperature":0.7,"enable_thinking":true,"max_completion_tokens":8192})
                    .as_object()
                    .unwrap()
                    .clone();
        })
        .await;
    let http = Http::new(Arc::clone(&fixture.manager)).await;
    for path in ["/api/chat", "/api/generate"] {
        for stream in [false, true] {
            let mut request = json!({"model":"alias:latest","stream":stream,"think":false,"options":{"temperature":0,"num_predict":8},"keep_alive":"1h30m"});
            if path.ends_with("chat") {
                request["messages"] = json!([{"role":"user","content":"Hello"}]);
            } else {
                request["prompt"] = json!("Hello");
            }
            let response = post(&http, path, request).await;
            assert_eq!(response.status(), 200);
            assert!(
                response.headers()["content-type"]
                    .to_str()
                    .unwrap()
                    .starts_with(if stream {
                        "application/x-ndjson"
                    } else {
                        "application/json"
                    })
            );
            let body = response.text().await.unwrap();
            let rows: Vec<Value> = body
                .lines()
                .map(|line| serde_json::from_str(line).unwrap())
                .collect();
            let last = rows.last().unwrap();
            assert_eq!(last["model"], "alias:latest");
            assert_eq!(last["done"], true);
            assert_eq!(last["eval_count"], 3);
            assert_eq!(last["prompt_eval_count"], 8);
            assert_eq!(last["prompt_eval_cached_count"], 4);
            assert_eq!(last["done_reason"], "stop");
            assert!(humantime::parse_rfc3339(last["created_at"].as_str().unwrap()).is_ok());
            let field = if path.ends_with("chat") {
                "/message/content"
            } else {
                "/response"
            };
            let thinking = if path.ends_with("chat") {
                "/message/thinking"
            } else {
                "/thinking"
            };
            assert_eq!(
                rows.iter()
                    .filter_map(|v| v.pointer(field).and_then(Value::as_str))
                    .collect::<String>(),
                "你好"
            );
            assert_eq!(
                rows.iter()
                    .filter_map(|v| v.pointer(thinking).and_then(Value::as_str))
                    .collect::<String>(),
                "思考"
            );
        }
    }
    let requests = fs::read_to_string(fixture.root.join("requests.jsonl")).unwrap();
    for line in requests.lines() {
        let request: Value = serde_json::from_str(line).unwrap();
        assert_eq!(request["enable_thinking"], false);
        assert_eq!(request["temperature"], 0);
        assert_eq!(request["max_completion_tokens"], 8);
    }
    assert_eq!(
        fixture
            .manager
            .runners
            .lock()
            .await
            .values()
            .next()
            .unwrap()
            .keep_alive_s
            .load(Ordering::Relaxed),
        5400
    );
    fixture.unload().await;
}

#[tokio::test]
async fn stream_defaults_to_ndjson_and_disconnect_releases_shared_resources() {
    let fixture = Fixture::new();
    spec(&fixture);
    // Hold before the terminal finish event, so this is a genuinely active stream.
    fs::write(
        fixture.root.join("response.json"),
        json!({"events":[{"choices":[{"delta":{"content":"hello"}}]}]}).to_string(),
    )
    .unwrap();
    fs::write(fixture.root.join("hold_response"), "").unwrap();
    let http = Http::new(Arc::clone(&fixture.manager)).await;
    let response = post(
        &http,
        "/api/chat",
        json!({"model":"fixture:2b","messages":[{"role":"user","content":"hi"}],"keep_alive":0}),
    )
    .await;
    assert_eq!(response.headers()["content-type"], "application/x-ndjson");
    let mut stream = response.bytes_stream();
    assert!(stream.next().await.unwrap().unwrap().starts_with(b"{"));
    let runner = fixture
        .manager
        .runners
        .lock()
        .await
        .values()
        .next()
        .unwrap()
        .clone();
    assert_eq!(runner.active.load(Ordering::Acquire), 1);
    fixture.manager.sweep().await;
    assert!(!fixture.manager.runners.lock().await.is_empty());
    drop(stream);
    eventually(|| fixture.manager.requests.available_permits() == 128).await;
    assert_eq!(fixture.manager.body_memory.available_permits(), 512);
    eventually(|| fixture.root.join("disconnected").exists()).await;
    fixture.manager.sweep().await;
    assert!(fixture.manager.runners.lock().await.is_empty());
    fixture.assert_gpu_released().await;
}

#[tokio::test]
async fn preload_unload_and_explicit_unsupported_options_do_not_generate() {
    let fixture = Fixture::new();
    let http = Http::new(Arc::clone(&fixture.manager)).await;
    for (extra, reason) in [
        (json!({"keep_alive":-1}), "load"),
        (json!({"keep_alive":0}), "unload"),
    ] {
        let mut request = json!({"model":"fixture:2b","stream":false});
        request
            .as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        let response = post(&http, "/api/generate", request).await;
        assert_eq!(response.status(), 200);
        assert_eq!(
            response.json::<Value>().await.unwrap()["done_reason"],
            reason
        );
        if reason == "load" {
            let ps: Value = http
                .client
                .get(format!("{}/api/ps", http.url))
                .send()
                .await
                .unwrap()
                .json()
                .await
                .unwrap();
            assert_eq!(ps["models"][0]["expires_at"], "9999-12-31T23:59:59Z");
        }
    }
    assert!(!fixture.root.join("requests.jsonl").exists());
    assert!(fixture.manager.runners.lock().await.is_empty());
    for bad in [
        json!({"raw":true}),
        json!({"options":{"num_ctx":16}}),
        json!({"options":{"bogus":1}}),
        json!({"keep_alive":{}}),
    ] {
        let mut request = json!({"model":"fixture:2b","prompt":"hi"});
        request
            .as_object_mut()
            .unwrap()
            .extend(bad.as_object().unwrap().clone());
        let response = post(&http, "/api/generate", request).await;
        assert_eq!(response.status(), 400);
        assert!(response.json::<Value>().await.unwrap()["error"].is_string());
    }
    assert_eq!(fixture.starts(), 1);
}

#[tokio::test]
async fn stream_failures_are_terminal_errors_instead_of_successful_truncations() {
    let fixture = Fixture::new();
    let http = Http::new(Arc::clone(&fixture.manager)).await;
    for spec in [
        json!({"events":[],"abrupt_eof":true}),
        json!({"events":[{"error":{"message":"GPU failed"}}]}),
        json!({"events":[]}),
    ] {
        fs::write(fixture.root.join("response.json"), spec.to_string()).unwrap();
        let response = post(
            &http,
            "/api/generate",
            json!({"model":"fixture:2b","prompt":"hi"}),
        )
        .await;
        let body = response.text().await.unwrap();
        let last: Value = serde_json::from_str(body.lines().last().unwrap()).unwrap();
        assert!(last["error"].is_string());
        assert!(last.get("done").is_none());
    }
    fixture.unload().await;
}

#[tokio::test]
#[ignore = "Requires ORINFER_OLLAMA_SDK_PYTHON pointing to a uv environment with the official ollama client"]
async fn official_ollama_python_client() {
    let fixture = Fixture::new();
    spec(&fixture);
    fs::write(fixture.root.join("config.json"), "{}").unwrap();
    fs::write(fixture.root.join("tokenizer_config.json"), "{}").unwrap();
    let http = Http::new(Arc::clone(&fixture.manager)).await;
    let python = std::env::var("ORINFER_OLLAMA_SDK_PYTHON").unwrap();
    let script = r#"
import json, os, pathlib, sys, ollama
headers={}
if os.environ.get("ORINFER_API_KEY"): headers["Authorization"]="Bearer "+os.environ["ORINFER_API_KEY"]
c=ollama.Client(host=sys.argv[1],headers=headers)
assert c.list().models[0].model == 'fixture:2b'
assert c.show('fixture:2b').modelinfo['test.context_length'] == 8192
assert not c.ps().models
for stream in (False,True):
    r=c.chat('fixture:2b',messages=[{'role':'user','content':'hi'}],stream=stream,think=False,options={'num_predict':8})
    rows=list(r) if stream else [r]
    assert ''.join(x.message.content or '' for x in rows)=='你好'
    assert ''.join(x.message.thinking or '' for x in rows)=='思考'
    assert rows[-1].done and rows[-1].eval_count==3
    r=c.generate('fixture:2b',prompt='hi',stream=stream,think=False)
    rows=list(r) if stream else [r]
    assert ''.join(x.response for x in rows)=='你好'
assert len(c.ps().models)==1
spec=pathlib.Path(sys.argv[2])/'response.json'
call={'id':'call_1','type':'function','function':{'name':'lookup','arguments':'{"key":"hello"}'}}
usage={'prompt_tokens':8,'completion_tokens':3}
spec.write_text(json.dumps({'body':{'choices':[{'message':{'role':'assistant','content':None,'tool_calls':[call]},'finish_reason':'tool_calls'}],'usage':usage},
    'events':[{'choices':[{'delta':{'tool_calls':[{'index':0,**call}]}}]},
        {'choices':[{'delta':{},'finish_reason':'tool_calls'}]}, {'choices':[],'usage':usage}]}))
for stream in (False,True):
    r=c.chat('fixture:2b',messages=[{'role':'user','content':'lookup hello'}],stream=stream,
        tools=[{'type':'function','function':{'name':'lookup','parameters':{'type':'object','properties':{'key':{'type':'string'}}}}}])
    rows=list(r) if stream else [r]
    calls=[call for row in rows for call in (row.message.tool_calls or [])]
    assert len(calls)==1 and calls[0].function.arguments=={'key':'hello'}
    if stream: assert any(not row.done and row.message.tool_calls for row in rows)
    c.chat('fixture:2b',messages=[{'role':'user','content':'lookup hello'},
        {'role':'assistant','tool_calls':calls},{'role':'tool','tool_name':'lookup','content':'found'}])
assert c.generate('fixture:2b',keep_alive=0).done_reason=='unload'

assert not c.ps().models
c.copy('fixture:2b','alias:latest')
assert len(c.list().models)==2
c.delete('alias:latest')
assert len(c.list().models)==1
try:
    c.chat('missing',messages=[{'role':'user','content':'hi'}])
except ollama.ResponseError as e:
    assert e.status_code==404
else: raise AssertionError('missing model accepted')
print('Official Ollama client: discovery, chat/generate JSON+NDJSON, thinking, counts, errors, preload/unload passed')
"#;
    let status = tokio::process::Command::new(python)
        .args(["-c", script, &http.url, fixture.root.to_str().unwrap()])
        .status()
        .await
        .unwrap();
    fixture.unload().await;
    assert!(status.success());
}

#[tokio::test]
async fn native_alias_management_is_atomic_and_preserves_imported_weights() {
    let fixture = Fixture::new();
    let http = Http::new(Arc::clone(&fixture.manager)).await;
    let copied = post(
        &http,
        "/api/copy",
        json!({"source":"fixture:2b","destination":"alias"}),
    )
    .await;
    assert_eq!(copied.status(), 200);
    let records = registry::read(&fixture.root).unwrap();
    assert_eq!(records["alias"].identity, records["fixture:2b"].identity);
    assert_eq!(
        post(
            &http,
            "/api/copy",
            json!({"source":"fixture:2b","destination":"../bad"})
        )
        .await
        .status(),
        400
    );
    let response = http
        .client
        .delete(format!("{}/api/delete", http.url))
        .json(&json!({"model":"alias:latest"}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    assert!(fixture.root.join("cache/model.json").exists());
    assert_eq!(registry::read(&fixture.root).unwrap().len(), 1);
    assert_eq!(fixture.starts(), 0);
}
