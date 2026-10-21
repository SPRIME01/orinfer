//! Local management uses a private Unix socket; inference keeps Chat/SSE semantics.
use super::{Result, runtime_dir};
use serde_json::{Value, json};
use std::{
    io::{BufRead, Write},
    path::Path,
};

fn client(data: &Path) -> Result<reqwest::blocking::Client> {
    let data = data
        .canonicalize()
        .map_err(|_| "Model service is not running; start orinfer serve first".to_string())?;
    reqwest::blocking::Client::builder()
        .no_proxy()
        .unix_socket(runtime_dir(&data).join("control.sock"))
        .timeout(None)
        .build()
        .map_err(|e| e.to_string())
}
pub fn request(data: &Path, method: &str, route: &str, body: Option<&Value>) -> Result<Value> {
    let client = client(data)?;
    let method = method
        .parse::<reqwest::Method>()
        .map_err(|e| e.to_string())?;
    let mut request = client.request(method, format!("http://localhost{route}"));
    if let Some(body) = body {
        request = request.json(body);
    }
    let response = request.send().map_err(|e| format!("Model service: {e}"))?;
    let status = response.status();
    let value: Value = response.json().map_err(|e| e.to_string())?;
    if !status.is_success() {
        return Err(value["error"]["message"]
            .as_str()
            .unwrap_or("Management request failed")
            .to_owned());
    }
    Ok(value)
}
pub fn show(data: &Path, name: &str) -> Result<Value> {
    let response = client(data)?
        .get("http://localhost/show")
        .query(&[("name", name)])
        .send()
        .map_err(|e| e.to_string())?;
    let status = response.status();
    let value: Value = response.json().map_err(|e| e.to_string())?;
    if !status.is_success() {
        return Err(value["error"]["message"]
            .as_str()
            .unwrap_or("Unknown model")
            .to_owned());
    }
    Ok(value)
}
pub fn chat(data: &Path, mut body: Value) -> Result<Value> {
    body["stream"] = json!(true);
    body["stream_options"] = json!({"include_usage":true});
    let mut request = client(data)?
        .post("http://localhost/v1/chat/completions")
        .json(&body);
    if let Ok(key) = std::env::var("ORINFER_API_KEY") {
        request = request.bearer_auth(key);
    }
    let response = request.send().map_err(|e| e.to_string())?;
    if !response.status().is_success() {
        return Err(response.text().map_err(|e| e.to_string())?);
    }
    read_chat(
        std::io::BufReader::new(response),
        &mut std::io::stdout().lock(),
    )
}
fn read_chat(reader: impl BufRead, output: &mut impl Write) -> Result<Value> {
    let mut data = String::new();
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut finish = false;
    for line in reader.lines() {
        let line = line.map_err(|e| e.to_string())?;
        if let Some(value) = line.strip_prefix("data:") {
            if !data.is_empty() {
                data.push('\n');
            }
            data.push_str(value.strip_prefix(' ').unwrap_or(value));
        } else if line.is_empty() && !data.is_empty() {
            if data == "[DONE]" {
                if !finish {
                    return Err("Stream ended without a finish reason".into());
                }
                output.write_all(b"\n").map_err(|e| e.to_string())?;
                return Ok(
                    json!({"role":"assistant","content":content,"reasoning_content":reasoning}),
                );
            }
            let value: Value = serde_json::from_str(&data).map_err(|e| e.to_string())?;
            data.clear();
            if let Some(error) = value.get("error") {
                return Err(error.to_string());
            }
            for choice in value["choices"].as_array().into_iter().flatten() {
                if choice["finish_reason"].as_str().is_some() {
                    finish = true;
                }
                if let Some(text) = choice["delta"]["reasoning_content"].as_str() {
                    reasoning.push_str(text);
                }
                if let Some(text) = choice["delta"]["content"].as_str() {
                    content.push_str(text);
                    output
                        .write_all(text.as_bytes())
                        .map_err(|e| e.to_string())?;
                    output.flush().map_err(|e| e.to_string())?;
                }
            }
        }
    }
    Err("Incomplete Chat response".into())
}

/// Submit a local audio file through the same lazy ASR worker as HTTP clients.
pub fn transcribe(
    data: &Path,
    name: &str,
    path: &Path,
    language: Option<&str>,
    prompt: Option<&str>,
) -> Result<Value> {
    let metadata = std::fs::metadata(path).map_err(|e| e.to_string())?;
    if metadata.len() == 0 || metadata.len() > 24 * 1024 * 1024 {
        return Err("Audio file must contain 1..24 MiB".into());
    }
    let filename = path
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or("Invalid audio filename")?
        .to_owned();
    let part =
        reqwest::blocking::multipart::Part::bytes(std::fs::read(path).map_err(|e| e.to_string())?)
            .file_name(filename);
    let mut form = reqwest::blocking::multipart::Form::new()
        .text("model", name.to_owned())
        .part("file", part);
    if let Some(language) = language {
        form = form.text("language", language.to_owned());
    }
    if let Some(prompt) = prompt {
        form = form.text("prompt", prompt.to_owned());
    }
    let mut request = client(data)?
        .post("http://localhost/v1/audio/transcriptions")
        .multipart(form);
    if let Ok(key) = std::env::var("ORINFER_API_KEY") {
        request = request.bearer_auth(key);
    }
    let response = request.send().map_err(|e| e.to_string())?;
    let status = response.status();
    let value: Value = response.json().map_err(|e| e.to_string())?;
    if !status.is_success() {
        return Err(value["error"]["message"]
            .as_str()
            .unwrap_or("Transcription failed")
            .to_owned());
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sse_requires_finish_and_preserves_unicode() {
        let good = "data: {\"choices\":[{\"delta\":{\"content\":\"你好\"}}]}\r\n\r\ndata: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
        let mut output = Vec::new();
        assert_eq!(
            read_chat(good.as_bytes(), &mut output).unwrap()["content"],
            "你好"
        );
        assert_eq!(String::from_utf8(output).unwrap(), "你好\n");
        assert!(read_chat("data: [DONE]\n\n".as_bytes(), &mut Vec::new()).is_err());
        assert!(read_chat("data: {}\n\n".as_bytes(), &mut Vec::new()).is_err());
    }
}
