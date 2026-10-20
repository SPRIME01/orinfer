use super::{Kind, ModelRecord, Result};
use base64::{Engine, prelude::BASE64_STANDARD};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use std::{collections::VecDeque, time::Duration};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    model: String,
    #[serde(default)]
    messages: Vec<Message>,
    #[serde(default)]
    prompt: String,
    #[serde(default)]
    system: String,
    #[serde(default)]
    images: Vec<String>,
    stream: Option<bool>,
    keep_alive: Option<Value>,
    think: Option<Value>,
    format: Option<Value>,
    tools: Option<Vec<Value>>,
    options: Option<Map<String, Value>>,
    logprobs: Option<bool>,
    top_logprobs: Option<usize>,
    #[serde(default)]
    raw: bool,
    #[serde(default)]
    suffix: String,
    #[serde(default)]
    template: String,
    #[serde(default)]
    context: Vec<u32>,
    truncate: Option<bool>,
    shift: Option<bool>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Message {
    role: String,
    #[serde(default)]
    content: String,
    #[serde(default)]
    thinking: String,
    #[serde(default)]
    images: Vec<String>,
    #[serde(default)]
    tool_calls: Vec<Value>,
    tool_name: Option<String>,
    tool_call_id: Option<String>,
}

pub(in crate::manager) struct Native {
    pub chat: Value,
    pub model: String,
    pub stream: bool,
    pub empty: bool,
    pub keep_alive: Option<u64>,
    num_ctx: Option<u64>,
}
impl Native {
    pub fn validate(&self, record: &ModelRecord) -> Result<()> {
        if let Some(context) = self.num_ctx
            && context != record.max_context as u64
        {
            return Err(format!(
                "options.num_ctx: this model uses a fixed context limit of {}; omit num_ctx or use this value",
                record.max_context
            ));
        }
        Ok(())
    }
}

pub(super) fn keep_alive(value: &Value) -> Result<u64> {
    let duration = match value {
        Value::Number(number) => {
            let seconds = number.as_f64().ok_or("Invalid keep_alive")?;
            if seconds < 0.0 {
                return Ok(u64::MAX);
            }
            Duration::try_from_secs_f64(seconds).map_err(|_| "Invalid keep_alive")?
        }
        Value::String(text) => {
            let negative = text.starts_with('-');
            let text = text.strip_prefix('-').unwrap_or(text);
            let duration = if text == "0" {
                Duration::ZERO
            } else {
                humantime::parse_duration(text).map_err(|e| format!("keep_alive: {e}"))?
            };
            if negative && !duration.is_zero() {
                return Ok(u64::MAX);
            }
            duration
        }
        _ => return Err("keep_alive must be a duration string or seconds".into()),
    };
    duration
        .as_secs()
        .checked_add(u64::from(duration.subsec_nanos() != 0))
        .ok_or("keep_alive overflow".into())
}

pub(in crate::manager) fn translate(value: &Value, kind: Kind) -> Result<Native> {
    let input: Input = serde_json::from_value(value.clone()).map_err(|e| e.to_string())?;
    if input.model.is_empty() {
        return Err("model is required".into());
    }
    if input.raw
        || !input.suffix.is_empty()
        || !input.template.is_empty()
        || !input.context.is_empty()
    {
        return Err("raw, suffix, template overrides and legacy token context are not supported; use templated generate or chat messages".into());
    }
    if input.shift == Some(true) || input.truncate == Some(true) {
        return Err("Automatic context truncation/shift is not supported; requests exceeding the model context return an error".into());
    }
    let mut messages = Vec::new();
    match kind {
        Kind::Chat => {
            if !input.prompt.is_empty() || !input.system.is_empty() || !input.images.is_empty() {
                return Err("Use messages for /api/chat".into());
            }
            let mut pending = VecDeque::<(String, String)>::new();
            for (index, message) in input.messages.into_iter().enumerate() {
                let role = message.role.to_lowercase();
                let mut converted =
                    json!({"role":role,"content":content(message.content, message.images)?});
                if !message.thinking.is_empty() {
                    converted["reasoning_content"] = json!(message.thinking);
                }
                if !message.tool_calls.is_empty() {
                    if role != "assistant" {
                        return Err("tool_calls require an assistant message".into());
                    }
                    let mut calls = Vec::new();
                    for (i, call) in message.tool_calls.into_iter().enumerate() {
                        let name = call["function"]["name"]
                            .as_str()
                            .filter(|s| !s.is_empty())
                            .ok_or("Tool function name is required")?;
                        let args = &call["function"]["arguments"];
                        if !args.is_object() {
                            return Err("Tool arguments must be a JSON object".into());
                        }
                        let id = call["id"]
                            .as_str()
                            .filter(|s| !s.is_empty())
                            .map(str::to_owned)
                            .unwrap_or_else(|| format!("call_ollama_{index}_{i}"));
                        pending.push_back((id.clone(), name.into()));
                        calls.push(json!({"id":id,"type":"function","function":{"name":name,"arguments":args.to_string()}}));
                    }
                    converted["tool_calls"] = json!(calls);
                }
                if role == "tool" {
                    let position = pending
                        .iter()
                        .position(|(id, name)| {
                            message.tool_call_id.as_ref().is_none_or(|s| s == id)
                                && message.tool_name.as_ref().is_none_or(|s| s == name)
                        })
                        .ok_or("Tool result does not match a pending tool call")?;
                    converted["tool_call_id"] = json!(pending.remove(position).unwrap().0);
                }
                messages.push(converted);
            }
        }
        Kind::Generate => {
            if !input.messages.is_empty() || input.tools.is_some() {
                return Err("messages/tools require /api/chat".into());
            }
            if !input.system.is_empty() {
                messages.push(json!({"role":"system","content":input.system}));
            }
            if !input.prompt.is_empty() || !input.images.is_empty() {
                messages
                    .push(json!({"role":"user","content":content(input.prompt, input.images)?}));
            }
        }
    }
    let empty = messages.is_empty();
    let stream = input.stream.unwrap_or(true);
    let mut chat = json!({"model":input.model,"messages":messages,"stream":stream});
    if stream {
        chat["stream_options"] = json!({"include_usage":true});
    }
    if let Some(think) = input.think {
        match think {
            Value::Bool(enabled) => chat["enable_thinking"] = json!(enabled),
            Value::String(effort)
                if ["low", "medium", "high", "xhigh", "max", "minimal", "none"]
                    .contains(&effort.as_str()) =>
            {
                chat["reasoning_effort"] = json!(effort)
            }
            _ => return Err("think must be a boolean or supported reasoning level".into()),
        }
    }
    if let Some(format) = input.format {
        chat["response_format"] = match format {
            Value::String(s) if s == "json" => json!({"type":"json_object"}),
            Value::String(s) if s.is_empty() => Value::Null,
            Value::Object(schema) => {
                json!({"type":"json_schema","json_schema":{"name":"ollama","strict":true,"schema":schema}})
            }
            _ => return Err("format must be json or a JSON schema object".into()),
        };
    }
    if let Some(tools) = input.tools {
        chat["tools"] = json!(tools);
    }
    if let Some(logprobs) = input.logprobs {
        chat["logprobs"] = json!(logprobs);
    }
    if let Some(top) = input.top_logprobs {
        chat["top_logprobs"] = json!(top);
    }
    let mut num_ctx = None;
    for (key, value) in input.options.unwrap_or_default() {
        let target = match key.as_str() {
            "temperature" | "top_p" | "top_k" | "seed" | "stop" | "presence_penalty"
            | "frequency_penalty" => key.as_str(),
            "repeat_penalty" => "repetition_penalty",
            "num_predict" => {
                if value.as_u64().is_none_or(|v| v == 0) {
                    return Err("options.num_predict must be positive".into());
                }
                "max_completion_tokens"
            }
            "num_ctx" => {
                num_ctx = Some(
                    value
                        .as_u64()
                        .filter(|v| *v > 0)
                        .ok_or("options.num_ctx must be positive")?,
                );
                continue;
            }
            _ => return Err(format!("Unsupported Ollama option: {key}")),
        };
        chat[target] = value;
    }
    Ok(Native {
        chat,
        model: input.model,
        stream,
        empty,
        keep_alive: input.keep_alive.as_ref().map(keep_alive).transpose()?,
        num_ctx,
    })
}

fn content(text: String, images: Vec<String>) -> Result<Value> {
    if images.is_empty() {
        return Ok(json!(text));
    }
    let mut content = vec![json!({"type":"text","text":text})];
    for encoded in images {
        let bytes = BASE64_STANDARD
            .decode(&encoded)
            .map_err(|e| format!("Invalid base64 image: {e}"))?;
        let mime = match image::guess_format(&bytes).map_err(|e| e.to_string())? {
            image::ImageFormat::Png => "image/png",
            image::ImageFormat::Jpeg => "image/jpeg",
            image::ImageFormat::WebP => "image/webp",
            _ => return Err("Images must be PNG, JPEG or WebP".into()),
        };
        content.push(
            json!({"type":"image_url","image_url":{"url":format!("data:{mime};base64,{encoded}")}}),
        );
    }
    Ok(json!(content))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn duration_forms_and_invalid_inputs() {
        for (value, expected) in [
            (json!(0), 0),
            (json!(1.2), 2),
            (json!("500ms"), 1),
            (json!("1h30m"), 5400),
            (json!(-1), u64::MAX),
            (json!("-1s"), u64::MAX),
            (json!("0"), 0),
        ] {
            assert_eq!(keep_alive(&value).unwrap(), expected);
        }
        for bad in [
            json!({}),
            json!(true),
            json!("broken"),
            json!("-bad"),
            json!(1e100),
        ] {
            assert!(keep_alive(&bad).is_err());
        }
    }
    #[test]
    fn tools_history_schema_sampling_and_images_translate_without_losing_identity() {
        let png = "iVBORw0KGgo=";
        let request=translate(&json!({"model":"x","stream":false,"think":"medium",
            "messages":[{"role":"user","content":"看图","images":[png]},
                {"role":"assistant","thinking":"查一下","tool_calls":[{"function":{"name":"lookup","arguments":{"key":"你好"}}}]},
                {"role":"tool","tool_name":"lookup","content":"找到"}],
            "format":{"type":"object"},"tools":[],"options":{"repeat_penalty":1.1,"stop":["END"],"num_predict":32}}),Kind::Chat).unwrap().chat;
        assert_eq!(request["reasoning_effort"], "medium");
        assert_eq!(
            request["messages"][0]["content"][1]["image_url"]["url"],
            format!("data:image/png;base64,{png}")
        );
        assert_eq!(request["messages"][1]["reasoning_content"], "查一下");
        assert_eq!(
            request["messages"][1]["tool_calls"][0]["id"],
            request["messages"][2]["tool_call_id"]
        );
        let args: Value = serde_json::from_str(
            request["messages"][1]["tool_calls"][0]["function"]["arguments"]
                .as_str()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(args, json!({"key":"你好"}));
        assert_eq!(
            request["response_format"]["json_schema"]["schema"],
            json!({"type":"object"})
        );
        assert_eq!(request["repetition_penalty"], 1.1);
        assert_eq!(request["max_completion_tokens"], 32);
    }
}
