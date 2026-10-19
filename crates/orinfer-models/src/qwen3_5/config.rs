//! Qwen configuration semantics and compiled arithmetic signature.
//! Weight files contain data; operator packages contain bindings. Control flow
//! belongs to the registered architecture implementation.
use crate::artifact::Result;
use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Deserialize)]
pub(crate) struct TextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub layer_types: Vec<String>,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "query_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "kv_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "head_dim")]
    pub head_dim: usize,
    #[serde(default)]
    pub num_experts: usize,
    #[serde(default)]
    pub num_experts_per_tok: usize,
    #[serde(default)]
    pub moe_intermediate_size: usize,
    #[serde(default)]
    pub shared_expert_intermediate_size: usize,
    #[serde(default)]
    pub mlp_only_layers: Vec<usize>,
}

fn query_heads() -> usize {
    24
}
fn kv_heads() -> usize {
    4
}
fn head_dim() -> usize {
    256
}

impl TextConfig {
    pub fn qkv(&self) -> usize {
        (2 * self.linear_num_key_heads + self.linear_num_value_heads) * self.linear_key_head_dim
    }
    pub fn mixer(&self) -> usize {
        self.linear_num_value_heads * self.linear_value_head_dim
    }
    pub fn full_q(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }
    pub fn kv(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }
    pub fn full_x(&self) -> usize {
        2 * (self.full_q() + self.kv())
    }
}

fn parse_text(text_value: &Value) -> Result<TextConfig> {
    let text: TextConfig =
        serde_json::from_value(text_value.clone()).map_err(|e| format!("Text config: {e}"))?;
    if text.hidden_size == 0
        || text.vocab_size == 0
        || text.num_hidden_layers == 0
        || text.layer_types.len() != text.num_hidden_layers
        || text
            .layer_types
            .iter()
            .any(|s| !matches!(s.as_str(), "linear_attention" | "full_attention"))
        || text.linear_num_key_heads == 0
        || text.linear_num_value_heads == 0
        || text.linear_key_head_dim == 0
        || text.linear_value_head_dim == 0
        || text.linear_conv_kernel_dim < 2
    {
        return Err("Invalid Qwen3_5 layer/state configuration".into());
    }
    Ok(text)
}

pub(crate) struct Configuration {
    text: TextConfig,
    pub vision_depth: usize,
    pub signature: Value,
}

impl Configuration {
    pub(super) fn text(&self) -> &TextConfig {
        &self.text
    }

    pub fn parse(config: Value) -> Result<Self> {
        if !matches!(
            config["model_type"].as_str(),
            Some("qwen3_5" | "qwen3_5_text" | "qwen3_5_moe")
        ) {
            return Err("Unsupported Qwen model configuration".into());
        }
        let text_value = config.get("text_config").unwrap_or(&config);
        let text = parse_text(text_value)?;
        if (config["model_type"] == "qwen3_5_moe") != (text.num_experts > 0) {
            return Err("Qwen dense/MoE configuration mismatch".into());
        }
        // These values affect compiled arithmetic, weight contracts or state.
        // Checkpoint identity, naming, dtype labels and quantization provenance
        // deliberately do not determine architecture/operator compatibility.
        let contract: Value = serde_json::from_str(include_str!(
            "../../../../configs/architecture-contract.json"
        ))
        .map_err(|e| e.to_string())?;
        let text_keys = contract["families"]["qwen3_5"]["text_keys"]
            .as_array()
            .ok_or("Missing architecture signature contract")?;
        let text_signature: serde_json::Map<String, Value> = text_keys
            .iter()
            .filter_map(|k| k.as_str())
            .filter_map(|k| text_value.get(k).map(|v| (k.into(), v.clone())))
            .collect();
        let vision_depth = config["vision_config"]["depth"]
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .unwrap_or(0);
        let signature = serde_json::json!({"text": text_signature,
            "vision": config.get("vision_config").cloned().unwrap_or(Value::Null)});
        Ok(Self {
            text,
            vision_depth,
            signature,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn config() -> Value {
        serde_json::json!({"model_type":"qwen3_5_text","hidden_size":8,"intermediate_size":16,"vocab_size":16,
            "num_hidden_layers":2,"layer_types":["linear_attention","full_attention"],
            "linear_num_key_heads":1,"linear_num_value_heads":2,
            "linear_key_head_dim":2,"linear_value_head_dim":3,"linear_conv_kernel_dim":4})
    }
    #[test]
    fn configuration_rejects_unknown_or_incomplete_architectures() {
        let mut raw = config();
        raw["model_type"] = "other".into();
        assert!(Configuration::parse(raw).is_err());
        let mut raw = config();
        raw["layer_types"] = serde_json::json!(["linear_attention"]);
        assert!(Configuration::parse(raw).is_err());
        let mut raw = config();
        raw["linear_num_value_heads"] = 0.into();
        assert!(Configuration::parse(raw).is_err());
    }
    #[test]
    fn kernel_compatibility_tracks_math_not_checkpoint_provenance() {
        let original = Configuration::parse(config()).unwrap();
        let mut raw = config();
        raw["quantization_config"] = serde_json::json!({"source":"another checkpoint"});
        raw["dtype"] = "float16".into();
        assert_eq!(
            original.signature,
            Configuration::parse(raw).unwrap().signature
        );
        let mut raw = config();
        raw["rms_norm_eps"] = serde_json::json!(0.1);
        assert_ne!(
            original.signature,
            Configuration::parse(raw).unwrap().signature
        );
    }
}
