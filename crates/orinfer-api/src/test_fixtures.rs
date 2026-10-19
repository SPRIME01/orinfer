//! Selected Ollama cases, adapted under MIT; see fixtures/ollama/LICENSE and cases.json.
use serde_json::Value;
use std::sync::OnceLock;

pub(crate) fn ollama() -> &'static Value {
    static CASES: OnceLock<Value> = OnceLock::new();
    CASES.get_or_init(|| {
        serde_json::from_str(include_str!("../../../fixtures/ollama/cases.json")).unwrap()
    })
}
