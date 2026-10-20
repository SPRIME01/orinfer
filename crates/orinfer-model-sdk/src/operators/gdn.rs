//! Explicit state and prefill contracts selected by offline GDN construction.
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Execution {
    pub prefill: Prefill,
    pub decode: Decode,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Prefill {
    Sequential,
    Chunked,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Decode {
    Dense,
    Deferred { capacity: usize },
}

impl Execution {
    pub fn deferred_capacity(&self) -> Option<usize> {
        match self.decode {
            Decode::Dense => None,
            Decode::Deferred { capacity } => Some(capacity),
        }
    }
}
