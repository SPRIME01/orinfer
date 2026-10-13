use serde::{Deserialize, Serialize};
/// Architecture-owned views into row-major workspace or private request state.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct BufferView {
    pub buffer: String,
    pub offset: usize,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct Invocation {
    pub operation: crate::model::Operation,
    pub sequence: Option<usize>,
    pub launch: Option<crate::operators::dynamic::RowLaunch>,
    pub views: std::collections::BTreeMap<String, BufferView>,
}
