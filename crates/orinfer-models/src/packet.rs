//! Common row invocation binding; model programs retain their arithmetic order.
use crate::{
    artifact::Result,
    execution::{BufferView, Invocation},
    model::Manifest,
};
use orinfer_model_sdk::{abi::StateBindings, architecture::BatchSegment};
use std::collections::BTreeMap;

pub(crate) fn private_bindings(
    segments: &[BatchSegment],
    capacity: usize,
    roles: &[Option<String>],
) -> StateBindings {
    (0..capacity)
        .flat_map(|lane| {
            roles.iter().map(move |role| {
                segments
                    .get(lane)
                    .and_then(|s| role.as_ref().map(|name| (s.slot, name.clone())))
            })
        })
        .collect()
}

pub(crate) fn append(
    m: &Manifest,
    out: &mut Vec<Invocation>,
    program: &str,
    count: usize,
    sequence: Option<usize>,
    views: &BTreeMap<String, BufferView>,
) -> Result<()> {
    let rows = m.row_execution.as_ref().ok_or("Missing row execution")?;
    let operations = m.programs.get(program).ok_or("Missing row program")?;
    let mut calls = rows.specialize(program, operations, &m.dynamic_batch_kernels, count)?;
    for call in &mut calls {
        call.sequence = sequence;
        call.views = views.clone();
    }
    out.extend(calls);
    Ok(())
}
