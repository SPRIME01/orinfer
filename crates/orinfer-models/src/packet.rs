//! Common row invocation binding; model programs retain their arithmetic order.
use crate::{
    artifact::Result,
    execution::{BufferView, Invocation},
    model::{Manifest, Operation},
};
use orinfer_model_sdk::{abi::StateBindings, architecture::BatchSegment};
use std::collections::{BTreeMap, BTreeSet};

/// Contiguous live rows, independent of the model's computation or state layout.
pub(crate) struct Rows {
    pub starts: Vec<usize>,
    pub total: usize,
}

impl Rows {
    pub fn new(lengths: impl IntoIterator<Item = usize>, capacity: usize) -> Result<Self> {
        let mut starts = Vec::new();
        let mut total = 0usize;
        for length in lengths {
            if length == 0 {
                return Err("Empty packet segment".into());
            }
            starts.push(total);
            total = total.checked_add(length).ok_or("Packet row overflow")?;
            if total > capacity {
                return Err("Packet row capacity exceeded".into());
            }
        }
        if starts.is_empty() {
            return Err("Empty packet".into());
        }
        Ok(Self { starts, total })
    }

    pub fn packet(segments: &[BatchSegment], owners: usize, capacity: usize) -> Result<Self> {
        let mut slots = BTreeSet::new();
        if segments.len() > owners || segments.iter().any(|s| !slots.insert(s.slot)) {
            return Err("Packet requires distinct owners within capacity".into());
        }
        Self::new(segments.iter().map(|s| s.tokens), capacity)
    }
}

pub(crate) fn view(buffer: &str, row: usize, stride: usize) -> Result<BufferView> {
    Ok(BufferView {
        buffer: buffer.into(),
        offset: row
            .checked_mul(stride)
            .ok_or("Packet byte offset overflow")?,
    })
}

pub(crate) fn views<'a>(
    row: usize,
    strides: impl IntoIterator<Item = (&'a str, usize)>,
) -> Result<BTreeMap<String, BufferView>> {
    strides
        .into_iter()
        .map(|(name, stride)| Ok((name.into(), view(name, row, stride)?)))
        .collect()
}

/// Resolve each model's ordered role candidates; absent roles stay null.
pub(crate) fn layer_roles<S: AsRef<str>, I: IntoIterator<Item = String>>(
    m: &Manifest,
    layers: usize,
    columns: &[S],
    resolve: impl Fn(&str, usize) -> I,
) -> Vec<Option<String>> {
    let names: BTreeSet<_> = m.buffers.iter().map(|b| b.name.as_str()).collect();
    (0..layers)
        .flat_map(|layer| {
            let (names, resolve) = (&names, &resolve);
            columns.iter().map(move |column| {
                resolve(column.as_ref(), layer)
                    .into_iter()
                    .find(|name| names.contains(name.as_str()))
            })
        })
        .collect()
}

pub(crate) fn invocation(
    operation: Operation,
    sequence: Option<usize>,
    views: &BTreeMap<String, BufferView>,
) -> Invocation {
    Invocation {
        operation,
        sequence,
        views: views.clone(),
        launch: None,
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed_rows_and_heads_keep_independent_owner_order() {
        let segments = [
            BatchSegment { slot: 9, tokens: 1 },
            BatchSegment { slot: 2, tokens: 7 },
            BatchSegment { slot: 7, tokens: 3 },
        ];
        let rows = Rows::packet(&segments, 128, 1024).unwrap();
        let heads = Rows::new([1, 7, 1], 128).unwrap();
        assert_eq!(rows.starts, [0, 1, 8]);
        assert_eq!(rows.total, 11);
        assert_eq!(heads.starts, [0, 1, 8]);
        assert_eq!(heads.total, 9);
        let tail = views(rows.starts[2], [("Hidden", 5120), ("Scale", 2)]).unwrap();
        assert_eq!(tail["Hidden"].offset, 40960);
        assert_eq!(tail["Scale"].offset, 16);
        let roles = [Some("KV".to_owned()), None];
        let forward = private_bindings(&segments, 4, &roles);
        assert_eq!(forward[0], Some((9, "KV".into())));
        assert_eq!(forward[2], Some((2, "KV".into())));
        assert_eq!(&forward[6..], &[None, None]);
        let reversed = private_bindings(
            &segments[..2].iter().rev().cloned().collect::<Vec<_>>(),
            4,
            &roles,
        );
        assert_eq!(reversed[0], Some((2, "KV".into())));
        assert_eq!(&reversed[4..], &[None, None, None, None]);
    }

    #[test]
    fn packet_bounds_fail_before_creating_views() {
        assert!(Rows::new([], 128).is_err());
        assert!(Rows::new([1, 0], 128).is_err());
        assert!(Rows::new([usize::MAX, 1], usize::MAX).is_err());
        assert!(Rows::new([127, 2], 128).is_err());
        assert!(view("Hidden", usize::MAX, 2).is_err());
        let segments = [
            BatchSegment { slot: 3, tokens: 1 },
            BatchSegment { slot: 3, tokens: 1 },
        ];
        assert!(Rows::packet(&segments, 128, 128).is_err());
        assert!(Rows::packet(&segments[..1], 0, 128).is_err());
    }
}
