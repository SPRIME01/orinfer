//! Shared sampling program registration; independent of model architecture.
use crate::{
    artifact::Result,
    model::{Manifest, Operation, Sampling},
};
use std::collections::BTreeSet;

pub(crate) fn register(m: &mut Manifest, bindings: &BTreeSet<String>) -> Result<()> {
    m.sampling = None;
    let mut names: Vec<String> = ["initialize", "history", "scores", "masses"]
        .into_iter()
        .map(|s| format!("sampling/{s}"))
        .collect();
    for mode in 0..3 {
        for digit in 0..12 {
            names.extend([
                format!("sampling/{mode}/{digit}/hist"),
                format!("sampling/{mode}/{digit}/choose"),
            ]);
        }
    }
    let count = bindings
        .iter()
        .filter(|n| n.starts_with("sampling/"))
        .count();
    if count == 0 {
        return Ok(());
    }
    if count != names.len() || names.iter().any(|n| !bindings.contains(n)) {
        return Err("Incomplete shared GPU sampling bindings".into());
    }
    m.programs.insert(
        "sampling".into(),
        names
            .into_iter()
            .map(|name| Operation::Kernel { name })
            .collect(),
    );
    m.sampling = Some(Sampling {
        program: "sampling".into(),
        history: "SamplingHistory".into(),
        bias_ids: "SamplingBiasIds".into(),
        bias_values: "SamplingBiasValues".into(),
        mask: "SamplingMask".into(),
        control: "SamplingControl".into(),
        parameters: "SamplingParameters".into(),
        output: "SamplingOutput".into(),
    });
    Ok(())
}
