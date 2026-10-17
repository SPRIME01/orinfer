//! Freeze model-declared layer cuts without changing a forward pass.
use crate::{
    artifact::Result,
    model::{Manifest, Operation},
};

pub(crate) fn register(
    m: &mut Manifest,
    program: &str,
    stages: Vec<Vec<Operation>>,
    mut live: Vec<String>,
) -> Result<()> {
    if stages.is_empty()
        || stages.iter().any(Vec::is_empty)
        || stages.iter().flatten().ne(m
            .programs
            .get(program)
            .ok_or("Missing prefill body")?
            .iter())
    {
        return Err("Prefill stages must partition the complete forward pass".into());
    }
    live.sort();
    live.dedup();
    if live.is_empty()
        || live.iter().any(|name| {
            !m.buffers
                .iter()
                .any(|b| &b.name == name && b.data.is_none())
        })
    {
        return Err("Invalid prefill continuation buffers".into());
    }
    let plan = m
        .prefill_plans
        .iter_mut()
        .find(|p| p.prefill_program == program)
        .ok_or("Missing prefill plan")?;
    let capacity = plan.chunk_tokens;
    let kv = m.kv_cache.as_mut().ok_or("Missing prefill KV growth")?;
    let growth = kv
        .growth
        .get(program)
        .ok_or("Missing prefill growth contract")?
        .clone();
    plan.stages.clear();
    plan.live_buffers = live;
    for (index, operations) in stages.into_iter().enumerate() {
        let name = format!("{program}/stage{index}");
        if m.programs.contains_key(&name) {
            return Err("Duplicate prefill stage".into());
        }
        plan.stages.push(name.clone());
        kv.growth.insert(name.clone(), growth.clone());
        if let Some(rows) = &mut m.row_execution {
            rows.programs.insert(name.clone());
            rows.program_capacities.insert(name.clone(), capacity);
        }
        m.programs.insert(name, operations);
    }
    Ok(())
}
