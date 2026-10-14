//! Local tiling choices share the canonical operand and row contracts.
use crate::artifact::{Access, Argument, Buffer, Dtype, Kernel, Result};
use orinfer_model_sdk::operators::rows::KernelVariant;
use std::collections::BTreeMap;

fn argument(k: &Kernel, index: usize) -> Result<&str> {
    match k.args.get(index) {
        Some(Argument::Buffer { name }) => Ok(name.as_str()),
        _ => Err("HC stream projection requires whole-buffer operands".into()),
    }
}

pub(super) fn build<Data>(
    kernels: &[Kernel],
    buffers: &[Buffer<Data>],
) -> Result<BTreeMap<String, Vec<KernelVariant>>> {
    let contracts: BTreeMap<_, _> = kernels.iter().map(|k| (k.name.as_str(), k)).collect();
    let mut variants: BTreeMap<String, Vec<KernelVariant>> = BTreeMap::new();
    for kernel in kernels {
        let Some((slot, suffix)) = kernel.name.rsplit_once('/') else {
            continue;
        };
        if suffix == "dispatch_planes" {
            let canonical = contracts.get(slot).ok_or("Missing expert dispatch slot")?;
            if !slot.starts_with("prefill_m4096/")
                || canonical.symbol != "main_kernel"
                || canonical.args.len() != 9
                || kernel.args.len() != 11
                || kernel.symbol != "dispatch_planes_kernel"
                || !matches!(canonical.args[8], Argument::I32 { value: 4096 })
                || !matches!(kernel.args[10], Argument::I32 { value: 4096 })
            {
                return Err("Invalid direct expert dispatch contract".into());
            }
            let source = argument(canonical, 1)?;
            let layer = slot
                .rsplit_once('/')
                .ok_or("Missing expert dispatch layer")?
                .0;
            let pack: Vec<_> = contracts.values().copied().filter(|k| {
                k.name.starts_with(&format!("{layer}/")) && k.symbol == "pack_exact_a8_kernel"
                    && matches!(k.args.first(), Some(Argument::Buffer { name }) if name == source)
            }).collect();
            let [pack] = pack.as_slice() else {
                return Err("Expert dispatch requires one matching plane pack slot".into());
            };
            if pack.args.len() != 5 || !matches!(pack.args[4], Argument::I32 { value: 40960 }) {
                return Err("Invalid expert plane pack extent".into());
            }
            for (from, to) in [(0, 0), (2, 1), (3, 3), (4, 5), (5, 6), (6, 7), (7, 8)] {
                if argument(canonical, from)? != argument(kernel, to)? {
                    return Err("Direct expert dispatch changes its routing operands".into());
                }
            }
            for (from, to) in [(1, 2), (2, 4), (3, 9)] {
                if argument(pack, from)? != argument(kernel, to)? {
                    return Err("Direct expert dispatch changes its activation planes".into());
                }
            }
            variants
                .entry(slot.to_owned())
                .or_default()
                .push(KernelVariant {
                    min_rows: 128,
                    max_rows: 4096,
                    multiple_of: 1,
                    kernels: vec![kernel.name.clone()],
                });
            variants
                .entry(pack.name.clone())
                .or_default()
                .push(KernelVariant {
                    min_rows: 128,
                    max_rows: 4096,
                    multiple_of: 1,
                    kernels: vec![],
                });
            continue;
        }
        if suffix == "hc_finish" && !contracts.contains_key(format!("{slot}/hc_partials").as_str())
        {
            return Err("Missing HC stream producer".into());
        }
        if suffix == "hc_partials" {
            let canonical = contracts.get(slot).ok_or("Missing HC projection slot")?;
            let finish = contracts
                .get(format!("{slot}/hc_finish").as_str())
                .ok_or("Missing HC stream finish")?;
            let extent = |k: &Kernel, index: usize| {
                matches!(k.args.get(index), Some(Argument::I32 { value: 4096 }))
            };
            if !slot.starts_with("prefill_m4096/")
                || canonical.symbol != "hc_prefill_projection_grouped_kernel"
                || kernel.symbol != "hc_prefill_partial_kernel"
                || finish.symbol != "hc_prefill_finish_kernel"
                || canonical.args.len() != 4
                || kernel.args.len() != 4
                || finish.args.len() != 3
                || !extent(canonical, 3)
                || !extent(kernel, 3)
                || !extent(finish, 2)
                || argument(canonical, 0)? != argument(kernel, 0)?
                || argument(canonical, 2)? != argument(kernel, 2)?
                || argument(canonical, 1)? != argument(finish, 0)?
                || argument(kernel, 1)? != argument(finish, 1)?
            {
                return Err("HC stream projection producer/consumer mismatch".into());
            }
            let partial_name = argument(kernel, 1)?;
            let partial = buffers
                .iter()
                .find(|b| b.name == partial_name)
                .ok_or("Missing HC partial workspace")?;
            if partial.dtype != Dtype::F32
                || partial.shape != [4, 4096, 320]
                || partial.data.is_some()
                || partial.access != Access::ReadWrite
            {
                return Err("Invalid HC partial workspace".into());
            }
            for (index, dtype, elements, immutable) in [
                (0, Dtype::Bf16, 4096 * 10240, false),
                (1, Dtype::Bf16, 4096 * 320, false),
                (2, Dtype::Bf16, 320 * 10240, true),
            ] {
                let name = argument(canonical, index)?;
                let b = buffers
                    .iter()
                    .find(|b| b.name == name)
                    .ok_or("Missing HC canonical operand")?;
                if b.dtype != dtype
                    || b.bytes()? != elements * dtype.bytes()
                    || b.data.is_some() != immutable
                    || b.layout != "native_contiguous"
                {
                    return Err("Invalid HC canonical stream operand".into());
                }
            }
            variants
                .entry(slot.to_owned())
                .or_default()
                .push(KernelVariant {
                    min_rows: 1,
                    max_rows: 4096,
                    multiple_of: 1,
                    kernels: vec![kernel.name.clone(), finish.name.clone()],
                });
            continue;
        }
        let (min_rows, max_rows, multiple_of, symbol) = match suffix {
            "aligned_rows" => (128, 4096, 128, "hc_prefill_up_mix_kernel"),
            "medium_rows" => (128, 1023, 1, "main_kernel"),
            "large_rows" => (1024, 4096, 1, "main_kernel"),
            _ => continue,
        };
        let canonical = contracts
            .get(slot)
            .ok_or("Missing canonical prefill variant slot")?;
        let same = canonical.args.len() == kernel.args.len()
            && canonical
                .args
                .iter()
                .zip(&kernel.args)
                .all(|(a, b)| match (a, b) {
                    (Argument::Buffer { name: a }, Argument::Buffer { name: b }) => a == b,
                    (Argument::I32 { value: a }, Argument::I32 { value: b }) => a == b,
                    _ => false,
                });
        if !slot.starts_with("prefill_m4096/")
            || canonical.symbol != symbol
            || kernel.symbol != symbol
            || !same
        {
            return Err("Prefill variant changes the canonical operand contract".into());
        }
        variants
            .entry(slot.to_owned())
            .or_default()
            .push(KernelVariant {
                min_rows,
                max_rows,
                multiple_of,
                kernels: vec![kernel.name.clone()],
            });
    }
    for choices in variants.values_mut() {
        choices.sort_by_key(|v| v.min_rows);
    }
    Ok(variants)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn kernel(name: &str, buffer: &str) -> Kernel {
        serde_json::from_value(serde_json::json!({
            "name":name,"module":{"file":"k.cubin","sha256":"0".repeat(64)},
            "source":{"file":"k.cu","sha256":"0".repeat(64)},
            "host_abi":{"file":"host.txt","sha256":"0".repeat(64)},
            "symbol":"main_kernel","grid":[1,1,1],"block":[128,1,1],"shared_memory_bytes":0,"cooperative":false,
            "args":[{"kind":"buffer","name":buffer},{"kind":"i32","value":4096}]
        }))
        .unwrap()
    }
    #[test]
    fn tiling_ranges_keep_the_same_operands() {
        let slot = "prefill_m4096/layer0/k36";
        let v = build::<()>(
            &[
                kernel(slot, "A"),
                kernel(&format!("{slot}/large_rows"), "A"),
                kernel(&format!("{slot}/medium_rows"), "A"),
            ],
            &[],
        )
        .unwrap();
        assert_eq!((v[slot][0].min_rows, v[slot][0].max_rows), (128, 1023));
        assert_eq!((v[slot][1].min_rows, v[slot][1].max_rows), (1024, 4096));
        assert!(
            build::<()>(
                &[
                    kernel(slot, "A"),
                    kernel(&format!("{slot}/medium_rows"), "B")
                ],
                &[]
            )
            .is_err()
        );
        assert!(build::<()>(&[kernel(&format!("{slot}/medium_rows"), "A")], &[]).is_err());
    }

    #[test]
    fn hc_stream_pair_checks_workspace_and_endpoints() {
        let slot = "prefill_m4096/layer0/k3";
        let args = |names: &[&str]| -> Vec<Argument> {
            names
                .iter()
                .map(|name| Argument::Buffer {
                    name: (*name).into(),
                })
                .chain([Argument::I32 { value: 4096 }])
                .collect()
        };
        let mut canonical = kernel(slot, "A");
        canonical.symbol = "hc_prefill_projection_grouped_kernel".into();
        canonical.args = args(&["A", "C", "W"]);
        let mut partial = kernel(&format!("{slot}/hc_partials"), "A");
        partial.symbol = "hc_prefill_partial_kernel".into();
        partial.args = args(&["A", "P", "W"]);
        let mut finish = kernel(&format!("{slot}/hc_finish"), "A");
        finish.symbol = "hc_prefill_finish_kernel".into();
        finish.args = args(&["C", "P"]);
        let buffer = |name: &str, dtype, shape, data| Buffer {
            name: name.into(),
            dtype,
            shape,
            layout: "native_contiguous".into(),
            alignment: 256,
            access: if data {
                Access::Read
            } else {
                Access::ReadWrite
            },
            data: data.then_some(()),
        };
        let mut buffers = vec![
            buffer("P", Dtype::F32, vec![4, 4096, 320], false),
            buffer("A", Dtype::Bf16, vec![4096, 4, 2560], false),
            buffer("C", Dtype::Bf16, vec![4096, 320], false),
            buffer("W", Dtype::Bf16, vec![320, 10240], true),
        ];
        let kernels = vec![canonical, partial, finish];
        let v = build(&kernels, &buffers).unwrap();
        assert_eq!(
            v[slot][0].kernels,
            [format!("{slot}/hc_partials"), format!("{slot}/hc_finish")]
        );
        assert!(build(&kernels[..2], &buffers).is_err());
        assert!(build(&[kernels[0].clone(), kernels[2].clone()], &buffers).is_err());
        buffers[0].shape[0] = 3;
        assert!(build(&kernels, &buffers).is_err());
        buffers[0].shape[0] = 4;
        let mut invalid = kernels.clone();
        invalid[2].args[1] = Argument::Buffer {
            name: "Other".into(),
        };
        assert!(build(&invalid, &buffers).is_err());
    }

    #[test]
    fn direct_dispatch_consumes_only_its_matching_pack() {
        let slot = "prefill_m4096/layer0/k44";
        let pack_slot = "prefill_m4096/layer0/k45";
        let args = |names: &[&str], value| -> Vec<Argument> {
            names
                .iter()
                .map(|name| Argument::Buffer {
                    name: (*name).into(),
                })
                .chain([Argument::I32 { value }])
                .collect()
        };
        let mut canonical = kernel(slot, "A");
        canonical.args = args(
            &[
                "A", "D", "DS", "IDs", "Relative", "Offsets", "Scale", "SlotMap",
            ],
            4096,
        );
        let mut pack = kernel(pack_slot, "D");
        pack.symbol = "pack_exact_a8_kernel".into();
        pack.args = args(&["D", "High", "Low", "Sum"], 40960);
        let mut direct = kernel(&format!("{slot}/dispatch_planes"), "A");
        direct.symbol = "dispatch_planes_kernel".into();
        direct.args = args(
            &[
                "A", "DS", "High", "IDs", "Low", "Relative", "Offsets", "Scale", "SlotMap", "Sum",
            ],
            4096,
        );
        let kernels = vec![canonical, pack, direct];
        let v = build::<()>(&kernels, &[]).unwrap();
        assert_eq!(v[slot][0].min_rows, 128);
        assert!(v[pack_slot][0].kernels.is_empty());
        assert!(build::<()>(&[kernels[0].clone(), kernels[2].clone()], &[]).is_err());
        let mut invalid = kernels.clone();
        invalid[2].args[4] = Argument::Buffer {
            name: "OtherLow".into(),
        };
        assert!(build::<()>(&invalid, &[]).is_err());
    }
}
