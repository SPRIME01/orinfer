//! Table-Q4A8: shared row computation with request-owned causal state.
use super::config::Configuration;
use crate::{
    adapter::Adapter,
    artifact::{Access, Dtype, Result},
    execution::{BufferView, Invocation},
    model::{KvGrowth, Manifest, Operation, PrefillPlan, SegmentControls},
    policy::Policy,
};
use orinfer_model_sdk::{
    abi::{self, CreateRequest, CreatedPlan},
    architecture::{BatchLayout, BatchSegment},
    operators::rows::RowExecution,
};
use std::collections::{BTreeMap, BTreeSet};

const CAPACITY: usize = 4096;
const LANES: usize = 128;
const PREFIX: &str = "qwen_rows";
#[path = "mtp.rs"]
mod mtp;
const PRE_GDN: &[&str] = &["norm_rotate", "qkv", "z", "ab"];
const PRE_ATTN: &[&str] = &["norm_rotate", "q", "k", "v"];
const PRIVATE_GDN: &[&str] = &["gates", "conv", "history", "recurrent", "gated_norm"];
const PRIVATE_ATTN: &[&str] = &["prepare", "attention", "attention_merge"];
const POST: &[&str] = &[
    "quantize",
    "out",
    "norm_rotate",
    "gate",
    "up",
    "swiglu_rotate",
    "swiglu_quantize",
    "down",
];
const HEAD: &[&str] = &["rotate", "project", "partials", "merge"];

struct QwenRows {
    manifest: Manifest,
}
fn kernels(prefix: &str, names: &[&str]) -> Vec<Operation> {
    names
        .iter()
        .map(|name| Operation::Kernel {
            name: format!("{prefix}/{name}"),
        })
        .collect()
}
fn section(m: &mut Manifest, suffix: &str, names: &[&str]) -> Vec<Operation> {
    let name = format!("{PREFIX}/{suffix}");
    let ops = kernels(&name, names);
    m.programs.insert(name, ops.clone());
    ops
}
fn validate_weights(m: &Manifest, config: &Configuration, e8p: bool) -> Result<()> {
    let t = config.text();
    let map: BTreeMap<_, _> = m.buffers.iter().map(|b| (b.name.as_str(), b)).collect();
    let check = |name: String, dtype: Dtype, shape: Vec<usize>| -> Result<()> {
        if map.get(name.as_str()).is_none_or(|b| {
            b.dtype != dtype || b.shape != shape || b.access != Access::Read || b.data.is_none()
        }) {
            return Err(format!("Invalid Qwen Q4A8 operand: {name}"));
        }
        Ok(())
    };
    if e8p {
        check("orinfer.e8p_q4.basis".into(), Dtype::I8, vec![256, 8])?;
        check("orinfer.e8p_q4.selectors".into(), Dtype::U32, vec![256, 1])?;
    }
    let q4 = |name: String, n: usize, k: usize| -> Result<()> {
        let (dtype, shape) = if e8p {
            (Dtype::U32, vec![k / 128, n, 16])
        } else {
            (Dtype::U8, vec![n, k / 2])
        };
        let packed = format!("{name}.packed");
        check(packed.clone(), dtype, shape)?;
        if e8p && map[packed.as_str()].layout != "e8p-q4-k128-u32" {
            return Err(format!("Invalid Qwen E8P physical layout: {packed}"));
        }
        check(format!("{name}.scale"), Dtype::F16, vec![n])
    };
    q4("lm_head.weight".into(), t.vocab_size, t.hidden_size)?;
    for (layer, kind) in t.layer_types.iter().enumerate() {
        let p = format!("model.language_model.layers.{layer}");
        for (name, n, k) in [
            ("gate_proj", t.intermediate_size, t.hidden_size),
            ("up_proj", t.intermediate_size, t.hidden_size),
            ("down_proj", t.hidden_size, t.intermediate_size),
        ] {
            q4(format!("{p}.mlp.{name}.weight"), n, k)?;
        }
        let shapes: &[(&str, usize)] = if kind == "linear_attention" {
            &[
                ("linear_attn.in_proj_qkv", 10240),
                ("linear_attn.in_proj_z", 6144),
            ]
        } else {
            &[
                ("self_attn.q_proj", 12288),
                ("self_attn.k_proj", 1024),
                ("self_attn.v_proj", 1024),
            ]
        };
        for &(name, n) in shapes {
            q4(format!("{p}.{name}.weight"), n, t.hidden_size)?;
        }
    }
    Ok(())
}

fn vision(m: &mut Manifest, depth: usize) -> Result<()> {
    let Some(v) = m.vision.as_ref() else {
        return Ok(());
    };
    if depth != 27
        || v.hidden != 5120
        || v.max_patches > 32768
        || v.plans.len() != 1
        || v.plans[0].min_patches != 4
        || v.plans[0].patches != v.max_patches
        || v.plans[0].program != "vision"
    {
        return Err("Unsupported Qwen compact vision contract".into());
    }
    let mut operations = kernels("qwen_rows/vision/begin", &["patch", "position"]);
    for layer in 0..depth {
        operations.extend(kernels(
            &format!("qwen_rows/vision/layer{layer}"),
            &[
                "norm1",
                "qkv",
                "rope",
                "attention",
                "projection",
                "residual1",
                "norm2",
                "fc1",
                "fc2",
                "residual2",
            ],
        ));
    }
    operations.extend(kernels(
        "qwen_rows/vision/end",
        &["norm", "merge1", "merge2"],
    ));
    m.programs.insert("vision".into(), operations);
    Ok(())
}

pub(super) fn create(mut request: CreateRequest, policy: Policy) -> Result<crate::registry::Built> {
    if request.architecture != "qwen3_5" || policy != Policy::Int8Quality {
        return Err("Unsupported Qwen row precision policy".into());
    }
    let quant = &request.config["quantization_config"];
    let e8p = match quant["quant_method"].as_str() {
        Some("orinfer_q4_int8") => false,
        Some("orinfer_e8p_q4_int8") => true,
        _ => return Err("Unsupported Qwen Q4A8 codec".into()),
    };
    if e8p
        && (quant["coefficients"] != serde_json::json!([4, 1])
            || quant["basis_tensor"] != "orinfer.e8p_q4.basis"
            || quant["layout"] != "u32-low-coarse-high-residual")
    {
        return Err("Unsupported Qwen E8P-Q4 encoding".into());
    }
    for (key, value) in [
        ("version", serde_json::json!(1)),
        ("rotation", serde_json::json!("signed-block128")),
        ("scale", serde_json::json!("fp16-output-row")),
        ("compute_dtype", serde_json::json!("int8_quality")),
    ] {
        if quant[key] != value {
            return Err(format!("Unsupported Qwen Q4A8 {key}"));
        }
    }
    let c = Configuration::parse(request.config)?;
    if c.signature != request.expected_signature {
        return Err("Qwen row architecture signature differs".into());
    }
    let t = c.text();
    if (
        t.hidden_size,
        t.intermediate_size,
        t.num_hidden_layers,
        t.vocab_size,
    ) != (5120, 17408, 64, 248320)
        || request.prefill_profiles.len() != 1
        || request.prefill_profiles[0].tokens != CAPACITY
        || request.prefill_profiles[0].min_tokens != 1
        || request.prefill_profiles[0].kind != "qwen_q4_rows"
    {
        return Err("Expected Qwen 27B compact 1..4096 row contract".into());
    }
    let m = &mut request.metadata;
    validate_weights(m, &c, e8p)?;
    m.programs.clear();
    let begin = if m.vision.is_some() {
        vec!["prepare", "embedding", "overlay"]
    } else {
        vec!["prepare", "embedding"]
    };
    let mut forward = section(m, "begin", &begin);
    let mut strides = BTreeMap::new();
    for (name, width, size) in [
        ("Hidden", 5120, 2),
        ("R0", 5120, 4),
        ("R1", 5120, 4),
        ("Norm", 5120, 2),
        ("QKV", 10240, 2),
        ("Zout", 6144, 2),
        ("AB", 96, 2),
        ("FullX", 14336, 2),
        ("FullQ", 6144, 2),
        ("FullGate", 6144, 2),
        ("MixerIn", 6144, 2),
        ("Mix", 5120, 2),
        ("GateUp", 34816, 2),
    ] {
        strides.insert(name.into(), width * size);
    }
    for (layer, kind) in t.layer_types.iter().enumerate() {
        let gdn = kind == "linear_attention";
        forward.extend(section(
            m,
            &format!("layer{layer}/pre"),
            if gdn { PRE_GDN } else { PRE_ATTN },
        ));
        let mut private = section(
            m,
            &format!("layer{layer}/private"),
            if gdn { PRIVATE_GDN } else { PRIVATE_ATTN },
        );
        if gdn {
            // History is a fixed-size private state, independent of live rows.
            private[2] = Operation::Copy {
                source: "HistoryOut".into(),
                destination: format!("L{layer}_History"),
                bytes: 3 * 10240 * 2,
            };
            m.programs
                .insert(format!("{PREFIX}/layer{layer}/private"), private.clone());
        }
        forward.extend(private);
        forward.extend(section(m, &format!("layer{layer}/post"), POST));
    }
    forward.extend(section(m, "end", &["advance"]));
    let gather = section(m, "gather", &["norm"]);
    section(m, "head", HEAD);
    let head = section(m, "head_last", HEAD);
    let select = section(m, "select", &["selection"]);
    let mut head_last = gather;
    head_last.extend(head);
    head_last.push(Operation::Copy {
        source: "HeadLogits".into(),
        destination: m.logits.clone(),
        bytes: m.vocab * 4,
    });
    head_last.extend(select);
    m.programs.insert("head".into(), head_last);
    m.programs.insert("prefill".into(), forward.clone());
    let mut decode = vec![Operation::Copy {
        source: m.token.clone(),
        destination: m.input.clone(),
        bytes: 4,
    }];
    decode.extend(forward);
    decode.extend(m.programs["head"].clone());
    m.programs.insert("decode".into(), decode);
    m.prefill_plans = vec![PrefillPlan {
        chunk_tokens: CAPACITY,
        min_tokens: 1,
        prefill_program: "prefill".into(),
        head_program: "head".into(),
        stages: vec![],
        live_buffers: vec![],
    }];
    m.batch_profiles = vec![1, LANES];
    m.prefill_batch_profiles = request.prefill_profiles.clone();
    m.segment_controls = Some(SegmentControls {
        length: "SegmentLength".into(),
        last_index: "LastIndex".into(),
    });
    m.batch_layout = Some(BatchLayout {
        layers: t.layer_types.clone(),
        row_strides: strides,
        profiles: BTreeMap::from([(CAPACITY, "qwen_q4_rows".into())]),
        hidden: t.hidden_size,
        small_mixed_shapes: vec![],
        segment_tokens: CAPACITY,
        state_columns: vec![],
        preparation_programs: BTreeMap::from([(1, "decode".into()), (CAPACITY, "prefill".into())]),
    });
    mtp::register(m)?;
    vision(m, c.vision_depth)?;
    let programs = m.programs.keys().cloned().collect::<BTreeSet<_>>();
    m.row_execution = Some(RowExecution {
        capacity: m
            .vision
            .as_ref()
            .map_or(CAPACITY, |v| CAPACITY.max(v.max_patches)),
        programs: programs.clone(),
        program_capacities: programs
            .iter()
            .map(|p| {
                (
                    p.clone(),
                    match p.as_str() {
                        "vision" => m.vision.as_ref().map_or(CAPACITY, |v| v.max_patches),
                        "decode"
                        | "mtp_draft"
                        | "mtp_head"
                        | "qwen_rows/select"
                        | "qwen_rows/head_last" => 1,
                        "mtp_warm" => 512,
                        "verify" | "restore" => 8,
                        "qwen_rows/head" => LANES,
                        _ => CAPACITY,
                    },
                )
            })
            .collect(),
        variants: BTreeMap::new(),
    });
    let slots = m
        .programs
        .values()
        .flatten()
        .filter_map(|op| match op {
            Operation::Kernel { name } => Some(name.as_str()),
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    let rows = m
        .row_execution
        .as_mut()
        .ok_or("Missing Qwen dynamic rows")?;
    for kernel in &request.kernel_contracts {
        let Some((slot, choice)) =
            orinfer_model_sdk::operators::rows::specialization(&kernel.name, CAPACITY)?
        else {
            continue;
        };
        if !slots.contains(slot) {
            return Err("Qwen local row choice has no computation slot".into());
        }
        rows.variants.entry(slot.into()).or_default().push(choice);
    }
    for choices in rows.variants.values_mut() {
        choices.sort_by_key(|v| v.min_rows);
    }
    let kv = m
        .kv_cache
        .as_mut()
        .ok_or("Qwen rows require INT8 KV storage")?;
    {
        kv.growth.insert(
            "prefill".to_owned(),
            KvGrowth {
                position: m.position.clone(),
                tokens: CAPACITY,
                length: Some("SegmentLength".into()),
                buffers: kv
                    .buffers
                    .keys()
                    .filter(|name| !name.starts_with("Mtp"))
                    .cloned()
                    .collect(),
            },
        );
    }
    // The one-token decoder is captured independently of prompt segments.
    kv.growth.insert(
        "decode".into(),
        KvGrowth {
            position: m.position.clone(),
            tokens: 1,
            length: None,
            buffers: kv
                .buffers
                .keys()
                .filter(|name| !name.starts_with("Mtp"))
                .cloned()
                .collect(),
        },
    );
    let metadata = request.metadata;
    Ok((
        Box::new(QwenRows {
            manifest: metadata.clone(),
        }),
        CreatedPlan {
            metadata,
            decode_programs: BTreeSet::from(["decode".into()]),
            expert_banks: vec![],
        },
    ))
}

fn append(
    m: &Manifest,
    out: &mut Vec<Invocation>,
    program: &str,
    rows: usize,
    owner: Option<usize>,
    views: &BTreeMap<String, BufferView>,
) -> Result<()> {
    let template = m
        .row_execution
        .as_ref()
        .ok_or("Missing Qwen row template")?;
    let mut calls = template.specialize(
        program,
        &m.programs[program],
        &m.dynamic_batch_kernels,
        rows,
    )?;
    for call in &mut calls {
        call.sequence = owner;
        call.views = views.clone();
    }
    out.extend(calls);
    Ok(())
}
impl Adapter for QwenRows {
    fn prepare(
        &self,
        program: &str,
        tokens: &[u32],
        _history: &[u32],
    ) -> Result<Vec<(String, Vec<u8>)>> {
        if tokens.is_empty() {
            return Err("Empty Qwen causal segment".into());
        }
        let length =
            u32::try_from(tokens.len()).map_err(|_| "Qwen causal segment length overflow")?;
        let name = if program == "mtp_warm" || program == "mtp_draft" {
            "MtpSegmentLength"
        } else {
            "SegmentLength"
        };
        Ok(vec![(name.into(), length.to_le_bytes().to_vec())])
    }
    fn visual(
        &self,
        tokens: &[u32],
        images: &[abi::ImageGrid],
        capacity: usize,
    ) -> Result<(Vec<i32>, Vec<u32>)> {
        crate::vision::layout(
            self.manifest
                .vision
                .as_ref()
                .ok_or("Model has no vision adapter")?,
            tokens,
            images,
            capacity,
        )
    }
    fn batch(&self, segments: &[BatchSegment], include_plan: bool) -> Result<abi::BatchPlan> {
        let m = &self.manifest;
        let total: usize = segments
            .iter()
            .map(|s| s.tokens)
            .try_fold(0usize, usize::checked_add)
            .ok_or("Qwen packet row overflow")?;
        let mut owners = BTreeSet::new();
        if segments.is_empty()
            || segments.len() > LANES
            || total > CAPACITY
            || segments
                .iter()
                .any(|s| s.tokens == 0 || !owners.insert(s.slot))
        {
            return Err("Invalid Qwen causal row packet".into());
        }
        if !include_plan {
            return Ok((vec![], vec![]));
        }
        let layout = m.batch_layout.as_ref().ok_or("Missing Qwen row arena")?;
        let (mut out, mut offset, mut views) = (Vec::new(), 0, Vec::new());
        let empty = BTreeMap::new();
        for segment in segments {
            let v = layout
                .row_strides
                .iter()
                .map(|(name, stride)| {
                    (
                        name.clone(),
                        BufferView {
                            buffer: name.clone(),
                            offset: offset * stride,
                        },
                    )
                })
                .collect();
            if segment.tokens == 1 {
                out.push(Invocation {
                    operation: Operation::Copy {
                        source: m.token.clone(),
                        destination: m.input.clone(),
                        bytes: 4,
                    },
                    sequence: Some(segment.slot),
                    launch: None,
                    views: BTreeMap::new(),
                });
            }
            append(
                m,
                &mut out,
                &format!("{PREFIX}/begin"),
                segment.tokens,
                Some(segment.slot),
                &v,
            )?;
            views.push(v);
            offset += segment.tokens;
        }
        for layer in 0..layout.layers.len() {
            append(
                m,
                &mut out,
                &format!("{PREFIX}/layer{layer}/pre"),
                total,
                None,
                &empty,
            )?;
            for (segment, v) in segments.iter().zip(&views) {
                append(
                    m,
                    &mut out,
                    &format!("{PREFIX}/layer{layer}/private"),
                    segment.tokens,
                    Some(segment.slot),
                    v,
                )?;
            }
            append(
                m,
                &mut out,
                &format!("{PREFIX}/layer{layer}/post"),
                total,
                None,
                &empty,
            )?;
        }
        for (lane, (segment, v)) in segments.iter().zip(&views).enumerate() {
            append(
                m,
                &mut out,
                &format!("{PREFIX}/end"),
                segment.tokens,
                Some(segment.slot),
                v,
            )?;
            if m.mtp.is_some() {
                append(
                    m,
                    &mut out,
                    "capture",
                    segment.tokens,
                    Some(segment.slot),
                    v,
                )?;
            }
            let mut gather = v.clone();
            gather.insert(
                "HeadHidden".into(),
                BufferView {
                    buffer: "HeadHidden".into(),
                    offset: lane * 5120 * 2,
                },
            );
            append(
                m,
                &mut out,
                &format!("{PREFIX}/gather"),
                segment.tokens,
                Some(segment.slot),
                &gather,
            )?;
        }
        append(
            m,
            &mut out,
            &format!("{PREFIX}/head"),
            segments.len(),
            None,
            &empty,
        )?;
        for (lane, segment) in segments.iter().enumerate() {
            let v = BTreeMap::from([
                (
                    "HeadLogits".into(),
                    BufferView {
                        buffer: "HeadLogits".into(),
                        offset: lane * m.vocab * 4,
                    },
                ),
                (
                    "Pairs".into(),
                    BufferView {
                        buffer: "Pairs".into(),
                        offset: lane * 8,
                    },
                ),
            ]);
            out.push(Invocation {
                operation: Operation::Copy {
                    source: "HeadLogits".into(),
                    destination: m.logits.clone(),
                    bytes: m.vocab * 4,
                },
                sequence: Some(segment.slot),
                launch: None,
                views: v.clone(),
            });
            append(
                m,
                &mut out,
                &format!("{PREFIX}/select"),
                1,
                Some(segment.slot),
                &v,
            )?;
        }
        Ok((out, vec![]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn request() -> CreateRequest {
        let config = json!({"model_type":"qwen3_5","text_config":{"hidden_size":5120,
            "intermediate_size":17408,"num_hidden_layers":64,"vocab_size":248320,
            "linear_num_key_heads":16,"linear_num_value_heads":48,"linear_key_head_dim":128,
            "linear_value_head_dim":128,"linear_conv_kernel_dim":4,
            "layer_types":(0..64).map(|i|if i%4==3 {"full_attention"} else {"linear_attention"}).collect::<Vec<_>>()},
            "quantization_config":{"quant_method":"orinfer_q4_int8","version":1,"rotation":"signed-block128",
            "scale":"fp16-output-row","compute_dtype":"int8_quality"}});
        let signature = Configuration::parse(config.clone()).unwrap().signature;
        let mut buffers = vec![];
        let mut q4 = |name: String, n: usize, k: usize| {
            for (suffix, dtype, shape) in
                [("packed", "u8", vec![n, k / 2]), ("scale", "f16", vec![n])]
            {
                buffers.push(
                    json!({"name":format!("{name}.{suffix}"),"dtype":dtype,"shape":shape,
                    "layout":"native_contiguous","alignment":256,"access":"read",
                    "data":{"tensor":format!("{name}.{suffix}"),"sha256":"0".repeat(64)}}),
                );
            }
        };
        q4("lm_head.weight".into(), 248320, 5120);
        for i in 0..64 {
            let p = format!("model.language_model.layers.{i}");
            for (name, n, k) in [
                ("gate_proj", 17408, 5120),
                ("up_proj", 17408, 5120),
                ("down_proj", 5120, 17408),
            ] {
                q4(format!("{p}.mlp.{name}.weight"), n, k);
            }
            let projections: &[(&str, usize)] = if i % 4 == 3 {
                &[
                    ("self_attn.q_proj", 12288),
                    ("self_attn.k_proj", 1024),
                    ("self_attn.v_proj", 1024),
                ]
            } else {
                &[
                    ("linear_attn.in_proj_qkv", 10240),
                    ("linear_attn.in_proj_z", 6144),
                ]
            };
            for (name, n) in projections {
                q4(format!("{p}.{name}.weight"), *n, 5120);
            }
        }
        serde_json::from_value(json!({"config":config,"expected_signature":signature,
            "architecture":"qwen3_5","compute_policy":"int8_quality",
            "prefill_profiles":[{"tokens":4096,"min_tokens":1,"kind":"qwen_q4_rows"}],
            "metadata":{"schema_version":2,"target":"sm_87","model":"test","chunk_tokens":4096,"max_context":262144,
                "vocab":248320,"toolchain":{},"buffers":buffers,"reset_buffers":["Step"],"input":"Input","token":"Token",
                "status":"TokenStatus","logits":"Logits","position":"Step","weight_bytes":0,"weight_parameters":0,"weight_scope":"test",
                "kv_cache":{"direct_prefill":true,"demand_mapping":true,"buffers":{},"prefill_workspace":{}},
                "dynamic_batch_kernels":{"qwen_rows/layer0/pre/norm_rotate":{"name":"qwen_rows/layer0/pre/norm_rotate","capacity":4096,
                    "grid":[{"op":"rows"},{"op":"constant","value":1},{"op":"constant","value":1}],"arguments":[]}}}})).unwrap()
    }

    #[test]
    fn two_stage_codec_reuses_rows_and_rejects_scalar_storage() {
        let candidate = || {
            let mut r = request();
            r.config["quantization_config"]["quant_method"] = json!("orinfer_e8p_q4_int8");
            r.config["quantization_config"]["coefficients"] = json!([4, 1]);
            r.config["quantization_config"]["basis_tensor"] = json!("orinfer.e8p_q4.basis");
            r.config["quantization_config"]["layout"] = json!("u32-low-coarse-high-residual");
            r.expected_signature = Configuration::parse(r.config.clone()).unwrap().signature;
            for b in &mut r.metadata.buffers {
                if b.name.ends_with(".packed") {
                    b.dtype = Dtype::U32;
                    b.shape = vec![b.shape[1] * 2 / 128, b.shape[0], 16];
                    b.layout = "e8p-q4-k128-u32".into();
                }
            }
            for (name, dtype, shape) in [
                ("orinfer.e8p_q4.basis", "i8", vec![256, 8]),
                ("orinfer.e8p_q4.selectors", "u32", vec![256, 1]),
            ] {
                r.metadata.buffers.push(
                    serde_json::from_value(json!({
                        "name": name, "dtype": dtype, "shape": shape,
                        "layout": "native_contiguous", "alignment": 256, "access": "read",
                        "data": {"tensor": name, "sha256": "0".repeat(64)}
                    }))
                    .unwrap(),
                );
            }
            r
        };
        let (adapter, _) = create(candidate(), Policy::Int8Quality).unwrap();
        assert!(
            adapter
                .batch(&[BatchSegment { slot: 0, tokens: 1 }], false)
                .is_ok()
        );
        let mut wrong = candidate();
        wrong.metadata.buffers[0].dtype = Dtype::U8;
        assert!(create(wrong, Policy::Int8Quality).is_err());
        let mut wrong = candidate();
        let packed = wrong
            .metadata
            .buffers
            .iter_mut()
            .find(|b| b.name.ends_with(".packed"))
            .unwrap();
        packed.layout = "e8p-q4-row-u32".into();
        assert!(create(wrong, Policy::Int8Quality).is_err());
        let mut wrong = candidate();
        let packed = wrong
            .metadata
            .buffers
            .iter_mut()
            .find(|b| b.name.ends_with(".packed"))
            .unwrap();
        packed.shape.swap(0, 1);
        assert!(create(wrong, Policy::Int8Quality).is_err());
        let mut wrong = candidate();
        wrong.config["quantization_config"]["coefficients"] = json!([3, 1]);
        assert!(create(wrong, Policy::Int8Quality).is_err());
    }

    #[test]
    fn mixed_rows_share_computation_without_padding_and_keep_private_owners() {
        let (adapter, plan) = create(request(), Policy::Int8Quality).unwrap();
        let segments = [
            BatchSegment { slot: 7, tokens: 1 },
            BatchSegment {
                slot: 2,
                tokens: 17,
            },
        ];
        let (calls, _) = adapter.batch(&segments, true).unwrap();
        let shared:Vec<_>=calls.iter().filter(|call|matches!(&call.operation,Operation::Kernel{name} if name=="qwen_rows/layer0/pre/norm_rotate")).collect();
        assert_eq!(shared.len(), 1);
        assert_eq!(shared[0].sequence, None);
        assert_eq!(shared[0].launch.as_ref().unwrap().grid[0], 18);
        let conv:Vec<_>=calls.iter().filter(|call|matches!(&call.operation,Operation::Kernel{name} if name=="qwen_rows/layer0/private/conv")).collect();
        assert_eq!(
            conv.iter().map(|call| call.sequence).collect::<Vec<_>>(),
            [Some(7), Some(2)]
        );
        assert_eq!(conv[1].views["QKV"].offset, 10240 * 2);
        assert!(!conv[1].views.contains_key("Q")); // compact causal scratch is reused, never offset as token-major data
        let gather:Vec<_>=calls.iter().filter(|call|matches!(&call.operation,Operation::Kernel{name} if name=="qwen_rows/gather/norm")).collect();
        assert_eq!(gather[1].views["HeadHidden"].offset, 5120 * 2);
        assert_eq!(plan.metadata.prefill_plans.len(), 1);
        assert!(
            plan.metadata.programs["head"].iter().any(
                |op| matches!(op,Operation::Kernel{name} if name=="qwen_rows/head_last/project")
            )
        );
        assert!(
            !plan.metadata.programs["head"]
                .iter()
                .any(|op| matches!(op,Operation::Kernel{name} if name=="qwen_rows/head/project"))
        );
    }

    #[test]
    fn invalid_packet_and_unquantized_operands_fail_before_execution() {
        let (adapter, _) = create(request(), Policy::Int8Quality).unwrap();
        for segments in [
            vec![],
            vec![BatchSegment { slot: 1, tokens: 0 }],
            vec![BatchSegment {
                slot: 1,
                tokens: 4097,
            }],
            vec![
                BatchSegment { slot: 1, tokens: 1 },
                BatchSegment { slot: 1, tokens: 1 },
            ],
        ] {
            assert!(adapter.batch(&segments, true).is_err());
        }
        let mut wrong = request();
        wrong.metadata.buffers[0].dtype = Dtype::F16;
        assert!(create(wrong, Policy::Int8Quality).is_err());
        let mut wrong = request();
        wrong.prefill_profiles[0].min_tokens = 0;
        assert!(create(wrong, Policy::Int8Quality).is_err());
        let mut wrong = request();
        wrong.config["quantization_config"]["rotation"] = json!("none");
        assert!(create(wrong, Policy::Int8Quality).is_err());
    }
}
