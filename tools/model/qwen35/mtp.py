"""Native one-layer W8 draft and compact, exact-prefix target verification."""

from tools.model.qwen35.build import primitive

WARM = 512
VERIFY = 8


def build(b):
    from kernels.model import (
        qwen35,
        control,
        speculation,
        speculative_factors,
        causal_rows,
        flash_mtp,
        flash_control,
    )
    from kernels.model.gdn_sequence import _gdn_sequence, gdn_commit
    from kernels.model.kv_int8 import full_prepare_mrope_int8
    from kernels.operators.op02_residual_norm import residual_norm
    from kernels.operators.op23_final_norm import final_norm
    from kernels.operators.op30_activation_quantization import _compile as a8_compile
    from kernels.model.greedy import greedy_partials, greedy_merge
    from kernels.vision.features import overlay

    w = b.weights
    for name, dtype, shape, scope in [
        ("MtpStep", "i32", [1], "sequence"),
        ("MtpInput", "i32", [WARM], "sequence"),
        ("MtpToken", "i32", [1], "sequence"),
        ("MtpStatus", "i32", [1], "sequence"),
        ("MtpSeqLength", "i32", [1], "sequence"),
        ("MtpSegmentLength", "i32", [1], "sequence"),
        ("MtpLastIndex", "i32", [1], "sequence"),
        ("MtpPositions", "i32", [WARM], "sequence"),
        ("MtpCondition", "f16", [WARM, 5120], "sequence"),
        ("MtpTargetHidden", "f16", [4096, 5120], "sequence"),
        ("MtpFeatureIndex", "i32", [b.context], "sequence"),
        ("MtpEmbedding", "f16", [WARM, 5120], "workspace"),
        ("MtpConcat", "f16", [WARM, 10240], "workspace"),
        ("MtpHidden", "f16", [WARM, 5120], "workspace"),
        ("MtpNorm", "f16", [WARM, 5120], "workspace"),
        ("MtpR0", "f32", [WARM, 5120], "workspace"),
        ("MtpR1", "f32", [WARM, 5120], "workspace"),
        ("MtpFullX", "f16", [WARM, 14336], "workspace"),
        ("MtpFullQ", "f16", [WARM, 6144], "workspace"),
        ("MtpFullGate", "f16", [WARM, 6144], "workspace"),
        ("MtpMixer", "f16", [WARM, 6144], "workspace"),
        ("MtpMix", "f16", [WARM, 5120], "workspace"),
        ("MtpGateUp", "f16", [WARM, 34816], "workspace"),
        ("MtpLastHidden", "f16", [1, 5120], "workspace"),
        ("MtpLogits", "f32", [1, 248320], "sequence"),
        ("SequenceLogits", "f32", [VERIFY, 248320], "sequence"),
        ("SequenceTokens", "i32", [VERIFY], "sequence"),
        ("SequenceStatus", "i32", [VERIFY], "sequence"),
        ("AcceptedInputs", "i32", [1], "sequence"),
    ]:
        w.fixed(name, dtype, shape, scope)
    b.reset.extend(["MtpStep", "MtpToken", "MtpStatus", "MtpCondition"])
    for suffix, dtype, width in [
        ("KPages", "i8", 1024),
        ("VPages", "i8", 1024),
        ("KPagesScale", "f16", 16),
        ("VPagesScale", "f16", 16),
    ]:
        name = "Mtp" + suffix
        w.fixed(name, dtype, [b.context, width], "sequence")
        b.kv[name] = width * (2 if dtype == "f16" else 1)
        b.reset.append(name)

    def emit(role, key, fn, args, rows=WARM, *, dynamic=True):
        return b.emit("mtp/" + role, key, fn, args, rows, dynamic=dynamic)

    def quant(role, k, source, swiglu=False):
        emit(
            role,
            f"mtp-a8-{k}-{swiglu}",
            lambda: primitive(a8_compile, k, k, False, swiglu, 256),
            dict(X=source, Mask="UnusedA8Mask", Q="A8", S="AS"),
        )

    def project(role, weight, n, k, out, *, stride=None, offset=0):
        b.projection(
            "mtp/" + role, weight, n, k, out, stride=stride, offset=offset, rows=WARM, kind="w8"
        )

    emit(
        "gather",
        "mtp-gather",
        lambda: primitive(
            speculation.gather_target_hidden, WARM, 5120, 4096, ring=True, dynamic_rows=True
        ),
        dict(Target="MtpTargetHidden", Step="MtpStep", Out="MtpCondition"),
    )
    emit(
        "prepare",
        "mtp-prepare",
        lambda: primitive(qwen35.prepare, WARM, 5120, b.context, mrope_input=True),
        dict(
            Step="MtpStep",
            Positions="MtpPositions",
            Length="MtpSeqLength",
            SegmentLength="MtpSegmentLength",
            Last="MtpLastIndex",
            Requests="Requests",
            MRope="MRopePositions",
            Residual="MtpR0",
        ),
    )
    embed = "model.language_model.embed_tokens.weight"
    emit(
        "embedding",
        "mtp-embedding",
        lambda: primitive(qwen35.embedding, WARM, 248320, 5120),
        dict(
            IDs="MtpInput",
            Weight=w.immutable(embed + ".int8"),
            Scale=w.immutable(embed + ".scale"),
            Output="MtpEmbedding",
        ),
    )
    emit(
        "overlay",
        "mtp-overlay",
        lambda: primitive(
            overlay, WARM, 5120, b.context, b.vision["max_features"], dynamic_rows=True
        ),
        dict(
            Embedding="MtpEmbedding",
            Features="Features",
            Index="MtpFeatureIndex",
            Position="MtpStep",
        ),
    )
    emit(
        "concat",
        "mtp-concat",
        lambda: primitive(speculation.mtp_norm_concat, 5120),
        dict(
            Embedding="MtpEmbedding",
            Target="MtpCondition",
            EmbeddingWeight=w.immutable("mtp.pre_fc_norm_embedding.weight", cast="float16"),
            HiddenWeight=w.immutable("mtp.pre_fc_norm_hidden.weight", cast="float16"),
            Out="MtpConcat",
        ),
    )
    quant("fc_quant", 10240, "MtpConcat")
    project("fc", "mtp.fc.weight", 5120, 10240, "MtpHidden")
    emit(
        "pre_norm",
        "norm",
        lambda: primitive(residual_norm, None),
        dict(
            X="MtpHidden",
            R="MtpR0",
            W=w.immutable("mtp.layers.0.input_layernorm.weight", cast="float16"),
            Y="MtpNorm",
            RO="MtpR1",
        ),
    )
    quant("pre_quant", 5120, "MtpNorm")
    p = "mtp.layers.0.self_attn"
    for role, n, offset in [("q", 12288, 0), ("k", 1024, 12288), ("v", 1024, 13312)]:
        project(role, p + f".{role}_proj.weight", n, 5120, "MtpFullX", stride=14336, offset=offset)
    emit(
        "kv_prepare",
        f"kv-prepare-{b.context}",
        lambda: primitive(
            full_prepare_mrope_int8,
            (b.context + 127) // 128,
            b.context,
            (11, 11, 10),
            max_position=b.context,
        ),
        dict(
            X="MtpFullX",
            WQ=w.immutable(p + ".q_norm.weight", cast="float16"),
            WK=w.immutable(p + ".k_norm.weight", cast="float16"),
            Cache="Rotary",
            Req="Requests",
            Pos="MtpPositions",
            Pages="Pages",
            Status="PrepareStatus",
            MRope="MRopePositions",
            Q="MtpFullQ",
            Gate="MtpFullGate",
            K="MtpKPages",
            V="MtpVPages",
            KS="MtpKPagesScale",
            VS="MtpVPagesScale",
        ),
    )
    b.attention(
        "mtp/attention",
        q="MtpFullQ",
        gate="MtpFullGate",
        position="MtpPositions",
        length="MtpSeqLength",
        output="MtpMixer",
        prefix="Mtp",
        rows=WARM,
    )
    quant("out_quant", 6144, "MtpMixer")
    project("out", p + ".o_proj.weight", 5120, 6144, "MtpMix")
    emit(
        "post_norm",
        "norm",
        lambda: primitive(residual_norm, None),
        dict(
            X="MtpMix",
            R="MtpR1",
            W=w.immutable("mtp.layers.0.post_attention_layernorm.weight", cast="float16"),
            Y="MtpNorm",
            RO="MtpR0",
        ),
    )
    quant("post_quant", 5120, "MtpNorm")
    for role, offset in [("gate", 0), ("up", 17408)]:
        project(
            role,
            "mtp.layers.0.mlp." + role + "_proj.weight",
            17408,
            5120,
            "MtpGateUp",
            stride=34816,
            offset=offset,
        )
    quant("down_quant", 17408, "MtpGateUp", True)
    project("down", "mtp.layers.0.mlp.down_proj.weight", 5120, 17408, "MtpHidden")
    emit(
        "advance",
        "advance",
        lambda: primitive(control.advance, WARM, dynamic_rows=True),
        dict(Step="MtpStep"),
    )
    emit(
        "final_norm",
        "mtp-final-norm",
        lambda: primitive(final_norm, WARM, 1),
        dict(
            X="MtpHidden",
            R="MtpR0",
            I="MtpLastIndex",
            W=w.immutable("mtp.norm.weight", cast="float16"),
            Y="MtpLastHidden",
        ),
        1,
        dynamic=False,
    )
    b.rotation("mtp/head_rotate", 5120, "MtpLastHidden", rows=1, dynamic=False, head=True)
    b.projection(
        "mtp/head_project",
        "lm_head.weight",
        248320,
        5120,
        "MtpLogits",
        rows=1,
        dynamic=False,
        head=True,
    )
    emit(
        "partials",
        "greedy-partials-1-False",
        lambda: primitive(greedy_partials, 248320),
        dict(
            Logits="MtpLogits",
            Values="GreedyValues",
            Indices="GreedyIndices",
            Invalid="GreedyInvalid",
        ),
        1,
        dynamic=False,
    )
    emit(
        "merge",
        "greedy-merge-1-False",
        lambda: primitive(greedy_merge, 248320),
        dict(
            Values="GreedyValues", Indices="GreedyIndices", Invalid="GreedyInvalid", Output="Pairs"
        ),
        1,
        dynamic=False,
    )
    emit(
        "selection",
        "selection",
        lambda: primitive(flash_control.selection),
        dict(Pair="Pairs", Token="MtpToken", Status="MtpStatus"),
        1,
        dynamic=False,
    )
    b.emit(
        "capture/hidden",
        "target-hidden-capture",
        lambda: primitive(speculation.capture_target_hidden, 5120, 4096, ring=True),
        dict(
            X="Hidden",
            Residual="R0",
            Weight=w.immutable("model.language_model.norm.weight", cast="float16"),
            Step="Step",
            Out="MtpTargetHidden",
        ),
    )
    for i, kind in enumerate(b.text["layer_types"]):
        if kind != "linear_attention":
            continue
        for suffix, dtype, shape in [
            ("VerifyUpdate", "f32", [VERIFY, 48, 128]),
            ("VerifyK", "f16", [16, VERIFY, 128]),
            ("VerifyG", "f32", [VERIFY, 48]),
            ("HistoryPrefixes", "f16", [VERIFY, 3, 10240]),
        ]:
            w.fixed(f"L{i}_{suffix}", dtype, shape, "sequence")
        b.emit(
            f"verify/layer{i}/history",
            "verify-history",
            lambda: primitive(flash_mtp.history_prefix, VERIFY, 10240, 3, dynamic_rows=True),
            dict(X="QKV", State=f"L{i}_History", Prefix=f"L{i}_HistoryPrefixes"),
            VERIFY,
        )
        b.emit(
            f"verify/layer{i}/recurrent",
            "verify-recurrent",
            lambda: primitive(
                _gdn_sequence, VERIFY, value_tile=16, threads=32, compact=True, dynamic_rows=True
            ),
            dict(
                Q="Q",
                K="K",
                V="V",
                G="g",
                Beta="Beta",
                State=f"L{i}_State",
                Prefix=f"L{i}_VerifyUpdate",
                Out="Y",
            ),
            VERIFY,
        )
        b.emit(
            f"verify/layer{i}/factors",
            "verify-factors",
            lambda: primitive(speculative_factors.save_gdn_factors, VERIFY, dynamic_rows=True),
            dict(K="K", G="g", SavedK=f"L{i}_VerifyK", SavedG=f"L{i}_VerifyG"),
            VERIFY,
        )
        b.emit(
            f"restore/layer{i}/state",
            "verify-commit",
            lambda: primitive(gdn_commit, VERIFY, value_tile=16, dynamic_rows=True),
            dict(
                K=f"L{i}_VerifyK",
                G=f"L{i}_VerifyG",
                Update=f"L{i}_VerifyUpdate",
                Accepted="AcceptedInputs",
                State=f"L{i}_State",
            ),
            VERIFY,
        )
        b.emit(
            f"restore/layer{i}/history",
            "verify-history-commit",
            lambda: primitive(
                speculation.select_prefix, VERIFY, 3 * 10240, "float16", dynamic_rows=True
            ),
            dict(Prefix=f"L{i}_HistoryPrefixes", Count="AcceptedInputs", State=f"L{i}_History"),
            VERIFY,
        )
    identity = w.constant("RowIdentity", __import__("numpy").arange(128, dtype="int32"))
    b.emit(
        "verify/norm",
        "verify-norm",
        lambda: primitive(final_norm, None, None),
        dict(
            X="Hidden",
            R="R0",
            I=identity,
            W=w.immutable("model.language_model.norm.weight", cast="float16"),
            Y="HeadHidden",
        ),
        VERIFY,
    )
    b.emit(
        "verify/logits",
        "verify-logits",
        lambda: primitive(causal_rows.copy, VERIFY, 248320, "float32"),
        dict(Source="HeadLogits", Destination="SequenceLogits"),
        VERIFY,
    )
    b.emit(
        "verify/selections",
        "verify-selections",
        lambda: primitive(flash_control.selections, VERIFY, dynamic_rows=True),
        dict(Pair="Pairs", Tokens="SequenceTokens", Status="SequenceStatus"),
        VERIFY,
    )
    b.mtp = dict(
        position="MtpStep",
        input="MtpInput",
        token="MtpToken",
        status="MtpStatus",
        verification_tokens="SequenceTokens",
        verification_status="SequenceStatus",
        draft_logits="MtpLogits",
        verification_logits="SequenceLogits",
        feature_index="MtpFeatureIndex",
        accepted_inputs="AcceptedInputs",
        target_length="SeqLength",
        draft_program="mtp_draft",
        hidden_ring="MtpTargetHidden",
        commit_always=True,
        default_verification_tokens=4,
        warm_plans=[dict(tokens=WARM, min_tokens=1, program="mtp_warm", head_program="mtp_head")],
        capture_plans=[dict(tokens=1, program="capture"), dict(tokens=4096, program="capture")],
        verification_plans=[
            dict(
                tokens=VERIFY,
                min_tokens=2,
                program="verify",
                restore_program="restore",
                capture_program="capture",
            )
        ],
        shared_verification_state=True,
    )
