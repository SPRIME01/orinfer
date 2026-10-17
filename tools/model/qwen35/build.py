"""Build native Qwen 27B Q4A8 model data and a schema-1 Rust/TileLang package.

Consumes only the completed checkpoint produced by tools.quantization.qwen35.
Weight shards are linked or losslessly K-repacked; transforms, scales and the
single packed weight copy remain fixed. All online scheduling and layer order are registered Rust code.
"""

import argparse
import hashlib
from contextlib import contextmanager
import fcntl
import json
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np
from tools.model.publication import atomic_model, file_hash, write_json
from tools.model.package import config_signature, execution_identity, model_library
from tools.model.compact import compact_package
from tools.model.qwen35.storage import Weights
from tools.quantization.row_q4 import MAGNITUDES
from tools.model.compile import primitive

ROOT = Path(__file__).resolve().parents[3]
CAPACITY = 4096
LANES = 128


def configuration(config, context):
    if config.get("model_type") != "qwen3_5":
        raise ValueError("Expected Qwen3_5 27B checkpoint")
    t = config["text_config"]
    expected = dict(
        hidden_size=5120,
        intermediate_size=17408,
        num_hidden_layers=64,
        vocab_size=248320,
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        num_attention_heads=24,
        num_key_value_heads=4,
        head_dim=256,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        tie_word_embeddings=False,
    )
    if any(t.get(k) != v or type(t.get(k)) is not type(v) for k, v in expected.items()) or t.get(
        "layer_types"
    ) != ["full_attention" if i % 4 == 3 else "linear_attention" for i in range(64)]:
        raise ValueError("Unsupported Qwen 27B arithmetic geometry")
    rope = t["rope_parameters"]
    if any(
        rope.get(k) != v
        for k, v in dict(
            rope_type="default",
            rope_theta=10000000,
            partial_rotary_factor=0.25,
            mrope_section=[11, 11, 10],
            mrope_interleaved=True,
        ).items()
    ):
        raise ValueError("Unsupported Qwen 27B RoPE contract")
    q = config.get("quantization_config", {})
    if any(
        q.get(k) != v or type(q.get(k)) is not type(v)
        for k, v in dict(
            version=1,
            rotation="signed-block128",
            scale="fp16-output-row",
            compute_dtype="int8_quality",
        ).items()
    ):
        raise ValueError("Expected our completed table-Q4A8 conversion")
    if q.get("quant_method") not in ("orinfer_q4_int8", "orinfer_e8p_q4_int8"):
        raise ValueError("Unsupported Qwen Q4 codec")
    if q["quant_method"] == "orinfer_e8p_q4_int8":
        if any(
            q.get(k) != v
            for k, v in dict(
                coefficients=[4, 1],
                basis_tensor="orinfer.e8p_q4.basis",
                layout="u32-low-coarse-high-residual",
            ).items()
        ) or any(type(v) is not int for v in q["coefficients"]):
            raise ValueError("Unsupported two-stage E8P-Q4 contract")
    if type(context) is not int or not 1 <= context <= min(t["max_position_embeddings"], 262144):
        raise ValueError("Invalid context capacity")
    return t


@contextmanager
def gpu_lock(path):
    if path is None:
        yield
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            import torch

            if torch.cuda.is_initialized():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            fcntl.flock(stream, fcntl.LOCK_UN)


class Builder:
    def __init__(self, source, root, output, cache, context, lock):
        from tools.model.qwen35.compiler import Aot

        self.source, self.root, self.output = source, root, output
        self.config = json.loads((source / "config.json").read_text())
        self.text = configuration(self.config, context)
        self.q4_codec = (
            "e8p_q4"
            if self.config["quantization_config"]["quant_method"] == "orinfer_e8p_q4_int8"
            else "q4"
        )
        state = json.loads((source / "progress.json").read_text())
        if not state.get("complete"):
            raise ValueError("Quantization is still running; no partial model may be published")
        self.state = state
        self.context = context
        self.lock = lock
        self.weights = Weights(source, root, packed_k=128 if self.q4_codec == "e8p_q4" else 0)
        self.aot = Aot(
            root / "cache", output, cache, self.weights.buffers, compile_lock=lambda: gpu_lock(lock)
        )
        self.aot.identity["weights"] = hashlib.sha256(state["contract"].encode()).hexdigest()
        from tools.model.row_tuning import RowTuner

        self.row_tuner = RowTuner(output / "row-tuning", self.aot.identity)
        self.workspace()
        from tools.model.qwen35 import vision

        vision.build(self)
        self.forward()
        self.head()
        from tools.model.qwen35 import mtp

        mtp.build(self)
        from tools.model.qwen35 import parallel

        parallel.build(self)

    def workspace(self):
        w = self.weights
        h, f, v = 5120, 17408, 248320
        c = CAPACITY
        for name, dtype, shape in [
            ("Hidden", "f16", [c, h]),
            ("R0", "f32", [c, h]),
            ("R1", "f32", [c, h]),
            ("Norm", "f16", [c, h]),
            ("QKV", "f16", [c, 10240]),
            ("Zout", "f16", [c, 6144]),
            ("AB", "f16", [c, 96]),
            ("g", "f32", [c, 48]),
            ("Beta", "f32", [c, 48]),
            ("Q", "f16", [16, c, 128]),
            ("K", "f16", [16, c, 128]),
            ("V", "f16", [48, c, 128]),
            ("HistoryOut", "f16", [1, 3, 10240]),
            ("PositionOut", "i32", [1]),
            ("GdnScratch", "f32", [1]),
            ("Y", "f16", [c, 6144]),
            ("MixerIn", "f16", [c, 6144]),
            ("FullX", "f16", [c, 14336]),
            ("FullQ", "f16", [c, 24, 256]),
            ("FullGate", "f16", [c, 24, 256]),
            ("Mix", "f16", [c, h]),
            ("GateUp", "f16", [2 * c, f]),
            ("RotationValues", "f16", [c, f]),
            ("RotationMaxima", "f32", [c, math.ceil(f / 256)]),
            ("AttentionPartial", "f32", [24, 32, 64, 256]),
            ("AttentionMaximum", "f32", [24, 32, 64]),
            ("AttentionDenominator", "f32", [24, 32, 64]),
            ("A8", "i8", [c * f]),
            ("AS", "f16", [c]),
            ("UnusedA8Mask", "u8", [f]),
            ("HeadHidden", "f16", [LANES, h]),
            ("HeadA8", "i8", [LANES, h]),
            ("HeadAS", "f16", [LANES]),
            ("HeadLogits", "f32", [LANES, v]),
            ("GreedyValues", "f32", [LANES * math.ceil(v / 1024)]),
            ("GreedyIndices", "i32", [LANES * math.ceil(v / 1024)]),
            ("GreedyInvalid", "i32", [LANES * math.ceil(v / 1024)]),
            ("Pairs", "i32", [LANES, 2]),
        ]:
            w.fixed(name, dtype, shape)
        for name, dtype, shape in [
            ("Input", "i32", [c]),
            ("Token", "i32", [1]),
            ("TokenStatus", "i32", [1]),
            ("Step", "i32", [1]),
            ("Positions", "i32", [c]),
            ("SeqLength", "i32", [1]),
            ("SegmentLength", "i32", [1]),
            ("LastIndex", "i32", [1]),
            ("Requests", "i32", [c]),
            ("MRopePositions", "i32", [self.context, 3]),
            ("PrepareStatus", "i32", [1]),
            ("Logits", "f32", [v]),
        ]:
            w.fixed(name, dtype, shape, "sequence")
        self.reset = ["Step", "Token", "TokenStatus"]
        self.kv = {}
        for i, kind in enumerate(self.text["layer_types"]):
            if kind == "linear_attention":
                for suffix, dtype, shape in [
                    ("History", "f16", [1, 3, 10240]),
                    ("State", "f32", [48, 128, 128]),
                ]:
                    name = f"L{i}_{suffix}"
                    w.fixed(name, dtype, shape, "sequence")
                    self.reset.append(name)
            else:
                for suffix, dtype, width in [
                    ("KPages", "i8", 1024),
                    ("VPages", "i8", 1024),
                    ("KPagesScale", "f16", 16),
                    ("VPagesScale", "f16", 16),
                ]:
                    name = f"L{i}_{suffix}"
                    w.fixed(name, dtype, [self.context, width], "sequence")
                    self.kv[name] = width * (2 if dtype == "f16" else 1)
                    self.reset.append(name)
        w.constant("Pages", np.arange(math.ceil(self.context / 128), dtype=np.int32)[None])
        freq = 1 / (10000000.0 ** (np.arange(0, 64, 2, dtype=np.float32) / 64))
        angles = np.arange(self.context, dtype=np.float32)[:, None] * freq[None]
        w.constant(
            "Rotary", np.concatenate([np.cos(angles), np.sin(angles)], axis=-1).astype(np.float16)
        )
        self.signs = {}
        for k in [h, f]:
            name = f"orinfer.rotation.k{k}.signs"
            signs = w.array(name)
            if signs.dtype != np.int8 or signs.shape != (k,) or not np.isin(signs, [-1, 1]).all():
                raise ValueError("Invalid rotation signs")
            self.signs[k] = w.constant(name, signs)
        if self.q4_codec == "e8p_q4":
            from tools.quantization.e8p import basis
            from tools.quantization.vq import e8p_selector_table

            table = w.array("orinfer.e8p_q4.basis")
            if table.dtype != np.int8 or not np.array_equal(table, basis()):
                raise ValueError("E8P-Q4 basis differs from compiled decoder")
            w.constant("orinfer.e8p_q4.basis", table)
            self.book = w.constant("orinfer.e8p_q4.selectors", e8p_selector_table(table))
        else:
            table = w.array("orinfer.q4.table")
            if not np.array_equal(
                table, np.array((*MAGNITUDES, *(-x for x in MAGNITUDES)), np.int8)
            ):
                raise ValueError("Q4 integer table differs from compiled decoder")
            w.constant("orinfer.q4.table", table)

    def emit(self, name, key, build, bindings, rows=CAPACITY, *, dynamic=True):
        return self.aot.emit("qwen_rows/" + name, key, build, bindings, rows, dynamic=dynamic)

    def norm_rotation(self, section, input, residual, weight, output_residual):
        from kernels.model.rotation_a8 import residual_norm_rotate_a8

        self.emit(
            section + "/norm_rotate",
            "norm-rotation-a8-5120",
            lambda: primitive(residual_norm_rotate_a8, CAPACITY, dynamic_rows=True),
            dict(
                X=input,
                R=residual,
                W=self.weights.immutable(weight, cast="float16"),
                Y="Norm",
                RO=output_residual,
                Signs=self.signs[5120],
                Q="A8",
                S="AS",
            ),
        )

    def rotation(
        self, name, k, input, *, swiglu=False, rows=CAPACITY, dynamic=True, head=False, plane_rows=0
    ):
        from kernels.model.rotation_a8 import rotate_activation_a8

        if swiglu:
            from kernels.model.rotation_a8 import rotate_activation_tiles, quantize_rotation_tiles

            with gpu_lock(self.lock):
                selection = self.row_tuner.rotation(k, swiglu, plane_rows=plane_rows)
            first = dict(
                X=input, Signs=self.signs[k], Rotated="RotationValues", Maxima="RotationMaxima"
            )
            second = dict(Rotated="RotationValues", Maxima="RotationMaxima", Q="A8", S="AS")
            for role, fn, args in [
                (name, rotate_activation_tiles, first),
                (name.removesuffix("_rotate") + "_quantize", quantize_rotation_tiles, second),
            ]:
                for suffix, config in self.row_choices(selection, rows):
                    tile = config["block_k"]
                    self.emit(
                        role + suffix,
                        f"rotation-tiles-{fn.__name__}-{k}-{tile}-{swiglu}-planes{plane_rows if fn is rotate_activation_tiles else 0}",
                        lambda fn=fn, tile=tile: primitive(
                            fn,
                            CAPACITY,
                            k,
                            tile,
                            **(
                                dict(swiglu=swiglu, dynamic_rows=True, plane_rows=plane_rows)
                                if fn is rotate_activation_tiles
                                else dict(dynamic_rows=True)
                            ),
                        ),
                        args,
                        rows,
                    )
            return
        self.emit(
            name,
            f"rotation-{k}-{swiglu}-{rows}-{dynamic}",
            lambda: primitive(rotate_activation_a8, rows, k, swiglu=swiglu, dynamic_rows=dynamic),
            dict(
                X=input,
                Signs=self.signs[k],
                Q="HeadA8" if head else "A8",
                S="HeadAS" if head else "AS",
            ),
            rows,
            dynamic=dynamic,
        )

    @staticmethod
    def row_choices(selection, capacity, fixed=()):
        from tools.model.row_selection import choices

        yield from choices(selection, capacity, fixed)

    def attention(self, name, *, q, gate, position, length, output, prefix="", rows=CAPACITY):
        from kernels.model.attention_rows import causal_attention_int8, merge_attention_int8
        from tools.model.row_tuning import ATTENTION_TARGET

        with gpu_lock(self.lock):
            selection = self.row_tuner.attention(self.context)
        partial = dict(
            Partial="AttentionPartial",
            Maximum="AttentionMaximum",
            Denominator="AttentionDenominator",
        )
        inputs = dict(
            Q=q,
            K=prefix + "KPages",
            V=prefix + "VPages",
            KS=prefix + "KPagesScale",
            VS=prefix + "VPagesScale",
            Gate=gate,
            Positions=position,
            Lengths=length,
            Y=output,
            **partial,
        )
        for role, fn, args in [
            (name, causal_attention_int8, inputs),
            (name + "_merge", merge_attention_int8, dict(**partial, Gate=gate, Y=output)),
        ]:
            for suffix, config in self.row_choices(selection, rows):
                bm, bn, group = config["block_m"], config["block_n"], config["head_group"]
                threads, stages = config.get("threads", 128), config.get("stages", 0)
                options = dict(
                    block_m=bm,
                    target_tiles=ATTENTION_TARGET,
                    max_splits=config["max_splits"],
                    head_group=group,
                )
                if fn is causal_attention_int8:
                    options.update(context=self.context, block_n=bn, threads=threads, stages=stages)
                bindings = args
                if fn is causal_attention_int8 and stages:
                    # The compiler names the shared-pointer KV views in its
                    # exported host ABI; both names refer to the same buffers.
                    bindings = args | {f"live_{k.lower()}": args[k] for k in ("K", "V", "KS", "VS")}
                self.emit(
                    role + suffix,
                    f"gqa-rows-{fn.__name__}-{self.context}-{bm}-{bn}-{group}-"
                    f"{config['max_splits']}-{threads}-{stages}",
                    lambda fn=fn, options=options: primitive(fn, CAPACITY, **options),
                    bindings,
                    rows,
                )

    def projection(
        self,
        name,
        weight,
        n,
        k,
        output,
        *,
        stride=None,
        offset=0,
        rows=CAPACITY,
        dynamic=True,
        head=False,
        kind="q4",
    ):
        from tools.model.flash_next.decode_tuning import DecodeTuner
        from tools.model.row_selection import samples

        w = self.weights
        if kind == "q4":
            kind = self.q4_codec
        compressed = kind in ("q4", "e8p_q4")
        suffix = ".packed" if compressed else ".int8"
        data = w.immutable(weight + suffix)
        scale = w.immutable(weight + ".scale")

        target = (
            dict(kind="buffer_slice", name=output, offset=offset * 4 if head else offset * 2)
            if offset
            else output
        )
        bind = (
            dict(
                A="HeadA8" if head else "A8",
                W=data,
                S=scale,
                AS="HeadAS" if head else "AS",
                Out=target,
            )
            if compressed
            else dict(A="A8", Weight=data, WeightScale=scale, TokenScale="AS", Output=target)
        )
        if kind == "e8p_q4":
            bind["Book"] = self.book
        from tools.model.flash_next.decode_policy import SPECIALIZATIONS

        shapes = [(rows, dynamic, name)]
        if dynamic:
            shapes += [(r, False, f"{name}/rows/{r}-{r}") for r in SPECIALIZATIONS if r <= rows]
        for live, symbolic, binding in shapes:
            key = f"{kind}-{live}-{symbolic}-{n}-{k}-{stride}-{head}-pk{w.packed_k if kind == 'e8p_q4' else 0}"

            def build():
                import torch

                actual = dict(
                    W=torch.from_numpy(w.array(weight + suffix).copy()).cuda(),
                    WS=torch.from_numpy(w.array(weight + ".scale").copy()).cuda(),
                )
                if kind == "q4":
                    actual["W"] = actual["W"].view(torch.uint16)
                if kind == "e8p_q4":
                    actual["Book"] = torch.from_numpy(w.extra[self.book].copy()).cuda()
                tuner = DecodeTuner(self.output / "tuning", self.aot.cache.root / "tuning")
                return tuner.compile(
                    kind,
                    live,
                    n,
                    k,
                    "float32" if head else "float16",
                    dynamic=symbolic,
                    weights=actual,
                    weight_identity=hashlib.sha256(
                        (
                            w.buffers[data]["data"]["sha256"]
                            + (w.buffers[self.book]["data"]["sha256"] if kind == "e8p_q4" else "")
                        ).encode()
                    ).hexdigest(),
                    measure_rows=samples(live) if symbolic and live == CAPACITY else None,
                    objective="total_latency"
                    if symbolic and live == CAPACITY
                    else "relative_geomean",
                    output_stride=stride,
                    packed_k=w.packed_k if kind == "e8p_q4" else 0,
                )[0]

            self.emit(binding, key, build, bind, live, dynamic=symbolic)

    def forward(self):
        from kernels.model import qwen35, control, kv_int8
        from kernels.model.gdn_sequence import _gdn_sequence
        from kernels.operators.op07_gdn_ab import gdn_ab_tensorcore
        from kernels.operators.op08_gdn_conv_prep import gdn_conv_prep
        from kernels.operators.op09_gdn_gates import _compile_gates
        from kernels.operators.op17_gdn_gated_norm import _compile_gated_norm
        from kernels.operators.op30_activation_quantization import _compile as a8_compile

        w = self.weights
        self.emit(
            "begin/prepare",
            "prepare",
            lambda: primitive(qwen35.prepare, CAPACITY, 5120, self.context, mrope_input=True),
            dict(
                Step="Step",
                Positions="Positions",
                Length="SeqLength",
                SegmentLength="SegmentLength",
                Last="LastIndex",
                Requests="Requests",
                MRope="MRopePositions",
                Residual="R0",
            ),
        )
        e = "model.language_model.embed_tokens.weight"
        self.emit(
            "begin/embedding",
            "embedding",
            lambda: primitive(qwen35.embedding, CAPACITY, 248320, 5120),
            dict(
                IDs="Input",
                Weight=w.immutable(e + ".int8"),
                Scale=w.immutable(e + ".scale"),
                Output="Hidden",
            ),
        )
        for i, kind in enumerate(self.text["layer_types"]):
            p = f"model.language_model.layers.{i}"
            pre = f"layer{i}/pre"
            private = f"layer{i}/private"
            post = f"layer{i}/post"
            self.norm_rotation(pre, "Hidden", "R0", p + ".input_layernorm.weight", "R1")
            if kind == "linear_attention":
                base = p + ".linear_attn"
                self.projection(pre + "/qkv", base + ".in_proj_qkv.weight", 10240, 5120, "QKV")
                self.projection(pre + "/z", base + ".in_proj_z.weight", 6144, 5120, "Zout")
                ab = w.combined(
                    f"L{i}_AB", [base + ".in_proj_a.weight", base + ".in_proj_b.weight"]
                )
                with gpu_lock(self.lock):
                    selection = self.row_tuner.gdn_ab()
                for suffix, config in self.row_choices(selection, CAPACITY):
                    self.emit(
                        pre + "/ab" + suffix,
                        "gdn-ab-" + json.dumps(config, sort_keys=True),
                        lambda config=config: primitive(
                            gdn_ab_tensorcore,
                            __import__("tilelang.language", fromlist=["dynamic"]).dynamic("m"),
                            **config,
                        ),
                        dict(X="Norm", W_ab=ab, Y="AB"),
                    )
                self.emit(
                    private + "/gates",
                    "gdn-gates",
                    lambda: primitive(_compile_gates, packed_ab=True),
                    dict(
                        A="AB",
                        B="AB",
                        Parameter=w.immutable(base + ".A_log", cast="float32"),
                        DtBias=w.immutable(base + ".dt_bias", cast="float32"),
                        G="g",
                        Beta="Beta",
                    ),
                )
                self.emit(
                    private + "/conv",
                    "gdn-conv",
                    lambda: primitive(gdn_conv_prep, 1, CAPACITY, dynamic_tokens=True),
                    dict(
                        X="QKV",
                        W=w.immutable(base + ".conv1d.weight", cast="float16", shape=[10240, 4]),
                        HI=f"L{i}_History",
                        lengths="SegmentLength",
                        positions="Step",
                        Q="Q",
                        K="K",
                        V="V",
                        HO="HistoryOut",
                        positions_out="PositionOut",
                    ),
                )
                with gpu_lock(self.lock):
                    selection = self.row_tuner.gdn_sequence()
                recurrent_bindings = dict(
                    Q="Q",
                    K="K",
                    V="V",
                    G="g",
                    Beta="Beta",
                    State=f"L{i}_State",
                    Prefix="GdnScratch",
                    Out="Y",
                )
                for suffix, config in self.row_choices(selection, CAPACITY):
                    self.emit(
                        private + "/recurrent" + suffix,
                        "gdn-recurrent-" + str(config["token_unroll"]),
                        lambda config=config: primitive(
                            _gdn_sequence,
                            CAPACITY,
                            in_place=True,
                            value_tile=16,
                            threads=32,
                            prefetch_qk=True,
                            dynamic_rows=True,
                            **config,
                        ),
                        recurrent_bindings,
                    )
                self.emit(
                    private + "/gated_norm",
                    "gdn-gated-norm",
                    lambda: primitive(_compile_gated_norm, None),
                    dict(
                        X="Y",
                        Z="Zout",
                        W=w.immutable(base + ".norm.weight", cast="float16"),
                        Y="MixerIn",
                    ),
                )
            else:
                base = p + ".self_attn"
                for role, n, offset in [("q", 12288, 0), ("k", 1024, 12288), ("v", 1024, 13312)]:
                    self.projection(
                        pre + "/" + role,
                        base + f".{role}_proj.weight",
                        n,
                        5120,
                        "FullX",
                        stride=14336,
                        offset=offset,
                    )
                pages = math.ceil(self.context / 128)
                self.emit(
                    private + "/prepare",
                    f"kv-prepare-{self.context}",
                    lambda: primitive(
                        kv_int8.full_prepare_mrope_int8,
                        pages,
                        self.context,
                        (11, 11, 10),
                        max_position=self.context,
                    ),
                    dict(
                        X="FullX",
                        WQ=w.immutable(base + ".q_norm.weight", cast="float16"),
                        WK=w.immutable(base + ".k_norm.weight", cast="float16"),
                        Cache="Rotary",
                        Req="Requests",
                        Pos="Positions",
                        Pages="Pages",
                        Status="PrepareStatus",
                        MRope="MRopePositions",
                        Q="FullQ",
                        Gate="FullGate",
                        K=f"L{i}_KPages",
                        V=f"L{i}_VPages",
                        KS=f"L{i}_KPagesScale",
                        VS=f"L{i}_VPagesScale",
                    ),
                )
                self.attention(
                    private + "/attention",
                    q="FullQ",
                    gate="FullGate",
                    position="Positions",
                    length="SeqLength",
                    output="MixerIn",
                    prefix=f"L{i}_",
                )
            self.emit(
                post + "/quantize",
                "a8-6144",
                lambda: primitive(a8_compile, 6144, 6144, False, False, 256),
                dict(X="MixerIn", Mask="UnusedA8Mask", Q="A8", S="AS"),
            )
            self.projection(
                post + "/out",
                base + (".out_proj.weight" if kind == "linear_attention" else ".o_proj.weight"),
                5120,
                6144,
                "Mix",
                kind="w8",
            )
            self.norm_rotation(post, "Mix", "R1", p + ".post_attention_layernorm.weight", "R0")
            for role, offset in [("gate", 0), ("up", CAPACITY * 17408)]:
                self.projection(
                    post + "/" + role,
                    p + f".mlp.{role}_proj.weight",
                    17408,
                    5120,
                    "GateUp",
                    stride=17408,
                    offset=offset,
                )
            self.rotation(
                post + "/swiglu_rotate", 17408, "GateUp", swiglu=True, plane_rows=CAPACITY
            )
            self.projection(post + "/down", p + ".mlp.down_proj.weight", 5120, 17408, "Hidden")
            print("BOUND layer", i, flush=True)
        self.emit(
            "end/advance",
            "advance",
            lambda: primitive(control.advance, CAPACITY, dynamic_rows=True),
            dict(Step="Step"),
        )

    def head(self):
        from kernels.operators.op23_final_norm import final_norm
        from kernels.model.greedy import greedy_partials, greedy_merge
        from kernels.model.flash_control import selection

        w = self.weights
        self.emit(
            "gather/norm",
            "final-norm",
            lambda: primitive(final_norm, None, 1),
            dict(
                X="Hidden",
                R="R0",
                I="LastIndex",
                W=w.immutable("model.language_model.norm.weight", cast="float16"),
                Y="HeadHidden",
            ),
        )
        for name, rows, dynamic in [("head", LANES, True), ("head_last", 1, False)]:
            self.rotation(
                name + "/rotate", 5120, "HeadHidden", rows=rows, dynamic=dynamic, head=True
            )
            self.projection(
                name + "/project",
                "lm_head.weight",
                248320,
                5120,
                "HeadLogits",
                rows=rows,
                dynamic=dynamic,
                head=True,
            )
            self.emit(
                name + "/partials",
                f"greedy-partials-{rows}-{dynamic}",
                lambda: primitive(greedy_partials, 248320, rows, dynamic_rows=dynamic),
                dict(
                    Logits="HeadLogits",
                    Values="GreedyValues",
                    Indices="GreedyIndices",
                    Invalid="GreedyInvalid",
                ),
                rows,
                dynamic=dynamic,
            )
            self.emit(
                name + "/merge",
                f"greedy-merge-{rows}-{dynamic}",
                lambda: primitive(greedy_merge, 248320, rows, dynamic_rows=dynamic),
                dict(
                    Values="GreedyValues",
                    Indices="GreedyIndices",
                    Invalid="GreedyInvalid",
                    Output="Pairs",
                ),
                rows,
                dynamic=dynamic,
            )
        self.emit(
            "select/selection",
            "selection",
            lambda: primitive(selection),
            dict(Pair="Pairs", Token="Token", Status="TokenStatus"),
            1,
            dynamic=False,
        )

    def publish(self, engine):
        w = self.weights
        total = w.finish()
        for name in [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "generation_config.json",
            "merges.txt",
            "vocab.json",
        ]:
            if (self.source / name).is_file():
                shutil.copyfile(self.source / name, self.root / name)
        metadata = dict(
            schema_version=2,
            target="sm_87",
            model="Qwen3.8-27B-Q4A8",
            chunk_tokens=CAPACITY,
            max_context=self.context,
            vocab=248320,
            toolchain=self.aot.identity,
            buffers=list(w.buffers.values()),
            reset_buffers=self.reset,
            input="Input",
            token="Token",
            status="TokenStatus",
            logits="Logits",
            position="Step",
            weight_bytes=total,
            weight_parameters=self.state["parameters"],
            weight_scope=f"Original BF16 -> signed-block128 {self.q4_codec} Q4A8; W8 embedding/output projections/MTP exceptions; one packed resident copy; no permanent W8 expansion",
            kv_cache=dict(
                direct_prefill=True, demand_mapping=True, buffers=self.kv, prefill_workspace={}
            ),
            vision=self.vision,
            mtp=self.mtp,
            batch_layout=dict(
                layers=self.text["layer_types"],
                row_strides={},
                profiles={},
                hidden=5120,
                state_columns=self.state_columns,
            ),
        )
        package = dict(
            schema_version=1,
            runtime_abi=1,
            target="sm_87",
            architecture="qwen3_5",
            compute_policy="int8_quality",
            config_signature=config_signature(self.config),
            prefill_profiles=[dict(tokens=CAPACITY, min_tokens=1, kind="qwen_q4_rows")],
            kernels=self.aot.kernels,
            dynamic_batch_kernels=self.aot.dynamic,
            buffer_contracts=[
                {k: v for k, v in b.items() if k != "data"} for b in w.buffers.values()
            ],
            toolchain=self.aot.identity,
            execution=execution_identity(model_library(), engine, "qwen3_5", "int8_quality"),
            plan_kernel_contracts=[k["name"] for k in self.aot.kernels if "/rows/" in k["name"]],
        )
        digest = compact_package(self.root / "cache", package, self.root / "cache/packages")
        data = dict(
            schema_version=1,
            architecture="qwen3_5",
            compute_policy="int8_quality",
            execution_package=digest,
            buffer_scopes=w.scopes,
            frontend_assets={
                name: file_hash(self.root / name)
                for name in ["tokenizer.json", "chat_template.jinja", "generation_config.json"]
            },
            metadata=metadata,
        )
        write_json(self.root / "cache/model.json", data)
        result = subprocess.run(
            [str(engine), "plan-model", str(self.root)], capture_output=True, text=True
        )
        if result.returncode:
            raise ValueError(f"Native plan validation failed: {result.stderr.strip()}")
        plan = json.loads(result.stdout)["manifest"]
        if len(plan["kernels"]) != len(self.aot.kernels):
            raise ValueError("Native execution bindings were lost during publication")
        shutil.rmtree(self.root / "cache/shared-rows")
        write_json(
            self.output / "result.json",
            dict(
                execution_package=digest,
                weight_bytes=total,
                kernels=len(plan["kernels"]),
                programs=len(plan["programs"]),
            ),
        )
        return digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["source", "model-output", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--cache", type=Path, default=ROOT / "artifacts/build-cache/qwen35")
    p.add_argument("--context", type=int, default=262144)
    p.add_argument("--gpu-lock", type=Path, default=ROOT / "artifacts/gpu-experiment.lock")
    p.add_argument(
        "--gpu-lock-held",
        action="store_true",
        help="The operator wrapper already holds the shared GPU lock",
    )
    p.add_argument("--engine", type=Path, default=ROOT / "target/release/orinfer")
    a = p.parse_args()
    if a.source.resolve() == a.model_output.resolve():
        p.error("Model output must differ from quantized checkpoint")
    a.output.mkdir(parents=True, exist_ok=True)
    with atomic_model(a.model_output) as staging:
        b = Builder(
            a.source, staging, a.output, a.cache, a.context, None if a.gpu_lock_held else a.gpu_lock
        )
        digest = b.publish(a.engine.resolve(strict=True))
    print("COMPLETE", digest, flush=True)


if __name__ == "__main__":
    main()
