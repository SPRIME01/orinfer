"""Owner-parallel causal kernels with device lengths and shared row arenas."""

import hashlib
import inspect
import json
import math
import re

from kernels.model.independent_rows import independent_rows
from kernels.model.rows import explicit_scalar
from tools.model.qwen35.parallel import LANES, geometry

TABLE = "QwenVerifyPointers"
OFFSET = "QwenVerifyRowOffset"
SCRATCH = "QwenRowScratch"


def invoke(factory, *args, **kwargs):
    return inspect.unwrap(getattr(factory, "func", factory))(*args, **kwargs)


def symbolic_rows(function):
    from tvm import tirx as tir

    variables = set()
    for buffer in function.buffer_map.values():
        for expression in [*buffer.shape, *buffer.strides]:
            tir.stmt_functor.post_order_visit(
                expression,
                lambda node: (
                    variables.add(node)
                    if isinstance(node, tir.Var) and node.name in ("m", "rows", "tokens")
                    else None
                ),
            )
    for variable in variables:
        function = explicit_scalar(function, variable)
    return function, variables


def factories(b, attention, conv):
    from kernels.model import flash_mtp, speculative_factors, gdn_deferred
    from kernels.model.gdn_sequence import _gdn_sequence
    from kernels.model.kv_int8 import full_prepare_mrope_int8
    from kernels.model.attention_rows import causal_attention_int8, merge_attention_int8
    from kernels.operators.op08_gdn_conv_prep import gdn_conv_prep
    from kernels.operators.op09_gdn_gates import _compile_gates
    from kernels.operators.op17_gdn_gated_norm import _compile_gated_norm

    gqa = dict(
        block_m=attention["block_m"],
        target_tiles=16,
        max_splits=attention["max_splits"],
        query_heads=b.qh,
        kv_heads=b.kvh,
        head_group=attention["head_group"],
    )
    common = dict(key_heads=b.hk, value_heads=b.hv)
    out = {
        "gates": lambda: invoke(_compile_gates, packed_ab=True, heads=b.hv),
        "history": lambda: invoke(flash_mtp.history_prefix, 8, b.qkv, 3, dynamic_rows=True),
        "conv": lambda: invoke(gdn_conv_prep, 1, 8, dynamic_tokens=True, **conv, **common),
        "recurrent": lambda: invoke(
            _gdn_sequence, 8, value_tile=16, threads=32, compact=True, dynamic_rows=True, **common
        ),
        "factors": lambda: invoke(
            speculative_factors.save_gdn_factors, 8, dynamic_rows=True, **common
        ),
        "gated_norm": lambda: invoke(_compile_gated_norm, None, heads=b.hv),
        "prepare": lambda: invoke(
            full_prepare_mrope_int8,
            (b.context + 127) // 128,
            b.context,
            (11, 11, 10),
            max_position=b.context,
            query_heads=b.qh,
            kv_heads=b.kvh,
        ),
        "attention": lambda: invoke(
            causal_attention_int8,
            8,
            context=b.context,
            block_n=attention["block_n"],
            threads=attention.get("threads", 128),
            stages=attention.get("stages", 0),
            **gqa,
        ),
        "attention_merge": lambda: invoke(merge_attention_int8, 8, **gqa),
    }
    log = next((v for n, v in b.weights.buffers.items() if n.endswith("_GdnLogGate")), None)
    if log:
        out["recurrent"] = lambda: gdn_deferred.verification(log["shape"][0], **common)
    return out


def build(b, *, attention=None, conv=None):
    import tilelang
    from tvm import tirx as tir

    selection = attention or b.row_tuner.attention(b.context)
    configs, choices = {}, []
    for width in range(2, 9):
        config = geometry(selection, width)
        key = json.dumps(config, sort_keys=True)
        name = f"attention{len(configs)}" if key not in configs else configs[key][0]
        configs[key] = (name, config)
        if choices and choices[-1]["program"] == name:
            choices[-1]["max_tokens"] = width
        else:
            choices.append(dict(min_tokens=width, max_tokens=width, program=name))
    attention = geometry(selection)
    conv = conv or geometry(b.row_tuner.gdn_convolution())
    functions = factories(b, attention, conv)
    b.weights.fixed(OFFSET, "i32", [1], "sequence")
    original = {k["name"]: k for k in b.aot.kernels}
    columns, scratch, entries = {OFFSET, "SegmentLength", "AcceptedInputs", SCRATCH}, {}, []
    row_roles = {"QKV", "Zout", "AB", "FullX", "FullQ", "FullGate", "MixerIn"}
    for layer, kind in enumerate(b.text["layer_types"]):
        roles = (
            [
                "gates",
                "history",
                "conv",
                *(["recurrent_materialize"] if "recurrent_materialize" in functions else []),
                "recurrent",
                "factors",
                "gated_norm",
            ]
            if kind == "linear_attention"
            else ["prepare"]
        )
        work = [(role, role, functions[role]) for role in roles]
        if kind != "linear_attention":
            for name, config in configs.values():
                selected = factories(b, config, conv)
                work.extend(
                    (f"{name}/{role}", role, selected[role])
                    for role in ("attention", "attention_merge")
                )
        for emitted, role, factory in work:
            verify = role in ("history", "recurrent", "factors")
            name = f"qwen_rows/{'verify/' if verify else ''}layer{layer}/{'' if verify else 'private/'}{role}"
            fn, variables = symbolic_rows(factory())
            pointers = b.aot.compiler.pointers(original[name])
            if role == "recurrent":
                pointers.update(
                    {
                        arg: dict(kind="buffer", name=f"L{layer}_{target}")
                        for arg, target in dict(
                            LogK="GdnLogKey",
                            LogD="GdnLogDelta",
                            LogG="GdnLogGate",
                            Count="GdnLogCount",
                        ).items()
                    }
                )
            used = set()
            tir.stmt_functor.post_order_visit(
                fn.body,
                lambda n: (
                    used.add(n.buffer.data)
                    if isinstance(n, (tir.BufferLoad, tir.BufferStore))
                    else used.add(n)
                    if isinstance(n, tir.Var)
                    else None
                ),
            )
            modes = {}
            for buf in fn.buffer_map.values():
                if buf.data not in used:
                    continue
                arg = pointers.get(buf.name) or pointers.get("live_" + buf.name.lower())
                if arg is None:
                    raise ValueError(f"Missing causal pointer {role}/{buf.name}")
                pointers[buf.name] = arg
                target = arg["name"]
                storage = b.weights.buffers[target]
                if storage.get("data") is not None:
                    modes[buf.name] = ("weight", target)
                elif target in row_roles:
                    modes[buf.name] = ("row", target)
                elif b.weights.scopes[target] == "sequence":
                    identity = re.sub(r"^L\d+_", "", target)
                    columns.add(identity)
                    modes[buf.name] = ("private", identity)
                else:
                    modes[buf.name] = ("private", SCRATCH)
                    substitute = {var: 8 for var in variables}
                    shape = tir.stmt_functor.substitute(math.prod(buf.shape), substitute)
                    size = (
                        int(tilelang.tvm.arith.Analyzer().simplify(shape))
                        * dict(float16=2, float32=4, int32=4, uint32=4, int8=1, uint8=1)[buf.dtype]
                    )
                    scratch[target] = max(scratch.get(target, 0), size)
            entries.append((layer, emitted, fn, variables, modes, pointers))
    cursor = 0
    offsets = {}
    for name, size in sorted(scratch.items()):
        offsets[name] = cursor
        cursor += (size + 255) // 256 * 256
    b.weights.buffers[SCRATCH]["shape"][0] = max(b.weights.buffers[SCRATCH]["shape"][0], cursor)
    columns = sorted(columns)
    column_map = {name: i for i, name in enumerate(columns)}
    layers = len(b.text["layer_types"])
    b.weights.fixed(TABLE, "u64", [LANES * layers * len(columns)])
    b.mtp["verification_table"] = dict(
        buffer=TABLE,
        columns=columns,
        row_offset=OFFSET,
        deferred=any(n.endswith("_GdnLogCount") for n in b.weights.buffers),
        attention=choices,
    )
    for layer, role, fn, variables, modes, pointers in entries:
        fn, inputs = independent_rows(
            fn,
            LANES,
            modes,
            column_map,
            row_stride=layers * len(columns),
            dynamic=True,
            offsets={
                name: offsets[arg["name"]]
                for name, arg in pointers.items()
                if arg["name"] in offsets
            },
            scalar_columns={v.name: ("SegmentLength", 8) for v in variables},
            row_offset=OFFSET,
            row_capacity=4096,
            batch_parameters=("independent_instances",)
            if role in ("attention", "attention_merge")
            else (),
        )
        bindings = {name: pointers[name] for name in inputs if name not in (None, "__rows__")}
        bindings["Pointers"] = dict(
            kind="buffer_slice", name=TABLE, offset=layer * len(columns) * 8
        )

        def compile_fn(fn=fn):
            return tilelang.compile(
                fn, out_idx=[], execution_backend="nvrtc", target={"kind": "cuda", "arch": "sm_87"}
            )

        key = "causal-rows-" + role + "-" + hashlib.sha256(str(fn).encode()).hexdigest()
        b.emit(f"verify_state/layer{layer}/{role}", key, compile_fn, bindings, LANES)
