"""Request-parallel M1 state, wrapping the original TileLang arithmetic."""

import inspect
import math
import re

TABLE = "QwenRowPointers"
SCRATCH = "QwenRowScratch"
LANES = 128


def geometry(selection, row=1):
    key = selection["default"]
    for choice in selection["selections"]:
        if (
            choice["min_rows"] <= row <= choice["max_rows"]
            and row % choice.get("multiple_of", 1) == 0
        ):
            key = choice["implementation"]
    return selection["configs"][int(key)]


def factory(role, context, attention):
    from kernels.model import kv_int8
    from kernels.model.attention_rows import causal_attention_int8, merge_attention_int8
    from kernels.model.gdn_sequence import _gdn_sequence
    from kernels.model.rows import static_rows
    from kernels.operators.op08_gdn_conv_prep import gdn_conv_prep
    from kernels.operators.op09_gdn_gates import _compile_gates
    from kernels.operators.op17_gdn_gated_norm import _compile_gated_norm
    from tools.model.row_tuning import ATTENTION_TARGET

    def invoke(fn, *args, **kwargs):
        return inspect.unwrap(getattr(fn, "func", fn))(*args, **kwargs)

    if role == "gates":
        fn = invoke(_compile_gates, packed_ab=True)
    elif role == "conv":
        fn = invoke(gdn_conv_prep, 1, 1)
    elif role == "recurrent":
        fn = invoke(_gdn_sequence, 1, in_place=True, value_tile=16, threads=32, prefetch_qk=True)
    elif role == "gated_norm":
        fn = invoke(_compile_gated_norm, 1)
    elif role == "prepare":
        fn = invoke(
            kv_int8.full_prepare_mrope_int8,
            (context + 127) // 128,
            context,
            (11, 11, 10),
            max_position=context,
        )
    else:
        options = dict(
            block_m=attention["block_m"],
            target_tiles=ATTENTION_TARGET,
            max_splits=attention["max_splits"],
            head_group=attention["head_group"],
        )
        if role == "attention":
            options.update(
                context=context,
                block_n=attention["block_n"],
                threads=attention.get("threads", 128),
                stages=attention.get("stages", 0),
            )
            fn = invoke(causal_attention_int8, 1, **options)
        elif role == "attention_merge":
            fn = invoke(merge_attention_int8, 1, **options)
        else:
            raise ValueError(f"Unknown Qwen private role: {role}")
    return static_rows(fn)


def build(b, attention=None):
    """Add one dynamic 1..128-request kernel per private computation slot."""
    import tilelang
    from kernels.model.independent_rows import independent_rows, history_commit
    from tools.model.qwen35.build import gpu_lock

    if attention is None:
        with gpu_lock(b.lock):
            attention = geometry(b.row_tuner.attention(b.context))
    w = b.weights
    kernels = {k["name"]: k for k in b.aot.kernels}
    row_roles = set(["QKV", "Zout", "AB", "FullX", "FullQ", "FullGate", "MixerIn"])
    layers = []
    scratch = {}
    columns = {SCRATCH}
    dtypes = dict(f16=2, f32=4, i8=1, u8=1, i32=4)
    for layer, kind in enumerate(b.text["layer_types"]):
        entries = []
        names = (
            ["gates", "conv", "recurrent", "gated_norm"]
            if kind == "linear_attention"
            else ["prepare", "attention", "attention_merge"]
        )
        for role in names:
            old = kernels[f"qwen_rows/layer{layer}/private/{role}"]
            pointers = b.aot.compiler.pointers(old)
            fn = factory(role, b.context, attention)
            modes = {}
            normalized = dict(pointers)
            from tvm import tirx as tir

            used = set()

            def visit(node):
                if isinstance(node, (tir.BufferLoad, tir.BufferStore)):
                    used.add(node.buffer.data)
                if isinstance(node, tir.Var):
                    used.add(node)

            tir.stmt_functor.post_order_visit(fn.body, visit)
            for parameter, buf in fn.buffer_map.items():
                if buf.data not in used:
                    continue
                arg = pointers.get(buf.name)
                if arg is None:
                    # Compressed KV prefetch aliases use the same pointer.
                    arg = pointers.get("live_" + buf.name.lower())
                if arg is None:
                    raise ValueError(f"Missing private ABI {role}/{buf.name}")
                name = arg["name"]
                normalized[buf.name] = arg
                storage = w.buffers[name]
                if storage.get("data") is not None:
                    mode, identity = "weight", name
                elif name in row_roles:
                    mode, identity = "row", name
                elif w.scopes[name] == "sequence":
                    mode, identity = "private", re.sub(r"^L\d+_", "", name)
                    columns.add(identity)
                else:
                    mode, identity = "private", SCRATCH
                    size = math.prod(int(n) for n in buf.shape) * dtypes[storage["dtype"]]
                    scratch[name] = max(scratch.get(name, 0), size)
                modes[buf.name] = (mode, identity)
            entries.append((role, fn, modes, normalized))
        layers.append(entries)
    offsets = {}
    cursor = 0
    for name in sorted(scratch, key=lambda name: (name != "HistoryOut", name)):
        offsets[name] = cursor
        cursor += (scratch[name] + 255) // 256 * 256
    w.fixed(SCRATCH, "u8", [cursor], "sequence")
    columns = sorted(columns)
    b.state_columns = columns
    w.fixed(TABLE, "u64", [LANES * len(layers) * len(columns)])
    column_map = {name: i for i, name in enumerate(columns)}
    print("PRIVATE ROWS", len(columns), "columns", cursor, "scratch bytes/request", flush=True)
    for layer, entries in enumerate(layers):
        for role, fn, modes, pointers in entries:
            args_offsets = {
                name: offsets[arg["name"]]
                for name, arg in pointers.items()
                if arg["name"] in offsets
            }
            wrapped, inputs = independent_rows(
                fn,
                LANES,
                modes,
                column_map,
                row_stride=len(layers) * len(columns),
                dynamic=True,
                offsets=args_offsets,
            )
            bindings = {}
            for name in inputs:
                if name is None:
                    continue
                if name == "__rows__":
                    continue
                bindings[name] = pointers[name]
            bindings["Pointers"] = dict(
                kind="buffer_slice", name=TABLE, offset=layer * len(columns) * 8
            )
            key = f"private-rows-{role}-{attention if role.startswith('attention') else ''}"

            def compile_wrapped(wrapped=wrapped, role=role):
                return tilelang.compile(
                    wrapped,
                    out_idx=[],
                    execution_backend="nvrtc",
                    target={"kind": "cuda", "arch": "sm_87"},
                    pass_configs={
                        "tl.disable_thread_storage_sync": True,
                        "tl.disable_shared_memory_reuse": True,
                    }
                    if role == "recurrent"
                    else {},
                )

            b.emit(f"state/layer{layer}/{role}", key, compile_wrapped, bindings, LANES)
            if role == "conv":
                if offsets["HistoryOut"] != 0:
                    raise ValueError("History must begin private scratch")
                b.emit(
                    f"state/layer{layer}/history",
                    "private-history",
                    lambda: history_commit(
                        len(columns),
                        column_map[SCRATCH],
                        column_map["History"],
                        row_stride=len(layers) * len(columns),
                    ),
                    dict(
                        Pointers=dict(
                            kind="buffer_slice", name=TABLE, offset=layer * len(columns) * 8
                        )
                    ),
                    LANES,
                )
