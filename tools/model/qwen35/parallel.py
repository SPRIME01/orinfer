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


def factory(role, context, attention, text=None, recurrent=None):
    from kernels.model import kv_int8
    from kernels.model.attention_rows import causal_attention_int8, merge_attention_int8
    from kernels.model.gdn_sequence import _gdn_sequence
    from kernels.model.rows import static_rows
    from kernels.operators.op08_gdn_conv_prep import gdn_conv_prep
    from kernels.operators.op09_gdn_gates import _compile_gates
    from kernels.operators.op17_gdn_gated_norm import _compile_gated_norm
    from tools.model.row_tuning import ATTENTION_TARGET

    text = text or {}
    hk, hv = text.get("linear_num_key_heads", 16), text.get("linear_num_value_heads", 48)
    qh, kvh = text.get("num_attention_heads", 24), text.get("num_key_value_heads", 4)

    def invoke(fn, *args, **kwargs):
        return inspect.unwrap(getattr(fn, "func", fn))(*args, **kwargs)

    if role == "gates":
        fn = invoke(_compile_gates, packed_ab=True, heads=hv)
    elif role == "conv":
        # A request-owned decode CTA has exactly one time row. The compact
        # layout preserves the prefill reduction tree and caches the old raw
        # tail before updating this head's history in place.
        fn = invoke(
            gdn_conv_prep,
            1,
            1,
            compact_decode=True,
            inplace_history=True,
            key_heads=hk,
            value_heads=hv,
        )
    elif role == "recurrent" and (recurrent or {}).get("deferred"):
        from kernels.model.gdn_deferred import recurrent as logged_recurrent

        fn = logged_recurrent(
            recurrent["capacity"],
            key_heads=hk,
            value_heads=hv,
            value_tile=recurrent["value_tile"],
            threads=recurrent["threads"],
        )
    elif role == "recurrent":
        fn = invoke(
            _gdn_sequence,
            1,
            in_place=True,
            key_heads=hk,
            value_heads=hv,
            **(recurrent or dict(value_tile=16, threads=32, prefetch_qk=True)),
        )
    elif role == "gated_norm":
        fn = invoke(_compile_gated_norm, 1, heads=hv)
    elif role == "prepare":
        fn = invoke(
            kv_int8.full_prepare_mrope_int8,
            (context + 127) // 128,
            context,
            (11, 11, 10),
            max_position=context,
            query_heads=qh,
            kv_heads=kvh,
        )
    else:
        options = dict(
            block_m=attention["block_m"],
            query_heads=qh,
            kv_heads=kvh,
            target_tiles=ATTENTION_TARGET,
            max_splits=attention["max_splits"],
            head_group=attention["head_group"],
            parallel_instances=True,
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


def build(b, attention=None, recurrent=None):
    """Bind dynamic private slots, including an optional whole log pipeline.

    Changing log capacity while a request is live requires materialization;
    one package therefore uses one log geometry for its complete row domain.
    """
    import tilelang
    from kernels.model.independent_rows import independent_rows
    from tools.model.qwen35.build import gpu_lock

    if attention is None:
        with gpu_lock(b.lock):
            attention = geometry(b.row_tuner.attention(b.context))
    if recurrent is None:
        with gpu_lock(b.lock):
            recurrent = b.row_tuner.private_gdn()
    w = b.weights
    recurrent_choices = list(b.row_choices(recurrent, LANES))
    deferred_configs = [c for _, c in recurrent_choices if c.get("deferred")]
    deferred = bool(deferred_configs)
    if deferred:
        if len(recurrent_choices) != 1:
            raise ValueError("Deferred GDN requires one request-state geometry")
        from kernels.model import gdn_deferred
        from tools.model.compile import primitive

        config = deferred_configs[0]
        capacity = config["capacity"]
        hk, hv = b.text["linear_num_key_heads"], b.text["linear_num_value_heads"]
        gdn_deferred.validate_geometry(capacity, hk, hv, config["value_tile"], config["threads"])
        for layer, kind in enumerate(b.text["layer_types"]):
            if kind != "linear_attention":
                continue
            bindings = dict(State=f"L{layer}_State")
            for argument, suffix, dtype, shape in [
                ("LogK", "GdnLogKey", "f16", [capacity, hk, 128]),
                ("LogD", "GdnLogDelta", "f32", [capacity, hv, 128]),
                ("LogG", "GdnLogGate", "f32", [capacity, hv]),
                ("Count", "GdnLogCount", "i32", [hv, 128]),
            ]:
                name = f"L{layer}_{suffix}"
                w.fixed(name, dtype, shape, "sequence")
                b.reset.append(name)
                bindings[argument] = name
            b.emit(
                f"layer{layer}/private/recurrent_materialize",
                f"gdn-materialize-{hk}-{hv}-{capacity}",
                lambda: primitive(gdn_deferred.materialize, capacity, key_heads=hk, value_heads=hv),
                bindings,
                dynamic=False,
            )
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
        for role, suffix, config in [
            (role, suffix, config)
            for role in names
            for suffix, config in (
                b.row_choices(recurrent, LANES) if role == "recurrent" else [("", None)]
            )
        ]:
            old = kernels[f"qwen_rows/layer{layer}/private/{role}"]
            pointers = b.aot.compiler.pointers(old)
            if config and config.get("deferred"):
                pointers.update(
                    {
                        key: dict(kind="buffer", name=f"L{layer}_{value}")
                        for key, value in dict(
                            LogK="GdnLogKey",
                            LogD="GdnLogDelta",
                            LogG="GdnLogGate",
                            Count="GdnLogCount",
                        ).items()
                    }
                )
            if role == "conv":
                pointers["History"] = pointers["HI"]
            fn = factory(role, b.context, attention, b.text, config)
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
            entries.append((role, suffix, config, fn, modes, normalized))
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
        for role, suffix, config, fn, modes, pointers in entries:
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
                batch_parameters=("independent_instances",)
                if role in ("attention", "attention_merge")
                else (),
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
            key = f"private-rows-{role}-{b.h}-{b.hv}-{b.qh}-{attention if role.startswith('attention') else config}"

            def compile_wrapped(wrapped=wrapped, role=role, config=config):
                return tilelang.compile(
                    wrapped,
                    out_idx=[],
                    execution_backend="nvrtc",
                    target={"kind": "cuda", "arch": "sm_87"},
                    pass_configs={
                        "tl.disable_thread_storage_sync": True,
                        "tl.disable_shared_memory_reuse": True,
                    }
                    if role == "recurrent" and config.get("prefetch_qk")
                    else {},
                )

            b.emit(f"state/layer{layer}/{role}{suffix}", key, compile_wrapped, bindings, LANES)
