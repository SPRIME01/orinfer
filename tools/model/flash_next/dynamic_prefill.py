"""Publish one compact-row Flash prefill template and local algorithm variants.

The native Rust recipe owns layer order and resumable stages. This offline pass
replaces implementations at its existing operand slots; input lengths only
change host launch scalars and compact tensor strides.
"""

import argparse
import copy
import math
import shutil
from pathlib import Path

from tools.model.compact import compact_package
from tools.model.flash_next.compiler import pointer_bindings, replace_binding
from tools.model.publication import (
    AssetReader,
    atomic_model,
    clone_model,
    file_hash,
    link_or_copy,
    load_model,
    write_json,
)
from tools.model.upgrade_dynamic_batch import contract
from tools.operators.abi import evaluate, parse_host, validate_parameter_count

CAPACITY = 4096


def upgrade(model, destination, output):
    import tilelang
    from kernels.model import control, flash_control, flash_next as f, hc_prefill as hc
    from kernels.model import hyperconnection as h, moe, ple, qsa, qsa_attention
    from kernels.model.e8p_prefill import dispatch_prefill_a8, e8p_prefill_grouped, split_prefill_a8
    from kernels.model.gdn_sequence import _gdn_sequence as gdn_sequence
    from kernels.model.int8_projection import int8_projection
    from kernels.model.q4_projection import q4_projection
    from kernels.model.rotation_a8 import rotate_activation_a8
    from kernels.model.rows import explicit_rows
    from kernels.operators.op08_gdn_conv_prep import gdn_conv_prep
    from kernels.vision.features import overlay
    from tools.model.package import model_library
    from tools.operators.common import configure, export_kernel

    configure()
    data, origin, package = load_model(model)
    if data["architecture"] != "flash_next":
        raise ValueError("Expected native Flash Next")
    root = clone_model(model, destination, origin)
    reader = AssetReader(origin)
    buffers = {b["name"]: b for b in data["metadata"]["buffers"]}
    hosts, compiled, additions, contracts = {}, {}, [], []
    templates = [
        k
        for k in package["kernels"]
        if k["name"].startswith(("prefill_m512/", "head_m512/", "mtp_capture_m512/"))
    ]
    if not templates:
        raise ValueError("Missing validated compact medium-prefill implementation")

    def resize(name, shape):
        b = buffers[name]
        if b.get("data") or data["buffer_scopes"][name] != "workspace":
            raise ValueError("Dynamic prefill may resize only mutable workspace")
        b["shape"] = shape
        next(x for x in package["buffer_contracts"] if x["name"] == name)["shape"] = shape.copy()

    for name, b in buffers.items():
        if name.startswith(("HC_M512_", "ComputeQ4_R512_")):
            if b["shape"][0] != 512:
                raise ValueError("Unexpected medium prefill arena")
            resize(name, [CAPACITY, *b["shape"][1:]])
        if name.startswith("E8P_prefill_m512_"):
            resize(name, [CAPACITY * 10 if n == 5120 else n for n in b["shape"]])

    partial_name = "HC_Prefill_StreamPartials"
    if partial_name in buffers:
        raise ValueError("HC partial workspace already exists")
    partial = dict(
        name=partial_name,
        dtype="f32",
        shape=[4, CAPACITY, 320],
        layout="native_contiguous",
        alignment=256,
        access="read_write",
    )
    buffers[partial_name] = partial
    data["metadata"]["buffers"].append(partial)
    data["buffer_scopes"][partial_name] = "workspace"
    package["buffer_contracts"].append(copy.deepcopy(partial))

    def host(k):
        digest = k["host_abi"]["sha256"]
        if digest not in hosts:
            hosts[digest] = parse_host(reader.read(k["host_abi"]).decode())[0]
        return hosts[digest]

    def shape(a):
        return buffers[a["name"]]["shape"]

    def dtype(a):
        return {"f16": "float16", "f32": "float32", "bf16": "bfloat16"}[buffers[a["name"]]["dtype"]]

    def emit(k, key, build, args, multiplier=1, options=None):
        if key not in compiled:
            fn = explicit_rows(build())
            generated = tilelang.compile(
                fn,
                out_idx=[],
                execution_backend="nvrtc",
                target={"kind": "cuda", "arch": "sm_87"},
                pass_configs=options or {},
            )
            folder = output / "aot" / key
            export_kernel(generated, folder)
            assets = {}
            for field, filename in [
                ("module", "kernel.cubin"),
                ("source", "kernel.cu"),
                ("host_abi", "host.txt"),
            ]:
                target = root / "dynamic-prefill" / key / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                link_or_copy(folder / filename, target)
                assets[field] = dict(file=str(target.relative_to(root)), sha256=file_hash(target))
            abi = parse_host((folder / "host.txt").read_text())[0]
            validate_parameter_count((folder / "kernel.cu").read_text(), abi)
            compiled[key] = abi, assets
        abi, assets = compiled[key]
        k = replace_binding(
            k,
            abi,
            assets,
            args,
            dimensions=dict(m=CAPACITY * multiplier, rows=CAPACITY * multiplier),
        )
        contracts.append(contract(k, abi, CAPACITY, multiplier))
        additions.append(k)
        print("DYNAMIC PREFILL BIND", k["name"], key, flush=True)

    for old in templates:
        k = copy.deepcopy(old)
        k["name"] = k["name"].replace("_m512/", "_m4096/")
        args = pointer_bindings(old, host(old), allow_scalars=True)
        args.pop("Length", None)
        roles = set(args) - {"rows", "m"}
        rows, opts = CAPACITY, dict(dynamic_rows=True)
        private = k["name"].startswith("prefill_m4096/")
        key, fn, multiplier = None, None, 1
        if (
            not private
            and not k["name"].endswith("logical_gather")
            and k["name"].split("/")[-1] in {"k4", "k5", "k6", "k7", "k8", "q4_rotation"}
        ):
            additions.append(k)
            continue
        if roles == {"Embedding", "Residual"}:
            key, fn = "initialize", (lambda: f.hc_initialize.__wrapped__(rows, **opts))
        elif roles == {"Embedding", "Features", "Index", "Position"}:
            features = math.prod(shape(args["Features"])) // 2560
            key, fn = (
                "overlay",
                (
                    lambda: overlay.__wrapped__(
                        rows, 2560, data["metadata"]["max_context"], features, **opts
                    )
                ),
            )
        elif roles == {"InvRms", "NormedBF", "Residual", "Weight"}:
            key, fn = "hc-norm", (lambda: hc.hc_norm_prefill.__wrapped__(rows, 2560, **opts))
        elif {"Block", "Inject", "InvRms", "NormedBF", "Residual", "Weight"} <= roles:
            key, fn = (
                "hc-combine-norm",
                (lambda: hc.hc_combine_norm_prefill.__wrapped__(rows, 2560, **opts)),
            )
        elif roles == {"Normed", "Residual", "Weight"}:
            key, fn = (
                "hc-head-norm",
                (lambda: h.hc_norm.__wrapped__(rows, 2560, 4, dtype="float16", **opts)),
            )
        elif roles == {"A", "C", "W"}:
            n, width = shape(args["W"])
            cached = buffers[args["A"]["name"]]["dtype"] == "bf16"
            factory = hc.hc_projection_prefill if cached else h.hc_projection
            bm, bn = (64, 128) if n in (256, 320) else (16, 16) if n == 4 else (128, 128)
            settings = dict(
                block_m=bm,
                block_n=bn,
                group_m=4 if n in (256, 320) else 0,
                dtype="float16",
                silu=n in (256, 320),
                reduction_groups=4 if n in (256, 320) and width == 10240 else 1,
                **opts,
            )
            if cached:
                settings["output_dtype"] = dtype(args["C"])
            key = f"hc-proj-{n}-{width}-{int(cached)}-{dtype(args['C'])}"

            def fn():
                return factory.__wrapped__(rows, n, width, **settings)
        elif roles == {"A", "W", "Residual", "Weight", "InvRms", "Out"}:
            key, fn = (
                "hc-up-mix",
                (
                    lambda: hc.hc_up_mix_prefill.__wrapped__(
                        rows, 2560, 320, padded_inputs=True, **opts
                    )
                ),
            )
        elif roles == {"Mixed", "Normed", "Up"}:
            key, fn = "hc-mix", (lambda: h.hc_mix.__wrapped__(rows, 2560, 4, "float16", **opts))
        elif roles == {"Block", "Inject", "Output", "Residual"}:
            key, fn = (
                "hc-combine",
                (lambda: h.hc_combine.__wrapped__(rows, 2560, 4, "float16", **opts)),
            )
        elif roles == {"A", "AS", "Out", "S", "W"}:
            n, packed = shape(args["W"])
            key, fn = (
                f"q4-{n}-{packed}",
                (
                    lambda: q4_projection.__wrapped__(
                        rows, n, packed * 4, BN=128, BK=256, stages=1, block_m=128, **opts
                    )
                ),
            )
        elif roles == {"A", "Output", "TokenScale", "Weight", "WeightScale"}:
            n, width = shape(args["Weight"])
            dt = dtype(args["Output"])
            key, fn = (
                f"w8-{n}-{width}-{dt}",
                (
                    lambda: int8_projection.__wrapped__(
                        rows, n, width, dt, block_m=128, block_n=128, block_k=128, **opts
                    )
                ),
            )
        elif roles == {"A", "Output", "Weight"}:
            n, width = shape(args["Weight"])
            dt = dtype(args["Output"])
            key = f"bf16-{n}-{width}-{dt}"

            def fn():
                return (
                    f.scalar_bf16_projection.__wrapped__(rows, width, **opts)
                    if n == 1
                    else f.dense_projection.__wrapped__(rows, n, width, "bfloat16", dt, **opts)
                )
        elif roles == {"Q", "S", "Signs", "X"}:
            width = math.prod(shape(args["Signs"]))
            multiplier = 10 if width == 640 else 1
            key, fn = (
                f"rotate-{width}-{multiplier}",
                (
                    lambda: rotate_activation_a8.__wrapped__(
                        rows * multiplier, width, swiglu=multiplier == 10, **opts
                    )
                ),
            )
        elif roles == {"Q", "S", "X"} or roles == {"A", "B", "Beta", "DtBias", "G", "Parameter"}:
            # These existing interfaces already export an exact row scalar.
            abi = host(old)
            dims = dict(m=rows, rows=rows)
            for value, parameter in zip(k["args"], abi["ordered_arguments"]):
                if value["kind"] == "i32":
                    value["value"] = evaluate(parameter["value"], dims)
            k["grid"] = [evaluate(abi["launch_expressions"]["gridDim" + a], dims) for a in "XYZ"]
            contracts.append(contract(k, abi, CAPACITY))
            additions.append(k)
            continue
        elif {"HI", "HO", "lengths", "positions_out"} <= roles:
            key, fn = (
                "gdn-conv",
                (
                    lambda: gdn_conv_prep.__wrapped__(
                        B=1,
                        tokens=rows,
                        normalize_round_fp16=False,
                        conv_product_round_fp16=False,
                        weight_dtype="float32",
                        dynamic_tokens=True,
                    )
                ),
            )
        elif roles == {"Q", "K", "V", "G", "Beta", "State", "Out"}:
            key, fn = (
                "gdn-recurrent",
                (
                    lambda: gdn_sequence(
                        rows,
                        in_place=True,
                        threads=32,
                        value_tile=16,
                        prefetch_qk=True,
                        token_unroll=4,
                        **opts,
                    )
                ),
            )
        elif roles == {"Weight", "X", "Y", "Z"}:
            key, fn = "gdn-norm", (lambda: f.gdn_sigmoid_norm.__wrapped__(rows, **opts))
        elif roles == {"IDs", "Logits", "Prob"}:
            key, fn = "router", (lambda: moe.router_topk.__wrapped__(rows, 512, 10, **opts))
        elif roles == {"IDs", "Partial", "Slots"}:
            key, fn = "route-local", (lambda: moe.expert_histogram_local.__wrapped__(rows, **opts))
        elif roles == {"Counts", "Offsets", "Partial"}:
            key, fn = (
                "route-offsets",
                (lambda: moe.expert_histogram_offsets.__wrapped__(rows, **opts)),
            )
        elif roles == {"IDs", "Offsets", "Slots"}:
            key, fn = (
                "route-finish",
                (lambda: moe.expert_histogram_finish.__wrapped__(rows, **opts)),
            )
        elif roles == {"Counts", "RowOffsets", "TileCount", "TileOffsets"}:
            additions.append(k)
            continue
        elif roles == {"Counts", "TileExpert", "TileOffsets", "TileRow"}:
            for role in ["TileExpert", "TileRow"]:
                resize(args[role]["name"], [(rows * 10 + 15) // 16 + 512])
            key, fn = "route-tiles", (lambda: moe.expert_tiles.__wrapped__(rows, dynamic_rows=True))
        elif {"A", "D", "DS", "IDs", "RelativeSlot", "RowOffsets", "Scale", "SlotMap"} == roles:
            key, fn = (
                "dispatch",
                (lambda: moe.expert_dispatch.__wrapped__(rows, 2560, scale_group=2560, **opts)),
            )
        elif roles == {"A", "High", "Low", "Sum"}:
            # Activation planes use assignment rows, rather than token rows.
            width = 2560 if "K2560" in args["High"]["name"] else 640
            multiplier = 10
            key, fn = (
                f"split-{width}",
                (lambda: split_prefill_a8.__wrapped__(rows * 10, width, **opts)),
            )
        elif {
            "High",
            "Low",
            "Sum",
            "Book",
            "P",
            "Counts",
            "Offsets",
            "Out",
            "AS",
            "WS",
            "TC",
            "TE",
            "TR",
        } == roles:
            e, groups, n, _ = shape(args["P"])
            tiles = (rows * 10 + 15) // 16 + 512
            multiplier = 10
            key = f"expert-{n}-{groups}"

            def fn():
                return e8p_prefill_grouped.__wrapped__(
                    rows * 10, e, tiles, n, groups * 64, sparse_rows=groups * 64 == 2560, **opts
                )
        elif roles == {"Gate", "Output", "Up"}:
            key, fn = "shared-swiglu", (lambda: f.swiglu.__wrapped__(rows, 640, **opts))
        elif roles == {"Expert", "Output", "Prob", "Shared", "SharedGate", "SlotMap"}:
            key, fn = (
                "moe-combine",
                (lambda: moe.moe_combine.__wrapped__(rows, 2560, rows * 10, **opts)),
            )
        elif roles == {"Gated", "Key", "Query", "Value"}:
            key, fn = "ple-gate", (lambda: ple.ple_gate.__wrapped__(rows, 2560, 4, **opts))
        elif roles == {"Gated", "Normed", "Output", "State", "Weight"}:
            key, fn = "ple-conv", (lambda: ple.ple_conv.__wrapped__(rows, 10240, **opts))
        elif roles in ({"NextState", "Normed"}, {"NextState", "Normed", "State"}):
            args.setdefault("State", args["NextState"])
            key, fn = "ple-history", (lambda: ple.ple_history.__wrapped__(rows, 10240, **opts))
        elif roles == {"Output", "Update", "X"}:
            key, fn = "ple-add", (lambda: f.residual_add.__wrapped__(rows, 10240, **opts))
        elif {"QGate", "Query", "KCache", "VCache"} <= roles:
            key, fn = (
                "qsa-prepare",
                (
                    lambda: f.qsa_prepare.__wrapped__(
                        rows,
                        data["metadata"]["max_context"],
                        staged=True,
                        is_neox_style=True,
                        mrope=True,
                        **opts,
                    )
                ),
            )
        elif roles == {"K", "KS", "Key", "Position", "V", "VS", "Value"}:
            key, fn = (
                "qsa-store",
                (lambda: qsa.kv_store.__wrapped__(rows, data["metadata"]["max_context"], **opts)),
            )
        elif roles == {"Coordinates", "Position", "QK", "Query", "Weight"}:
            key, fn = (
                "qsa-index-query",
                (
                    lambda: qsa.index_query.__wrapped__(
                        rows, data["metadata"]["max_context"], **opts
                    )
                ),
            )
        elif roles == {"Cache", "Coordinates", "Pending", "Position", "QK", "Weight"}:
            key, fn = (
                "qsa-index-compress",
                (
                    lambda: qsa.index_compress.__wrapped__(
                        rows, data["metadata"]["max_context"], mrope=True, **opts
                    )
                ),
            )
        elif roles == {"Pending", "Position", "QK"}:
            key, fn = "qsa-pending", (lambda: qsa.index_pending.__wrapped__(rows, **opts))
        elif roles == {"Cache", "Position", "Query", "Scores"}:
            key, fn = (
                "qsa-index-scores",
                (
                    lambda: qsa.index_scores.__wrapped__(
                        rows, data["metadata"]["max_context"], **opts
                    )
                ),
            )
        elif "Hist" in roles:
            slot = int(k["name"].rsplit("/k", 1)[1])
            shift = (24, 16, 8, 0)[(slot - 16) // 2]
            if "Scores" in roles:
                key, fn = (
                    f"qsa-hist-{shift}",
                    (
                        lambda: qsa.radix_histogram.__wrapped__(
                            rows, data["metadata"]["max_context"], shift, **opts
                        )
                    ),
                )
            else:
                key, fn = (
                    f"qsa-choose-{shift}",
                    (
                        lambda: qsa.radix_choose.__wrapped__(
                            rows, data["metadata"]["max_context"], shift, **opts
                        )
                    ),
                )
        elif roles == {"Counts", "Position", "Prefix", "Scores"}:
            key, fn = (
                "qsa-counts",
                (
                    lambda: qsa.selection_counts.__wrapped__(
                        rows, data["metadata"]["max_context"], **opts
                    )
                ),
            )
        elif roles == {"Counts", "Greater", "Offsets", "Position", "Selected"}:
            key, fn = (
                "qsa-offsets",
                (
                    lambda: qsa.selection_offsets.__wrapped__(
                        rows, data["metadata"]["max_context"], **opts
                    )
                ),
            )
        elif roles == {"Greater", "Offsets", "Position", "Prefix", "Scores", "Selected"}:
            key, fn = (
                "qsa-scatter",
                (
                    lambda: qsa.selection_scatter.__wrapped__(
                        rows, data["metadata"]["max_context"], **opts
                    )
                ),
            )
        elif {"Query", "Max", "Den", "Selected"} <= roles:
            key, fn = (
                "qsa-attention",
                (
                    lambda: qsa_attention.sparse_attention.__wrapped__(
                        rows, data["metadata"]["max_context"], packed=True, skip_empty=True, **opts
                    )
                ),
            )
        elif roles == {"Den", "Gate", "Max", "Out", "Output"}:
            args["Position"] = dict(kind="buffer", name="Position")
            key, fn = (
                "qsa-merge",
                (lambda: qsa.sparse_merge.__wrapped__(rows, split_size=256, **opts)),
            )
        elif roles == {"Step"}:
            key, fn = "advance", (lambda: control.advance.__wrapped__(rows, **opts))
        elif roles == {"Hidden", "Condition"}:
            key, fn = (
                "head-gather",
                (lambda: flash_control.last_hidden.__wrapped__(rows, 2560, **opts)),
            )
        elif roles == {"Hidden", "Position", "Ring"}:
            ring, width = shape(args["Ring"])
            key, fn = (
                f"capture-{ring}",
                (lambda: flash_control.capture.__wrapped__(rows, width, ring, **opts)),
            )
        else:
            raise ValueError(
                f"Unsupported dynamic prefill operand interface: {k['name']}: {sorted(roles)}"
            )
        options = (
            {
                "tl.disable_thread_storage_sync": True,
                "tl.disable_shared_memory_reuse": True,
                "tl.if_stmt_binding_inline_replayable_binds": False,
            }
            if key.startswith("expert-")
            else None
        )
        if key == "gdn-recurrent":
            options = {
                "tl.disable_thread_storage_sync": True,
                "tl.disable_shared_memory_reuse": True,
            }
        emit(k, key, fn, args, multiplier, options)
        if key == "dispatch":
            planes = {
                role: dict(kind="buffer", name=f"E8P_prefill_m512_K2560_{role}")
                for role in ("Low", "High", "Sum")
            }
            emit(
                dict(k, name=k["name"] + "/dispatch_planes"),
                "dispatch-planes",
                lambda: dispatch_prefill_a8.__wrapped__(rows, dynamic_rows=True),
                {
                    role: args[role]
                    for role in ("A", "Scale", "IDs", "RelativeSlot", "RowOffsets", "DS", "SlotMap")
                }
                | planes,
            )
        if key == "hc-proj-320-10240-1-bfloat16":
            partial_argument = dict(kind="buffer", name=partial_name)
            emit(
                dict(k, name=k["name"] + "/hc_partials"),
                "hc-stream-partials",
                lambda: hc.hc_projection_partial.__wrapped__(rows, 320, 10240, **opts),
                dict(A=args["A"], W=args["W"], P=partial_argument),
            )
            emit(
                dict(k, name=k["name"] + "/hc_finish"),
                "hc-stream-finish",
                lambda: hc.hc_projection_finish.__wrapped__(rows, 320, **opts),
                dict(C=args["C"], P=partial_argument),
            )
        if key == "bf16-512-2560-float32":
            for suffix, bm, bn in [("medium_rows", 64, 128), ("large_rows", 32, 64)]:
                variant = dict(k, name=k["name"] + "/" + suffix)
                emit(
                    variant,
                    key + "-" + suffix,
                    lambda bm=bm, bn=bn: f.dense_projection.__wrapped__(
                        rows,
                        512,
                        2560,
                        "bfloat16",
                        "float32",
                        dynamic_rows=True,
                        block_m=bm,
                        block_n=bn,
                    ),
                    args,
                )
        if key == "hc-up-mix":
            variant = dict(k, name=k["name"] + "/aligned_rows")
            emit(
                variant,
                "hc-up-mix-aligned",
                lambda: hc.hc_up_mix_prefill.__wrapped__(
                    rows, 2560, 320, dynamic_rows=True, aligned_rows=True
                ),
                args,
            )

    removed = {
        k["name"]
        for k in package["kernels"]
        if k["name"].startswith(("prefill_m", "head_m", "mtp_prefill_capture_m", "mtp_capture_m"))
        and not k["name"].startswith(("head_m1/", "mtp_capture_m1/"))
        and k["name"].split("/")[0] not in {f"mtp_capture_m{n}" for n in range(2, 9)}
    }
    package["kernels"] = [k for k in package["kernels"] if k["name"] not in removed] + additions
    package["dynamic_batch_kernels"] = [
        c for c in package.get("dynamic_batch_kernels", []) if c["name"] not in removed
    ] + contracts
    package["plan_kernel_contracts"] = sorted(
        set(package.get("plan_kernel_contracts", [])) - removed | {k["name"] for k in additions}
    )
    package["prefill_profiles"] = [
        dict(tokens=CAPACITY, min_tokens=1, kind="flash_dynamic_recurrent")
    ]
    if data["metadata"].get("mtp"):
        data["metadata"]["mtp"]["capture_plans"] = [
            p for p in data["metadata"]["mtp"]["capture_plans"] if p["tokens"] in (1, CAPACITY)
        ]
    path = root / "lib/model.so"
    path.unlink()
    shutil.copyfile(model_library(), path)
    package["execution"]["library"]["sha256"] = file_hash(path)
    digest = compact_package(root, package, destination / "cache/packages")
    shutil.rmtree(root)
    data["execution_package"] = digest
    write_json(destination / "cache/model.json", data)
    write_json(
        output / "results.json",
        dict(
            complete=True,
            capacity=CAPACITY,
            definitions=len(compiled),
            symbolic_bindings=len(contracts),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--compile-cache", type=Path)
    args = parser.parse_args()
    if args.compile_cache:
        from tools.model.publication import seed_compile_cache

        seed_compile_cache(args.compile_cache, args.output / "cache/0.1.15")
    with atomic_model(args.model_output) as staging:
        upgrade(args.model.resolve(), staging, args.output)


if __name__ == "__main__":
    main()
