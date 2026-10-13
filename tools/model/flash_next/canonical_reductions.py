"""Rebuild Flash QSA, recurrent and HC routes with canonical reductions."""

import argparse
import copy
import math
import re
import shutil
from pathlib import Path

from tools.model.compact import compact_package
from tools.model.flash_next.optimize_decode import pointer_bindings, replace_binding
from tools.model.publication import (
    AssetReader,
    atomic_model,
    clone_model,
    file_hash,
    link_or_copy,
    load_model,
    write_json,
)
from tools.operators.abi import parse_host, validate_parameter_count

ATTENTION = {"Den", "K", "KS", "Max", "Out", "Position", "Query", "Selected", "V", "VS"}
MERGE = {"Den", "Gate", "Max", "Out", "Output"}
SPLITS = 8


def private_layout(data, kernels):
    """Recover the published address-table arena from its original interfaces."""
    buffers = {b["name"]: b for b in data["metadata"]["buffers"]}
    scratch = {}
    for k in kernels:
        if not k["name"].startswith("flash_batch_m2/layer") or "/private/" not in k["name"]:
            continue
        for a in k["args"]:
            name = a.get("name", "")
            if name.startswith("BatchM2_") or data["buffer_scopes"].get(name) != "workspace":
                continue
            b = buffers[name]
            size = math.prod(b["shape"]) * {"f16": 2, "f32": 4, "i32": 4}[b["dtype"]]
            scratch[name] = size
    offsets, cursor = {}, 0
    for name, size in sorted(scratch.items()):
        cursor = (cursor + 255) // 256 * 256
        offsets[name] = cursor
        cursor += size
    if (cursor + 255) // 256 * 256 != math.prod(buffers["FlashPrivateScratch"]["shape"]):
        raise ValueError("Published private QSA arena differs from its interfaces")
    return offsets


def upgrade(model, destination, report, split_size=256, gdn_only=False, hc_only=False):
    import tilelang
    from kernels.model import qsa, qsa_attention
    from kernels.model.gdn_sequence import gdn_sequence
    from kernels.model.independent_rows import independent_rows
    from kernels.model.hyperconnection import hc_projection
    from kernels.model.hc_prefill import hc_projection_prefill
    from tools.model.package import model_library
    from tools.operators.common import configure, export_kernel

    configure()
    data, origin, package = load_model(model)
    if data["architecture"] != "flash_next":
        raise ValueError("Expected native Flash Next")
    root = clone_model(model, destination, origin)
    reader = AssetReader(origin)
    buffers = {b["name"]: b for b in data["metadata"]["buffers"]}
    hosts, exports, changed, capacities = {}, {}, [], {}
    original = copy.deepcopy(package["kernels"])
    offsets = None

    def host(k):
        key = k["host_abi"]["sha256"]
        if key not in hosts:
            hosts[key] = parse_host(reader.read(k["host_abi"]).decode())[0]
        return hosts[key]

    def roles(k):
        return {a["value"].removesuffix(".data_ptr()") for a in host(k)["ordered_arguments"]}

    def exported(key, factory):
        if key not in exports:
            folder = report / "aot" / key
            export_kernel(factory(), folder)
            assets = {}
            for field, filename in (
                ("module", "kernel.cubin"),
                ("source", "kernel.cu"),
                ("host_abi", "host.txt"),
            ):
                target = root / "canonical-reductions" / key / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                link_or_copy(folder / filename, target)
                assets[field] = dict(file=str(target.relative_to(root)), sha256=file_hash(target))
            h = parse_host((folder / "host.txt").read_text())[0]
            validate_parameter_count((folder / "kernel.cu").read_text(), h)
            exports[key] = h, assets
        return exports[key]

    def reserve(binding, elements):
        b = buffers[binding["name"]]
        if b["dtype"] != "f32" or b.get("data") or binding.get("offset", 0) != 0:
            raise ValueError("Expected whole mutable FP32 QSA partial")
        capacities[b["name"]] = max(capacities.get(b["name"], 0), elements)

    for position, k in enumerate(original):
        signature = roles(k)
        recurrent = {"Q", "K", "V", "G", "Beta", "State", "Out"}
        if not gdn_only and {"A", "C", "W"} <= signature:
            args = pointer_bindings(k, host(k), allow_scalars=True)
            weight = buffers[args["W"]["name"]]
            if not args["W"]["name"].endswith(
                (
                    "hc_attn_down.weight_value",
                    "hc_ffn_down.weight_value",
                    "output_hc_down.weight_value",
                )
            ):
                continue
            n, width = weight["shape"]
            if weight["dtype"] != "bf16" or n not in (256, 320) or width != 10240:
                raise ValueError("Unsupported HC down geometry")
            program = k["name"].split("/")[0]
            match = re.search(r"_m(\d+)", program)
            if match is None:
                raise ValueError("Missing HC row extent")
            rows = max(16, int(match[1]))
            dynamic = "m" in signature
            cached = buffers[args["A"]["name"]]["dtype"] == "bf16"
            out_dtype = {"bf16": "bfloat16", "f16": "float16"}[buffers[args["C"]["name"]]["dtype"]]
            silu = bool(re.search(r"\b(?:__)?expf\(", reader.read(k["source"]).decode()))
            bm = 64 if rows >= 64 else 16
            bn, gm = (128, 4) if rows >= 256 else (64, 0)
            key = f"hc-{rows}-{n}-{int(cached)}-{int(silu)}-{out_dtype}-{int(dynamic)}"

            def factory(
                rows=rows,
                n=n,
                bm=bm,
                bn=bn,
                gm=gm,
                cached=cached,
                out_dtype=out_dtype,
                silu=silu,
                dynamic=dynamic,
            ):
                options = dict(
                    block_m=bm,
                    dtype="float16",
                    block_n=bn,
                    silu=silu,
                    group_m=gm,
                    dynamic_rows=dynamic,
                    reduction_groups=4,
                )
                if cached:
                    return hc_projection_prefill(rows, n, 10240, output_dtype=out_dtype, **options)
                return hc_projection(rows, n, 10240, **options)

            h, assets = exported(key, factory)
            package["kernels"][position] = replace_binding(
                k, h, assets, args, dimensions={"m": rows}
            )
            changed.append(k["name"])
        elif not hc_only and signature in (recurrent, recurrent | {"Prefix"}):
            args = pointer_bindings(k, host(k))
            program = k["name"].split("/")[0]
            match = re.search(r"_m(\d+)", program)
            rows = (
                1
                if program == "decode" or program.startswith("flash_batch_")
                else (int(match[1]) if match else None)
            )
            if rows is None:
                raise ValueError(f"Unknown recurrent extent: {k['name']}")
            compact = "Prefix" in args
            prefetch = rows >= 256 and not compact

            def factory(rows=rows, compact=compact, prefetch=prefetch):
                return gdn_sequence(
                    rows,
                    compact=compact,
                    in_place=not compact,
                    threads=32 if prefetch else 128,
                    value_tile=16 if prefetch else 32,
                    prefetch_qk=prefetch,
                )

            h, assets = exported(f"gdn-{rows}-{int(compact)}", factory)
            package["kernels"][position] = replace_binding(k, h, assets, args)
            changed.append(k["name"])
        elif not hc_only and not gdn_only and (signature == ATTENTION or signature == MERGE):
            args = pointer_bindings(k, host(k))
            rows = k["grid"][0]
            if signature == ATTENTION:
                for arg, width in [("Max", 24), ("Den", 24), ("Out", 24 * 256)]:
                    reserve(args[arg], rows * SPLITS * width)
                key = f"attention-{rows}-{split_size}"

                def factory(rows=rows):
                    return qsa_attention.sparse_attention(
                        rows,
                        data["metadata"]["max_context"],
                        SPLITS,
                        packed=True,
                        split_size=split_size,
                    )
            else:
                key = f"merge-{rows}"

                def factory(rows=rows):
                    return qsa.sparse_merge(rows, SPLITS)

            h, assets = exported(key, factory)
            package["kernels"][position] = replace_binding(k, h, assets, args)
            changed.append(k["name"])
        elif (
            not hc_only
            and not gdn_only
            and signature == {"Pointers", "m"}
            and k["grid"][1:] == [2, 8]
        ):
            if not k["name"].startswith("flash_private_m"):
                raise ValueError("Unknown indirect QSA route")
            if offsets is None:
                offsets = private_layout(data, original)
            template = next(
                t
                for t in original
                if t["name"].startswith("flash_batch_m2/layer3/private/") and roles(t) == ATTENTION
            )
            direct = pointer_bindings(template, host(template))
            columns = {
                name: i for i, name in enumerate(data["metadata"]["batch_layout"]["state_columns"])
            }
            bindings, relative = {}, {}
            for name, arg in direct.items():
                value = arg["name"]
                if value in offsets:
                    relative[name] = offsets[value]
                    value = "FlashPrivateScratch"
                bindings[name] = ("private", re.sub(r"^State_\d+_", "State_", value))

            def factory():
                fn, inputs = independent_rows(
                    qsa_attention.sparse_attention.__wrapped__(
                        1,
                        data["metadata"]["max_context"],
                        SPLITS,
                        packed=True,
                        split_size=split_size,
                    ),
                    128,
                    bindings,
                    columns,
                    row_stride=48 * len(columns),
                    dynamic=True,
                    offsets=relative,
                )
                if inputs != [None, "__rows__"]:
                    raise ValueError("Indirect QSA lost its private pointer-table ABI")
                return tilelang.compile(
                    fn,
                    out_idx=[],
                    execution_backend="nvrtc",
                    target={"kind": "cuda", "arch": "sm_87"},
                )

            h, assets = exported(f"independent-{split_size}", factory)
            pointers = pointer_bindings(k, host(k), allow_scalars=True)
            package["kernels"][position] = replace_binding(
                k, h, assets, pointers, dimensions={"m": k["grid"][0]}
            )
            changed.append(k["name"])
    if not changed or (not gdn_only and not hc_only and not capacities):
        raise ValueError("Missing reduction execution routes")
    extra = 0
    for name, elements in capacities.items():
        b = buffers[name]
        old = math.prod(b["shape"])
        if elements > old:
            b["shape"] = [elements]
            extra += (elements - old) * 4
            contract = next(c for c in package["buffer_contracts"] if c["name"] == name)
            contract["shape"] = [elements]
    library = root / package["execution"]["library"]["file"]
    library.unlink()
    shutil.copyfile(model_library(), library)
    package["execution"]["library"]["sha256"] = file_hash(library)
    digest = compact_package(root, package, destination / "cache/packages")
    shutil.rmtree(root)
    data["execution_package"] = digest
    write_json(destination / "cache/model.json", data)
    write_json(
        report / "results.json",
        dict(
            changed=changed,
            compiled_variants=len(exports),
            split_size=split_size,
            extra_workspace_bytes=extra,
            weights_unchanged=True,
            execution_package=digest,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--gdn-only", action="store_true")
    modes.add_argument("--hc-only", action="store_true")
    args = parser.parse_args()
    with atomic_model(args.model_output) as staging:
        upgrade(
            args.model.resolve(strict=True),
            staging,
            args.output,
            256,
            args.gdn_only,
            args.hc_only,
        )
    print("CANONICAL REDUCTIONS READY", args.model_output, flush=True)


if __name__ == "__main__":
    main()
