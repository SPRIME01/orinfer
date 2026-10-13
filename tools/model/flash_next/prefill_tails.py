"""Publish exact small tails and logical 129..512-token prefill.

Only causal kernels are specialized to the actual length. Padding rows never
advance persistent state, and the head selects the last real residual row.
"""

import argparse
import copy
import math
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


def upgrade(model, destination, report, logical=False):
    from kernels.model import flash_next as fn, qsa, qsa_attention, flash_control
    from kernels.model.gdn_sequence import gdn_sequence
    from kernels.model import moe
    from kernels.model.control import advance
    from kernels.model.ple import ple_conv, ple_history
    from kernels.operators.op08_gdn_conv_prep import gdn_conv_prep
    from kernels.vision.features import overlay
    from tools.model.package import model_library
    from tools.operators.common import configure, export_kernel

    configure()
    data, origin, package = load_model(model)
    if data["architecture"] != "flash_next":
        raise ValueError("Expected Flash Next")
    if not logical and any(2 <= p["tokens"] < 16 for p in package["prefill_profiles"]):
        raise ValueError("Exact tail profiles are already present")
    if logical and any(p.get("min_tokens", 0) for p in package["prefill_profiles"]):
        raise ValueError("Logical prefill is already present")
    root = clone_model(model, destination, origin)
    reader = AssetReader(origin)
    buffers = {b["name"]: b for b in data["metadata"]["buffers"]}
    exports, hosts, additions = {}, {}, []
    capacity = data["metadata"]["max_context"]
    physical = 512 if logical else 16
    templates = [
        k
        for k in package["kernels"]
        if k["name"].startswith(
            tuple(f"{p}_m{physical}/" for p in ("prefill", "head", "mtp_capture"))
        )
    ]
    head_name, head_source = "LogicalPrefillHeadM512", None
    if logical:
        head = dict(
            name=head_name,
            dtype="f16",
            shape=[1, 2560],
            layout="native_contiguous",
            alignment=256,
            access="read_write",
        )
        data["metadata"]["buffers"].append(head)
        package["buffer_contracts"].append(copy.deepcopy(head))
        data["buffer_scopes"][head_name] = "workspace"

    def host(k):
        key = k["host_abi"]["sha256"]
        if key not in hosts:
            hosts[key] = parse_host(reader.read(k["host_abi"]).decode())[0]
        return hosts[key]

    def compiled(key, factory):
        if key not in exports:
            folder = report / "aot" / key
            export_kernel(factory(), folder)
            assets = {}
            for field, filename in [
                ("module", "kernel.cubin"),
                ("source", "kernel.cu"),
                ("host_abi", "host.txt"),
            ]:
                target = root / "prefill-tails" / key / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                link_or_copy(folder / filename, target)
                assets[field] = dict(file=str(target.relative_to(root)), sha256=file_hash(target))
            h = parse_host((folder / "host.txt").read_text())[0]
            validate_parameter_count((folder / "kernel.cu").read_text(), h)
            exports[key] = h, assets
        return exports[key]

    for rows in [512] if logical else range(2, 16):
        for suffix in [] if logical else ["Embedding", "Ple", "HistoryOut"]:
            source = buffers[f"M16_{suffix}"]
            name = f"M{rows}_{suffix}"
            if name in buffers:
                buffers[name]["shape"] = source["shape"].copy()
                next(b for b in package["buffer_contracts"] if b["name"] == name)["shape"] = source[
                    "shape"
                ].copy()
                continue
            b = copy.deepcopy(source)
            b["name"] = name
            data["metadata"]["buffers"].append(b)
            buffers[name] = b
            package["buffer_contracts"].append(copy.deepcopy(b))
            data["buffer_scopes"][name] = data["buffer_scopes"][source["name"]]
        counters = {}
        for old in templates:
            k = copy.deepcopy(old)
            k["name"] = old["name"].replace(f"_m{physical}/", f"_m{rows}/")
            if not logical and old["name"].startswith("mtp_capture_m16/"):
                k["name"] = k["name"].replace("mtp_capture_", "mtp_prefill_capture_", 1)
            for a in k["args"]:
                if a.get("name", "").startswith("M16_"):
                    a["name"] = a["name"].replace("M16_", f"M{rows}_", 1)
                if (
                    not logical
                    and old["name"].startswith("head_m16/")
                    and a["kind"] == "buffer_slice"
                    and a.get("offset", 0)
                ):
                    if a["offset"] % 15:
                        raise ValueError("Head slice is not the last 16-row token")
                    a["offset"] = a["offset"] // 15 * (rows - 1)
            args = pointer_bindings(k, host(old), allow_scalars=True)
            signature = set(args)
            if logical:
                args["Length"] = dict(kind="buffer", name="Length")
                if old["name"].startswith("head_m512/"):
                    if "X" in args and args["X"].get("offset", 0):
                        source = args["X"]
                        if source["offset"] != 511 * 5120:
                            raise ValueError("Unexpected last hidden row offset")
                        head_source = dict(kind="buffer", name=source["name"])
                        for a in k["args"]:
                            if a == source:
                                a.clear()
                                a.update(kind="buffer", name=head_name)
                    additions.append(k)
                    continue
            factory, key = None, None
            if old["name"].startswith(f"mtp_capture_m{physical}/"):
                width = (
                    10240 if logical else math.prod(buffers[args["Hidden"]["name"]]["shape"]) // 16
                )
                ring = math.prod(buffers[args["Ring"]["name"]]["shape"]) // width
                key, factory = "capture", lambda: flash_control.capture(rows, width, ring, logical)
            elif old["name"].startswith(f"prefill_m{physical}/"):
                layer = old["name"].split("/")[1]
                if layer == "end" and signature == {"Step"}:
                    key, factory = "advance", lambda: advance(rows, logical)
                elif signature == {"Embedding", "Features", "Index", "Position"}:
                    features = math.prod(buffers[args["Features"]["name"]]["shape"]) // 2560
                    key, factory = (
                        "overlay",
                        lambda: overlay(rows, 2560, capacity, features, logical_rows=logical),
                    )
                elif signature == {"Embedding", "Residual"}:
                    key, factory = (
                        "initialize",
                        lambda: fn.hc_initialize(rows, logical_rows=logical),
                    )
                elif {"HI", "HO", "lengths", "positions_out"} <= signature:
                    key, factory = (
                        "conv",
                        lambda: gdn_conv_prep(
                            B=1,
                            tokens=rows,
                            normalize_round_fp16=False,
                            conv_product_round_fp16=False,
                            weight_dtype="float32",
                        ),
                    )
                elif signature == {"Q", "K", "V", "G", "Beta", "State", "Out"}:
                    key, factory = (
                        "recurrent",
                        lambda: gdn_sequence(
                            rows,
                            in_place=True,
                            threads=32 if logical else 128,
                            value_tile=16 if logical else 32,
                            prefetch_qk=logical,
                            logical_rows=logical,
                        ),
                    )
                elif signature == {"Gated", "Normed", "Output", "State", "Weight"}:
                    key, factory = "ple-conv", lambda: ple_conv(rows, 10240)
                elif signature == {"NextState", "Normed"}:
                    args["State"] = args["NextState"]
                    key, factory = (
                        "ple-history",
                        lambda: ple_history(rows, 10240, logical_rows=logical),
                    )
                elif "QGate" in signature and "Query" in signature:
                    key, factory = (
                        "prepare",
                        lambda: fn.qsa_prepare(
                            rows,
                            capacity,
                            is_neox_style=True,
                            staged=True,
                            mrope="Coordinates" in signature,
                            logical_rows=logical,
                        ),
                    )
                elif {"Key", "Value", "KS", "VS"} <= signature:
                    key, factory = "store", lambda: qsa.kv_store(rows, capacity, logical)
                elif signature == {"Coordinates", "Position", "QK", "Query", "Weight"}:
                    key, factory = "index-query", lambda: qsa.index_query(rows, capacity, logical)
                elif {"Cache", "Pending", "Position", "QK", "Weight"} <= signature:
                    key, factory = (
                        "compress",
                        lambda: qsa.index_compress(
                            rows, capacity, mrope="Coordinates" in signature, logical_rows=logical
                        ),
                    )
                elif signature == {"Pending", "Position", "QK"}:
                    key, factory = "pending", lambda: qsa.index_pending(rows, logical)
                elif signature == {"Cache", "Position", "Query", "Scores"}:
                    key, factory = "scores", lambda: qsa.index_scores(rows, capacity, logical)
                elif {"Hist", "Position", "Scores"} <= signature:
                    turn = counters.get((layer, "hist"), 0)
                    counters[layer, "hist"] = turn + 1
                    shift = (24, 16, 8, 0)[turn]
                    key, factory = (
                        f"hist-{shift}",
                        lambda: qsa.radix_histogram(rows, capacity, shift, logical_rows=logical),
                    )
                elif {"Hist", "Position", "Prefix", "Remaining"} <= signature:
                    turn = counters.get((layer, "choose"), 0)
                    counters[layer, "choose"] = turn + 1
                    shift = (24, 16, 8, 0)[turn]
                    key, factory = (
                        f"choose-{shift}",
                        lambda: qsa.radix_choose(rows, capacity, shift, logical_rows=logical),
                    )
                elif signature == {"Counts", "Position", "Prefix", "Scores"}:
                    key, factory = (
                        "counts",
                        lambda: qsa.selection_counts(rows, capacity, logical_rows=logical),
                    )
                elif signature == {"Counts", "Greater", "Offsets", "Position", "Selected"}:
                    key, factory = (
                        "offsets",
                        lambda: qsa.selection_offsets(rows, capacity, logical_rows=logical),
                    )
                elif signature == {
                    "Greater",
                    "Offsets",
                    "Position",
                    "Prefix",
                    "Scores",
                    "Selected",
                }:
                    key, factory = (
                        "scatter",
                        lambda: qsa.selection_scatter(rows, capacity, logical_rows=logical),
                    )
                elif {"Query", "Max", "Den", "Selected"} <= signature:
                    key, factory = (
                        "attention",
                        lambda: qsa_attention.sparse_attention(
                            rows,
                            capacity,
                            packed=True,
                            split_size=256,
                            logical_rows=logical,
                        ),
                    )
                elif signature == {"Den", "Gate", "Max", "Out", "Output"}:
                    key, factory = "merge", lambda: qsa.sparse_merge(rows, 8, logical)
                elif logical and signature == {"IDs", "Partial", "Slots"}:
                    key, factory = (
                        "route-local",
                        lambda: moe.expert_histogram_local(rows, logical_rows=True),
                    )
                elif logical and signature == {"IDs", "Offsets", "Slots"}:
                    key, factory = (
                        "route-finish",
                        lambda: moe.expert_histogram_finish(rows, logical_rows=True),
                    )
                elif logical and {"A", "RelativeSlot", "SlotMap", "DS"} <= signature:
                    key, factory = (
                        "dispatch",
                        lambda: moe.expert_dispatch(
                            rows, 2560, scale_group=2560, logical_rows=True
                        ),
                    )
                elif logical and signature == {
                    "Expert",
                    "Output",
                    "Prob",
                    "Shared",
                    "SharedGate",
                    "SlotMap",
                }:
                    key, factory = (
                        "combine",
                        lambda: moe.moe_combine(rows, 2560, rows * 10, logical_rows=True),
                    )
            if factory is not None:
                h, assets = compiled(f"{key}-{rows}", factory)
                k = replace_binding(k, h, assets, args)
            additions.append(k)
        if not logical:
            package["prefill_profiles"].append(dict(tokens=rows, kind="flash_recurrent"))
        if not logical and data["metadata"].get("mtp"):
            data["metadata"]["mtp"]["capture_plans"].append(
                dict(tokens=rows, program=f"mtp_prefill_capture_m{rows}")
            )
    if logical:
        if head_source is None:
            raise ValueError("Missing final head input")
        h, assets = compiled("head-gather-512", lambda: flash_control.last_hidden(512, 2560, True))
        additions.append(
            replace_binding(
                dict(name="head_m512/body/logical_gather"),
                h,
                assets,
                dict(
                    Hidden=head_source,
                    Condition=dict(kind="buffer", name=head_name),
                    Length=dict(kind="buffer", name="Length"),
                ),
            )
        )
        updates = {k["name"]: k for k in additions}
        package["kernels"] = [updates.pop(k["name"], k) for k in package["kernels"]]
        package["kernels"].extend(updates.values())
        next(p for p in package["prefill_profiles"] if p["tokens"] == 512)["min_tokens"] = 129
    else:
        package["kernels"].extend(additions)
    package["plan_kernel_contracts"] = sorted(
        set(package.get("plan_kernel_contracts", [])) | {k["name"] for k in additions}
    )
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
            compiled_variants=len(exports),
            exact_tails=[] if logical else list(range(2, 16)),
            logical_rows=[129, 512] if logical else None,
            weights_unchanged=True,
            execution_package=digest,
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--model-output", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--compile-cache", type=Path)
    p.add_argument(
        "--logical-512",
        action="store_true",
        help="Publish logical 129..512 rows using a 512-row compute capacity",
    )
    args = p.parse_args()
    if args.compile_cache:
        from tools.model.publication import seed_compile_cache

        seed_compile_cache(args.compile_cache, args.output / "cache/0.1.15")
    with atomic_model(args.model_output) as staging:
        upgrade(args.model.resolve(strict=True), staging, args.output, args.logical_512)
    print(
        "LOGICAL PREFILL READY" if args.logical_512 else "EXACT PREFILL TAILS READY",
        args.model_output,
        flush=True,
    )


if __name__ == "__main__":
    main()
