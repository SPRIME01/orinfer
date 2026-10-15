"""Publish measured local prefill choices and continuous causal/warm segments."""

import argparse
import copy
import json
import os
import re
import shutil
from pathlib import Path

from tools.model.publication import (
    atomic_model,
    clone_model,
    load_model,
    file_hash,
    link_or_copy,
    write_json,
)
from tools.model.flash_next.compiler import Compiler
from tools.model.flash_next.compiler import replace_binding
from tools.model.upgrade_dynamic_batch import contract
from tools.operators.abi import parse_host, validate_parameter_count


def publish(data, package, compiler, tuning):
    report = json.loads((tuning / "results.json").read_text())
    if not report["complete"]:
        raise ValueError("Prefill autotune did not complete")
    original = {k["name"]: k for k in package["kernels"]}
    added, dynamic, removed = [], [], set()

    def emit(old, name, kind, implementation, bindings=None):
        folder = tuning / "aot" / kind / implementation
        assets = {}
        for field, file in [
            ("module", "kernel.cubin"),
            ("source", "kernel.cu"),
            ("host_abi", "host.txt"),
        ]:
            target = compiler.root / "prefill-tuned" / kind / implementation / file
            target.parent.mkdir(parents=True, exist_ok=True)
            link_or_copy(folder / file, target)
            assets[field] = dict(
                file=str(target.relative_to(compiler.root)), sha256=file_hash(target)
            )
        host = parse_host((folder / "host.txt").read_text())[0]
        validate_parameter_count((folder / "kernel.cu").read_text(), host)
        args = bindings if bindings is not None else compiler.pointers(old)
        args = {n: v for n, v in args.items() if n not in ("m", "rows")}
        k = replace_binding(
            dict(old, name=name), host, assets, args, dimensions=dict(m=4096, rows=4096)
        )
        # Candidate functions retain the primitive's pointer ABI and access modes.
        source = (folder / "kernel.cu").read_text()
        prototype = re.search(r"\b" + re.escape(k["symbol"]) + r"\s*\(([^)]*)\)", source)
        k["readonly_arguments"] = [
            "*" not in p or bool(re.search(r"\bconst\b", p)) for p in prototype[1].split(",")
        ]
        added.append(k)
        dynamic.append(contract(k, host, 4096))
        return k

    for old in original.values():
        name = old["name"]
        if not name.startswith("prefill_m4096/"):
            continue
        if name.endswith(("/medium_rows", "/large_rows", "/aligned_rows", "/dispatch_planes")):
            removed.add(name)
            continue
        if "/choice/" in name:
            raise ValueError("Already tuned prefill package")
        roles = compiler.pointers(old)
        kind = None
        if {"A", "Weight", "Output"} == set(roles) - {"m", "rows"}:
            if compiler.buffers[roles["Weight"]["name"]]["shape"] == [512, 2560]:
                kind = "router"
        if old["symbol"] == "hc_prefill_up_mix_kernel":
            kind = "hc"
        if name + "/dispatch_planes" in original:
            kind = "dispatch"
        if kind is None:
            continue
        selected = report["families"][kind]
        default = selected["default"]
        fused = original.get(name + "/dispatch_planes")
        pack = None
        if kind == "dispatch":
            output = roles["D"]["name"]
            pack = next(
                k
                for k in original.values()
                if k["name"].startswith(name.rsplit("/", 1)[0] + "/")
                and k["symbol"] == "pack_exact_a8_kernel"
                and compiler.pointers(k).get("A", {}).get("name") == output
            )
            # A mixed portfolio keeps dispatch+pack as its canonical fallback.
            if len(selected["used"]) > 1:
                default = "separate"
        removed.add(name)
        emit(fused if default == "fused" else old, name, kind, default)
        if kind == "dispatch" and default == "fused":
            removed.add(pack["name"])
        selections = selected["selections"]
        if kind == "dispatch" and len(selected["used"]) > 1 and selected["default"] != "separate":
            # Generate intervals with the required separate canonical ABI.
            selections = intervals(selected["records"], "fused")
        for choice in selections:
            implementation = choice["implementation"]
            if implementation == default:
                continue
            interval = f"{choice['min_rows']}-{choice['max_rows']}/{choice['multiple_of']}"
            emit(
                fused if implementation == "fused" else old,
                name + "/choice/" + interval + "/k0",
                kind,
                implementation,
            )
            if kind == "dispatch":
                marker = copy.deepcopy(pack)
                marker["name"] = pack["name"] + "/choice/" + interval + "/skip"
                added.append(marker)
                launch = copy.deepcopy(
                    next(c for c in package["dynamic_batch_kernels"] if c["name"] == pack["name"])
                )
                launch["name"] = marker["name"]
                dynamic.append(launch)
    for width, source, dest, label in [
        (2560, "DraftM512_Embedding", "DraftRows_Embedding", "embedding"),
        (10240, "DraftM512_Condition", "DraftRows_Condition", "condition"),
    ]:
        from kernels.model.causal_rows import copy as copy_rows

        bindings = dict(
            Source=dict(kind="buffer", name=source), Destination=dict(kind="buffer", name=dest)
        )
        k, c = compiler.emit(
            original["causal_rows/warm/capture"],
            "causal_rows/warm/" + label,
            bindings,
            512,
            build=lambda width=width: copy_rows.__wrapped__(512, width),
            key=f"warm-copy-{width}",
        )
        added.append(k)
        dynamic.append(c)
    package["kernels"] = [k for k in package["kernels"] if k["name"] not in removed] + added
    package["dynamic_batch_kernels"] = [
        k for k in package["dynamic_batch_kernels"] if k["name"] not in removed
    ] + dynamic
    package["plan_kernel_contracts"] = sorted(
        set(package["plan_kernel_contracts"]) - removed | {k["name"] for k in added}
    )
    layout = data["metadata"]["batch_layout"]
    layout.update(
        profiles={"512": "flash_recurrent"},
        small_mixed_shapes=[],
        segment_tokens=512,
        preparation_programs={},
    )
    package["prefill_batch_profiles"] = [dict(tokens=1024, kind="flash_mixed_grouped")]
    spec = data["metadata"]["mtp"]
    spec["warm_plans"] = [
        dict(tokens=512, min_tokens=1, program="mtp_warm", head_program="mtp_head")
    ]
    spec["batch_programs"] = [
        n
        for n in spec["batch_programs"]
        if not n.startswith(
            (
                "flash_segment_m",
                "flash_verify_segment_m",
                "flash_mixed_capture_m",
                "flash_verify_capture_m",
            )
        )
    ]
    return dict(
        removed=len(removed),
        added=len(added),
        tuning={
            k: dict(used=v["used"], default=v["default"]) for k, v in report["families"].items()
        },
    )


def intervals(records, implementation):
    from tools.model.flash_next.prefill_tuning import ranges

    # Reconstruct the tuner selection, including its default, without adding a threshold.
    selection = ranges(records)
    live = [selection["default"]] * 4096
    for c in selection["selections"]:
        for i in range(c["min_rows"], c["max_rows"] + 1):
            if i % c["multiple_of"] == 0:
                live[i - 1] = c["implementation"]
    result = []
    start = None
    for n, value in enumerate(live + [None], 1):
        if value == implementation and start is None:
            start = n
        if value != implementation and start is not None:
            result.append(
                dict(min_rows=start, max_rows=n - 1, multiple_of=1, implementation=implementation)
            )
            start = None
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "model-output", "output", "tuning"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--engine", type=Path)
    a = p.parse_args()
    if a.engine:
        os.environ["ORINFER_ENGINE"] = str(a.engine.resolve(strict=True))
    data, origin, package = load_model(a.model)
    from tools.operators.common import configure

    configure()
    with atomic_model(a.model_output, a.engine) as staging:
        root = clone_model(a.model, staging, origin)
        compiler = Compiler(
            origin, root, a.output, {b["name"]: b for b in data["metadata"]["buffers"]}
        )
        stats = publish(data, package, compiler, a.tuning)
        from tools.model.compact import compact_package

        digest = compact_package(root, package, staging / "cache/packages")
        shutil.rmtree(root)
        data["execution_package"] = digest
        write_json(staging / "cache/model.json", data)
        stats["execution_package"] = digest
    write_json(a.output / "results.json", stats)


if __name__ == "__main__":
    main()
