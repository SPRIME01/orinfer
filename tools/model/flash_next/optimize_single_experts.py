"""Recompile single-token expert projections over the same compressed banks."""

import argparse
import shutil
from pathlib import Path

from tools.model.compact import compact_package
from tools.model.flash_next.optimize_decode import pointer_bindings, replace_binding
from tools.model.publication import (
    AssetReader,
    atomic_model,
    clone_model,
    load_model,
    file_hash,
    link_or_copy,
    write_json,
)
from tools.operators.abi import parse_host, validate_parameter_count


def upgrade(model, destination, report, engine=None):
    from kernels.model.e8p_single_decode import e8p_single_decode_grouped
    from tools.operators.common import configure, export_kernel

    configure()
    data, origin, package = load_model(model)
    if data["architecture"] != "flash_next":
        raise ValueError("Expected Flash Next")
    operator = clone_model(model, destination, origin)
    reader = AssetReader(origin)
    buffers = {b["name"]: b for b in data["metadata"]["buffers"]}
    compiled = {}
    changed = []
    for pos, k in enumerate(package["kernels"]):
        if not k["name"].startswith("decode/") or k["symbol"] != "exact_e8p_int4_kernel":
            continue
        pointers = pointer_bindings(k, parse_host(reader.read(k["host_abi"]).decode())[0])
        if set(pointers) != {
            "Low",
            "High",
            "Sum",
            "P",
            "Book",
            "WS",
            "AS",
            "Counts",
            "Offsets",
            "TE",
            "TR",
            "TC",
            "Out",
        }:
            raise ValueError("Unexpected exact expert ABI")
        e, g, n, words = buffers[pointers["P"]["name"]]["shape"]
        width = g * 64
        if e != 512 or words != 8 or (n, width) not in ((1280, 2560), (2560, 640)):
            raise ValueError("Unsupported single-token expert geometry")
        for role, size in [
            ("Low", 10 * width // 2),
            ("High", 10 * width // 2),
            ("Sum", 40),
            ("TE", 44),
            ("TR", 44),
            ("TC", 4),
            ("Counts", 2048),
            ("Offsets", 2048),
        ]:
            b = buffers[pointers[role]["name"]]
            dtype_bytes = {"u32": 4, "i32": 4}[b["dtype"]]
            elements = 1
            for dim in b["shape"]:
                elements *= dim
            if pointers[role]["kind"] != "buffer" or elements * dtype_bytes != size:
                raise ValueError(f"Unexpected single-token expert {role}")
        key = f"n{n}-k{width}"
        if key not in compiled:
            export_kernel(e8p_single_decode_grouped(10, 512, 11, n, width), report / "aot" / key)
            assets = {}
            target = operator / "single-row-experts" / key
            for field, filename in [
                ("module", "kernel.cubin"),
                ("source", "kernel.cu"),
                ("host_abi", "host.txt"),
            ]:
                path = target / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                link_or_copy(report / "aot" / key / filename, path)
                assets[field] = dict(file=str(path.relative_to(operator)), sha256=file_hash(path))
            host = parse_host((target / "host.txt").read_text())[0]
            validate_parameter_count((target / "kernel.cu").read_text(), host)
            compiled[key] = (host, assets)
        host, assets = compiled[key]
        package["kernels"][pos] = replace_binding(k, host, assets, pointers)
        changed.append(k["name"])
    if len(changed) != 96:
        raise ValueError(f"Expected 96 decode expert projections, got {len(changed)}")
    digest = compact_package(operator, package, destination / "cache/packages")
    shutil.rmtree(operator)
    data["execution_package"] = digest
    write_json(destination / "cache/model.json", data)
    write_json(
        report / "results.json",
        dict(
            complete=True,
            changed=changed,
            compiled_variants=len(compiled),
            weights_unchanged=True,
            workspace_unchanged=True,
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--model-output", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--engine", type=Path)
    a = p.parse_args()
    with atomic_model(a.model_output, a.engine) as staging:
        upgrade(a.model.resolve(strict=True), staging, a.output, a.engine)
    print("SINGLE-TOKEN EXPERTS READY", a.model_output, flush=True)


if __name__ == "__main__":
    main()
