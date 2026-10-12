"""Rebuild single-token MoE metadata without changing weights or route order."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from tools.model.compact import compact_package
from tools.model.flash_next.optimize_decode import pointer_bindings, replace_binding
from tools.model.package import model_library
from tools.model.publication import (
    ROOT,
    AssetReader,
    atomic_model,
    clone_model,
    file_hash,
    link_or_copy,
    load_model,
    write_json,
)
from tools.operators.abi import parse_host, validate_parameter_count


def upgrade(model, destination, report, engine=None):
    from kernels.model.moe import single_route_metadata
    from tools.operators.common import configure, export_kernel

    configure()
    cli = engine if engine is not None else ROOT / "target/release/orinfer"
    manifest = json.loads(
        subprocess.check_output(
            [str(cli.resolve(strict=True)), "plan-model", str(model)], text=True
        )
    )["manifest"]
    data, origin, package = load_model(model)
    if data["architecture"] != "flash_next" or data["compute_policy"] != "int8_quality":
        raise ValueError("Expected native Flash Next INT8 quality model")
    operator = clone_model(model, destination, origin)
    reader = AssetReader(origin)
    specs = {k["name"]: k for k in package["kernels"]}
    buffers = {b["name"]: b for b in data["metadata"]["buffers"]}
    operations = manifest["programs"]["decode"]
    groups = []
    export_kernel(single_route_metadata(), report / "aot")
    host = parse_host((report / "aot/host.txt").read_text())[0]
    validate_parameter_count((report / "aot/kernel.cu").read_text(), host)
    assets = {}
    for field, filename in [
        ("module", "kernel.cubin"),
        ("source", "kernel.cu"),
        ("host_abi", "host.txt"),
    ]:
        target = operator / "single-route-metadata" / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        link_or_copy(report / "aot" / filename, target)
        assets[field] = dict(file=str(target.relative_to(operator)), sha256=file_hash(target))
    for index, operation in enumerate(operations):
        if operation.get("kind") != "kernel":
            continue
        name = operation["name"]
        kernel = specs[name]
        parsed = parse_host(reader.read(kernel["host_abi"]).decode())[0]
        if any(
            a["ctype"] != "ctypes.c_void_p" or not a["value"].endswith(".data_ptr()")
            for a in parsed["ordered_arguments"]
        ):
            continue
        roles = pointer_bindings(kernel, parsed)
        if set(roles) != {"Counts", "IDs", "RelativeSlot"}:
            continue
        following = operations[index + 1 : index + 3]
        if len(following) != 2 or any(op.get("kind") != "kernel" for op in following):
            raise ValueError("Missing adjacent routing metadata producers")
        names = [name, *[op["name"] for op in following]]
        prefix, tiles = [
            pointer_bindings(specs[n], parse_host(reader.read(specs[n]["host_abi"]).decode())[0])
            for n in names[1:]
        ]
        if (
            set(prefix) != {"Counts", "RowOffsets", "TileOffsets", "TileCount"}
            or set(tiles) != {"Counts", "TileExpert", "TileOffsets", "TileRow"}
            or roles["Counts"] != prefix["Counts"]
            or roles["Counts"] != tiles["Counts"]
            or prefix["TileOffsets"] != tiles["TileOffsets"]
        ):
            raise ValueError("Routing metadata producer/consumer mismatch")
        arguments = dict(
            IDs=roles["IDs"],
            Counts=roles["Counts"],
            Relative=roles["RelativeSlot"],
            Offsets=prefix["RowOffsets"],
            TileOffsets=prefix["TileOffsets"],
            TileExpert=tiles["TileExpert"],
            TileRow=tiles["TileRow"],
            TileCount=prefix["TileCount"],
        )
        for role, elements in [
            ("IDs", 10),
            ("Counts", 512),
            ("Relative", 10),
            ("Offsets", 512),
            ("TileOffsets", 512),
            ("TileExpert", 11),
            ("TileRow", 11),
            ("TileCount", 1),
        ]:
            binding = arguments[role]
            buffer = buffers[binding["name"]]
            if (
                binding["kind"] != "buffer"
                or buffer["dtype"] != "i32"
                or buffer["shape"] != [elements]
            ):
                raise ValueError(f"Invalid single-token routing buffer: {role}")
        replacement = replace_binding(kernel, host, assets, arguments)
        position = next(i for i, k in enumerate(package["kernels"]) if k["name"] == name)
        package["kernels"][position] = replacement
        groups.append(names)
    if len(groups) != 48:
        raise ValueError(f"Expected 48 decode routing groups, got {len(groups)}")
    package["plan_kernel_contracts"] = sorted(
        set(package.get("plan_kernel_contracts", [])) | {name for group in groups for name in group}
    )
    library = operator / package["execution"]["library"]["file"]
    library.unlink()
    shutil.copyfile(model_library(), library)
    package["execution"]["library"]["sha256"] = file_hash(library)
    digest = compact_package(operator, package, destination / "cache/packages")
    shutil.rmtree(operator)
    data["execution_package"] = digest
    write_json(destination / "cache/model.json", data)
    write_json(report / "results.json", dict(complete=True, groups=groups, operations_removed=96))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine", type=Path)
    args = parser.parse_args()
    with atomic_model(args.model_output, args.engine) as staging:
        upgrade(args.model.resolve(strict=True), staging, args.output, args.engine)
    print("SINGLE-TOKEN ROUTING PACKAGE READY", args.model_output, flush=True)


if __name__ == "__main__":
    main()
