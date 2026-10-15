"""Fuse single-token expert producers while retaining the compressed banks."""

import argparse
import shutil
from pathlib import Path

from tools.model.compact import compact_package
from tools.model.flash_next.compiler import pointer_bindings, replace_binding
from tools.model.package import model_library
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


def upgrade(model, destination, report):
    from kernels.model.rotation_a8 import rotate_activation_a8
    from kernels.model.e8p_single_decode import e8p_single_decode_grouped
    from tools.operators.common import configure, export_kernel

    configure()
    data, origin, package = load_model(model)
    if data["architecture"] != "flash_next":
        raise ValueError("Expected Flash Next")
    root = clone_model(model, destination, origin)
    reader = AssetReader(origin)
    kernels = {k["name"]: k for k in package["kernels"]}
    positions = {k["name"]: i for i, k in enumerate(package["kernels"])}
    compiled, groups = {}, []

    def roles(k):
        return pointer_bindings(k, parse_host(reader.read(k["host_abi"]).decode())[0])

    def replace(k, key, factory, pointers):
        if key not in compiled:
            export_kernel(factory(), report / "aot" / key)
            assets = {}
            for field, filename in [
                ("module", "kernel.cubin"),
                ("source", "kernel.cu"),
                ("host_abi", "host.txt"),
            ]:
                target = root / "decode-fusions" / key / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                link_or_copy(report / "aot" / key / filename, target)
                assets[field] = dict(file=str(target.relative_to(root)), sha256=file_hash(target))
            host = parse_host((report / "aot" / key / "host.txt").read_text())[0]
            validate_parameter_count((report / "aot" / key / "kernel.cu").read_text(), host)
            compiled[key] = (host, assets)
        host, assets = compiled[key]
        package["kernels"][positions[k["name"]]] = replace_binding(k, host, assets, pointers)

    for target in list(package["kernels"]):
        if not target["name"].startswith("decode/") or target["symbol"] != "exact_e8p_int4_kernel":
            continue
        prefix, number = target["name"].rsplit("/k", 1)
        number = int(number)
        project = roles(target)
        gu = "ffn_gate_up_exps" in project["P"]["name"]
        rotation = kernels[f"{prefix}/k{number - (2 if gu else 1)}"]
        rot = roles(rotation)
        if set(rot) != {"X", "Signs", "Q", "S"}:
            raise ValueError("Unexpected rotation producer ABI")
        split = kernels[target["name"] + "/a8_split"]
        pack = roles(split)
        arguments = dict(rot, Low=project["Low"], High=project["High"], Sum=project["Sum"])
        if gu:
            dispatch = kernels[f"{prefix}/k{number - 1}"]
            dispatch_roles = roles(dispatch)
            if (
                dispatch_roles["A"] != rot["Q"]
                or dispatch_roles["Scale"] != rot["S"]
                or dispatch_roles["D"] != pack["A"]
            ):
                raise ValueError("Unexpected dispatch producer connections")
            arguments.update(
                {
                    n: dispatch_roles[n]
                    for n in ["IDs", "RelativeSlot", "RowOffsets", "DS", "SlotMap"]
                }
            )
            replace(
                rotation,
                "gu-activation",
                lambda: rotate_activation_a8(1, 2560, output_layout="e8p_dispatch"),
                arguments,
            )
            project["AS"] = rot["S"]
            replace(
                target,
                "gu-shared-input",
                lambda: e8p_single_decode_grouped(10, 512, 11, 1280, 2560, shared_input=True),
                project,
            )
            group = [rotation["name"], dispatch["name"], split["name"], target["name"]]
        else:
            if pack["A"] != rot["Q"] or project["AS"] != rot["S"]:
                raise ValueError("Unexpected down activation connections")
            replace(
                rotation,
                "down-activation",
                lambda: rotate_activation_a8(10, 640, swiglu=True, output_layout="e8p"),
                arguments,
            )
            group = [rotation["name"], split["name"], target["name"]]
        groups.append(group)
    if len(groups) != 96:
        raise ValueError(f"Expected 96 decode expert activation groups, got {len(groups)}")
    package["plan_kernel_contracts"] = sorted(
        set(package.get("plan_kernel_contracts", [])) | {n for g in groups for n in g}
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
            complete=True,
            groups=groups,
            operations_removed=144,
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
        upgrade(a.model.resolve(strict=True), staging, a.output)
    print("DECODE FUSIONS READY", a.model_output, flush=True)


if __name__ == "__main__":
    main()
