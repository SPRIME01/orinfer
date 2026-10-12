"""Recompile HC stream gates with the SM87 FP16 rounding-preserving primitive."""

import argparse
import copy
from pathlib import Path
import re
import shutil

from tools.model.compact import compact_package
from tools.model.publication import (
    AssetReader,
    atomic_model,
    clone_model,
    file_hash,
    load_model,
    write_json,
)
from tools.operators.abi import evaluate, parse_host


def up_rank(buffers, roles, rows):
    """Main HC rank320 and draft/output-head rank256 share a launch geometry."""
    weight = buffers[roles["W"]["name"]]
    shape = weight["shape"]
    if (
        weight["dtype"] != "bf16"
        or len(shape) != 2
        or shape[0] != 4 * 2560
        or shape[1] not in (256, 320)
        or not weight.get("data")
    ):
        raise ValueError("Unexpected HC up weight contract")
    rank = shape[1]
    for role, elements in (("A", rows * rank), ("Normed", rows * 4 * 2560), ("Out", rows * 2560)):
        binding = roles[role]
        buffer = buffers[binding["name"]]
        size = 1
        for dimension in buffer["shape"]:
            size *= dimension
        if buffer["dtype"] != "f16" or binding.get("offset", 0) + elements * 2 > size * 2:
            raise ValueError(f"HC up {role} binding exceeds its stream workspace")
    return rank


def upgrade(model, destination, report):
    from kernels.model.hyperconnection import hc_combine, hc_combine_norm, hc_mix, hc_up_mix
    from tools.operators.common import configure, export_kernel

    configure()

    data, origin, package = load_model(model)
    if data["architecture"] != "flash_next":
        raise ValueError("Expected Flash Next model")
    root = clone_model(model, destination, origin)
    package = copy.deepcopy(package)
    reader = AssetReader(origin)
    buffers = {b["name"]: b for b in data["metadata"]["buffers"]}
    variants, updated, skipped = {}, {}, 0
    factories = {
        frozenset(("Normed", "Up", "Mixed")): ("mix", hc_mix, "Mixed"),
        frozenset(("Block", "Residual", "Inject", "Output")): ("combine", hc_combine, "Output"),
        frozenset(("Block", "Residual", "Inject", "Output", "Weight", "Normed")): (
            "combine_norm",
            hc_combine_norm,
            "Output",
        ),
        frozenset(("A", "W", "Normed", "Out")): ("up_mix", hc_up_mix, "Out"),
    }
    for i, kernel in enumerate(package["kernels"]):
        (host,) = parse_host(reader.read(kernel["host_abi"]).decode())
        roles = {}
        dynamic = False
        for argument, binding in zip(host["ordered_arguments"], kernel["args"], strict=True):
            if argument["ctype"] == "ctypes.c_void_p" and argument["value"].endswith(".data_ptr()"):
                roles[argument["value"].removesuffix(".data_ptr()")] = binding
            elif argument["value"] == "m":
                dynamic = True
            else:
                roles = {}
                break
        selected = factories.get(frozenset(roles))
        if selected is None:
            continue
        kind, factory, output = selected
        if buffers[roles[output]["name"]]["dtype"] != "f16":
            continue
        program = kernel["name"].split("/", 1)[0]
        match = re.fullmatch(
            r"(?:prefill|verify|head|mtp_warm|mtp_head|flash_batch|flash_dynamic)_m(\d+)", program
        )
        if dynamic:
            if match is None:
                raise ValueError("Unknown symbolic HC profile")
            rows = int(match[1])
        elif kind != "up_mix":
            rows = kernel["grid"][0]
        elif "/private/" in kernel["name"] or program == "decode":
            rows = 1
        elif match:
            rows = int(match[1])
        else:
            raise ValueError("Unknown static HC up profile")
        rank = 0
        if kind == "up_mix":
            rank = up_rank(buffers, roles, rows)
        key = f"{kind}-m{rows}-k{rank}-dynamic{int(dynamic)}"
        if key not in variants:
            if kind == "combine_norm":
                if dynamic:
                    skipped += 1
                    continue
                op = factory(rows, 2560, dtype="float16")
            elif kind == "up_mix":
                op = factory(rows, 2560, rank, dynamic_rows=dynamic)
            else:
                op = factory(rows, 2560, dtype="float16", dynamic_rows=dynamic)
            target = report / "aot" / key
            export_kernel(op, target)
            (new_host,) = parse_host((target / "host.txt").read_text())
            assets = {}
            for field, filename in (
                ("module", "kernel.cubin"),
                ("source", "kernel.cu"),
                ("host_abi", "host.txt"),
            ):
                path = root / "hc-half" / key / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(target / filename, path)
                assets[field] = dict(file=str(path.relative_to(root)), sha256=file_hash(path))
            variants[key] = (new_host, assets)
        new_host, assets = variants[key]
        if new_host["ordered_arguments"] != host["ordered_arguments"]:
            raise ValueError(f"HC ABI changed: {kernel['name']}")
        launch = new_host["launch_expressions"]
        geometry = {
            "grid": [evaluate(launch["gridDim" + axis], {"m": rows}) for axis in "XYZ"],
            "block": [evaluate(launch["blockDim" + axis], {"m": rows}) for axis in "XYZ"],
            "shared_memory_bytes": evaluate(launch["sharedMemBytes"], {"m": rows}),
        }
        if any(kernel[field] != value for field, value in geometry.items()):
            # A package may already have a different tuned execution tile.
            # A math-only rebuild must preserve its launch contracts.
            skipped += 1
            continue
        package["kernels"][i] = dict(kernel, **assets, symbol=new_host["symbol"])
        updated[kind] = updated.get(kind, 0) + 1
    if not updated:
        raise ValueError("No compatible HC gates")
    digest = compact_package(root, package, destination / "cache/packages")
    shutil.rmtree(root)
    data["execution_package"] = digest
    write_json(destination / "cache/model.json", data)
    write_json(
        report / "result.json",
        dict(
            execution_package=digest,
            updated=updated,
            skipped_geometry=skipped,
            variants=len(variants),
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with atomic_model(args.model_output) as staging:
        upgrade(args.model.resolve(strict=True), staging, args.output)


if __name__ == "__main__":
    main()
