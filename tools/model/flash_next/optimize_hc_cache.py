"""Add a counted BF16 read cache for short HC projections; prefill retains BF16."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from safetensors.numpy import save_file

from tools.model.compact import compact_package
from tools.model.flash_next.hc_bf16 import pack
from tools.model.flash_next.optimize_decode import pointer_bindings, replace_binding
from tools.model.flash_next.optimize_hc import up_rank
from tools.model.publication import (
    AssetReader,
    atomic_model,
    clone_model,
    load_model,
    file_hash,
    link_or_copy,
    write_json,
)
from tools.model.safetensors_source import local_header
from tools.operators.abi import parse_host, validate_parameter_count


def upgrade(model, destination, report, engine=None):
    from kernels.model.hc_bf16_cache import hc_up13_mix
    from tools.operators.common import configure, export_kernel

    configure()
    data, origin, package = load_model(model)
    if data["architecture"] != "flash_next":
        raise ValueError("Expected Flash Next")
    operator = clone_model(model, destination, origin)
    reader = AssetReader(origin)
    buffers = {b["name"]: b for b in data["metadata"]["buffers"]}
    index_path = destination / "cache/weights/model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    headers = {}
    cached = {}
    arrays = {}
    variants = {}
    changed = []
    added = 0
    for pos, kernel in enumerate(package["kernels"]):
        host = parse_host(reader.read(kernel["host_abi"]).decode())[0]
        if any(
            a["ctype"] != "ctypes.c_void_p" or not a["value"].endswith(".data_ptr()")
            for a in host["ordered_arguments"]
        ):
            continue
        roles = pointer_bindings(kernel, host)
        if set(roles) != {"A", "W", "Normed", "Out"}:
            continue
        rows = kernel["grid"][0]  # Short HC has one eight-row CTA in the token dimension.
        if rows != 1:
            continue
        name = kernel["name"]
        program = name.split("/")[0]
        if (
            program == "decode"
            or "/private/" in name
            or program in ["mtp_draft", "head", "mtp_head"]
        ):
            m = 1
        else:
            import re

            match = re.fullmatch(
                r"(?:verify|prefill|head|mtp_warm|mtp_head|flash_batch|flash_dynamic)_m(\d+)",
                program,
            )
            if not match or int(match[1]) > 8:
                continue
            m = int(match[1])
        rank = up_rank(buffers, roles, m)
        weight_name = roles["W"]["name"]
        if weight_name not in cached:
            tensor = buffers[weight_name]["data"]["tensor"]
            path = model / "cache/weights" / index["weight_map"][tensor]
            if path not in headers:
                headers[path] = local_header(path)
            start, header = headers[path]
            low, high = header[tensor]["data_offsets"]
            raw = np.memmap(
                path, mode="r", dtype=np.uint16, offset=start + low, shape=(10240, rank)
            )
            raw = (
                raw.reshape(4, 80, 32, rank // 64, 64)
                .transpose(1, 3, 0, 2, 4)
                .copy()
                .reshape(80, rank // 64, 128, 64)
            )
            packed = pack(raw)
            if packed["patch_count"] > 3:
                raise ValueError("Unsupported BF16 read-cache escape count")
            bindings = {}
            for role, arr in [
                ("Lo", packed["low"].view(np.uint32)),
                ("Hi", packed["high"]),
                ("Patches", packed["patches"]),
            ]:
                buf = "HC13_" + weight_name + "_" + role
                arr = np.ascontiguousarray(arr)
                arrays[buf] = arr
                b = dict(
                    name=buf,
                    dtype="u64" if role == "Patches" else "u32",
                    shape=list(arr.shape),
                    layout="native_contiguous",
                    alignment=256,
                    access="read",
                    data=dict(tensor=buf, sha256=hashlib.sha256(arr.tobytes()).hexdigest()),
                )
                data["metadata"]["buffers"].append(copy.deepcopy(b))
                package["buffer_contracts"].append(
                    {k: copy.deepcopy(v) for k, v in b.items() if k != "data"}
                )
                data["buffer_scopes"][buf] = "weights"
                bindings[role] = dict(kind="buffer", name=buf)
                added += arr.nbytes
            cached[weight_name] = (packed["base"], packed["patch_count"], bindings)
        base, count, bindings = cached[weight_name]
        key = f"m{m}-r{rank}-e{base}-p{count}"
        if key not in variants:
            export_kernel(
                hc_up13_mix(m, 2560, rank, stage_count=1, base_exp=base, patch_count=count),
                report / "aot" / key,
            )
            assets = {}
            target = operator / "hc13-read-cache" / key
            for field, filename in [
                ("module", "kernel.cubin"),
                ("source", "kernel.cu"),
                ("host_abi", "host.txt"),
            ]:
                path = target / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                link_or_copy(report / "aot" / key / filename, path)
                assets[field] = dict(file=str(path.relative_to(operator)), sha256=file_hash(path))
            new_host = parse_host((target / "host.txt").read_text())[0]
            validate_parameter_count((target / "kernel.cu").read_text(), new_host)
            variants[key] = (assets, new_host)
        assets, new_host = variants[key]
        package["kernels"][pos] = replace_binding(
            kernel,
            new_host,
            assets,
            dict(A=roles["A"], Normed=roles["Normed"], Out=roles["Out"], **bindings),
        )
        changed.append(name)
    if not cached or not changed:
        raise ValueError("No short HC projections selected")
    extra = destination / "cache/weights/hc-bf16-read-cache.safetensors"
    save_file(
        arrays, extra, metadata={f"orin.layout.{name}": "native_contiguous" for name in arrays}
    )
    for name in arrays:
        index["weight_map"][name] = extra.name
    index["metadata"]["total_size"] = index["metadata"].get("total_size", 0) + added
    write_json(index_path, index)
    data["metadata"]["weight_bytes"] += added
    digest = compact_package(operator, package, destination / "cache/packages")
    shutil.rmtree(operator)
    data["execution_package"] = digest
    write_json(destination / "cache/model.json", data)
    write_json(
        report / "results.json",
        dict(
            complete=True,
            changed=changed,
            matrices=len(cached),
            compiled_variants=len(variants),
            added_resident_weight_bytes=added,
            weight_parameters_unchanged=data["metadata"]["weight_parameters"],
            weight_bpp=8 * data["metadata"]["weight_bytes"] / data["metadata"]["weight_parameters"],
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
    print("HC BF16 READ CACHE READY", a.model_output, flush=True)


if __name__ == "__main__":
    main()
