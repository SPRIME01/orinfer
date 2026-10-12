"""Use exact direct BF16 MMA operands for single-token HC read caches."""

import argparse
import math
import shutil
from pathlib import Path

import numpy as np

from tools.model.compact import compact_package
from tools.model.flash_next.optimize_decode import pointer_bindings, replace_binding
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
from tools.model.safetensors_source import local_header
from tools.operators.abi import parse_host, validate_parameter_count


def upgrade(model, destination, report, from_partials=False):
    from kernels.model.hc_bf16_cache import hc_up13_single
    from tools.operators.common import configure, export_kernel
    import json

    configure()
    data, origin, package = load_model(model)
    if data["architecture"] != "flash_next":
        raise ValueError("Expected Flash Next")
    root = clone_model(model, destination, origin)
    reader = AssetReader(origin)
    buffers = {b["name"]: b for b in data["metadata"]["buffers"]}
    index = json.loads((model / "cache/weights/model.safetensors.index.json").read_text())
    compiled, changed = {}, []
    kernels = {k["name"]: k for k in package["kernels"]}
    contracts = set(package.get("plan_kernel_contracts", []))
    for position, k in enumerate(package["kernels"]):
        if k["name"].split("/")[0] not in ("decode", "head_m1") or k["symbol"] not in (
            ("hc_up13_vector_project_kernel", "hc_up13_direct_kernel")
            if from_partials
            else ("hc_up13_vector_project_kernel",)
        ):
            continue
        roles = pointer_bindings(k, parse_host(reader.read(k["host_abi"]).decode())[0])
        original = roles["Lo"]["name"].removeprefix("HC13_").removesuffix("_Lo")
        w = buffers[original]
        if (
            w["dtype"] != "bf16"
            or len(w["shape"]) != 2
            or w["shape"][0] != 10240
            or w["shape"][1] not in (256, 320)
        ):
            raise ValueError("Invalid original HC BF16 geometry")
        rank = w["shape"][1]
        for role, shape in [("Lo", [80, rank // 64, 128, 16]), ("Hi", [80, rank // 64, 128, 10])]:
            b = buffers[roles[role]["name"]]
            if (
                b["dtype"] != "u32"
                or b["shape"] != shape
                or b["layout"] != "native_contiguous"
                or not b.get("data")
            ):
                raise ValueError("Invalid HC packed cache geometry")
        for role, size in [("A", rank), ("Normed", 10240), ("Out", 2560)]:
            b = buffers[roles[role]["name"]]
            if (
                b["dtype"] != "f16"
                or b.get("data")
                or roles[role].get("offset", 0) + size * 2 > math.prod(b["shape"]) * 2
            ):
                raise ValueError("HC single-token workspace overflow")
        if from_partials:
            prefix, number = k["name"].rsplit("/k", 1)
            finish = kernels[f"{prefix}/k{int(number) - 1}"]
            down = kernels[f"{prefix}/k{int(number) - 2}"]
            finish_roles = pointer_bindings(
                finish, parse_host(reader.read(finish["host_abi"]).decode())[0]
            )
            down_roles = pointer_bindings(
                down, parse_host(reader.read(down["host_abi"]).decode())[0]
            )
            if (
                set(finish_roles) != {"Out", "Partials"}
                or finish_roles["Out"] != roles["A"]
                or finish_roles["Partials"] != down_roles.get("Partials")
                or down_roles.get("A") != roles["Normed"]
            ):
                raise ValueError("Invalid HC finish producer connections")
            partial = buffers[finish_roles["Partials"]["name"]]
            if (
                partial["dtype"] != "f32"
                or partial.get("data")
                or math.prod(partial["shape"]) != 4 * rank
                or finish_roles["Partials"].get("offset", 0)
            ):
                raise ValueError("Invalid HC partial workspace")
            roles["A"] = finish_roles["Partials"]
            contracts.update([down["name"], finish["name"], k["name"]])
        tensor = w["data"]["tensor"]
        path = model / "cache/weights" / index["weight_map"][tensor]
        start, header = local_header(path)
        begin, _ = header[tensor]["data_offsets"]
        raw = np.memmap(
            path, mode="r", dtype=np.uint16, offset=start + begin, shape=tuple(w["shape"])
        )
        exponent = (raw >> 7) & 255
        base = max(0, int(exponent.max()) - 30)
        count = int(((exponent < base) | (exponent > base + 30)).sum())
        if count > 3:
            raise ValueError("Unsupported HC escape count")
        if count and "Patches" not in roles:
            raise ValueError("Missing HC escape binding")
        key = f"r{rank}-e{base}-p{count}"
        if key not in compiled:
            export_kernel(
                hc_up13_single(
                    rank=rank, base_exp=base, patch_count=count, from_partials=from_partials
                ),
                report / "aot" / key,
            )
            assets = {}
            for field, filename in [
                ("module", "kernel.cubin"),
                ("source", "kernel.cu"),
                ("host_abi", "host.txt"),
            ]:
                target = root / "hc-direct" / key / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                link_or_copy(report / "aot" / key / filename, target)
                assets[field] = dict(file=str(target.relative_to(root)), sha256=file_hash(target))
            host = parse_host((report / "aot" / key / "host.txt").read_text())[0]
            validate_parameter_count((report / "aot" / key / "kernel.cu").read_text(), host)
            compiled[key] = (host, assets)
        host, assets = compiled[key]
        package["kernels"][position] = replace_binding(k, host, assets, roles)
        changed.append(k["name"])
    if len(changed) != 97:
        raise ValueError(f"Expected 97 target HC up projections, got {len(changed)}")
    if from_partials:
        package["plan_kernel_contracts"] = sorted(contracts)
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
            changed=changed,
            weights_unchanged=True,
            workspace_unchanged=True,
            compiled_variants=len(compiled),
            operations_removed=len(changed) if from_partials else 0,
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--model-output", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--engine", type=Path)
    p.add_argument("--from-partials", action="store_true", help="Fuse exact HC down finish into up")
    a = p.parse_args()
    with atomic_model(a.model_output, a.engine) as staging:
        upgrade(a.model.resolve(strict=True), staging, a.output, a.from_partials)
    print("HC DIRECT READY", a.model_output, flush=True)


if __name__ == "__main__":
    main()
