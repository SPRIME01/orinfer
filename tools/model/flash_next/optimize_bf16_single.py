"""Use exact direct BF16 operands for target routing and HC down projections."""

import argparse
import math
import shutil
from pathlib import Path

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
from tools.operators.abi import parse_host, validate_parameter_count


def upgrade(model, destination, report):
    from kernels.model.bf16_single import bf16_single
    from tools.operators.common import configure, export_kernel

    configure()
    data, origin, package = load_model(model)
    if data["architecture"] != "flash_next":
        raise ValueError("Expected Flash Next")
    root = clone_model(model, destination, origin)
    reader = AssetReader(origin)
    buffers = {b["name"]: b for b in data["metadata"]["buffers"]}
    compiled, changed = {}, []
    for position, k in enumerate(package["kernels"]):
        if k["name"].split("/")[0] not in ("decode", "head_m1"):
            continue
        host = parse_host(reader.read(k["host_abi"]).decode())[0]
        if any(
            a["ctype"] != "ctypes.c_void_p" or not a["value"].endswith(".data_ptr()")
            for a in host["ordered_arguments"]
        ):
            continue
        roles = pointer_bindings(k, host)
        if set(roles) not in ({"A", "Partials", "W"}, {"A", "Weight", "Output"}):
            continue
        weight = roles.get("W", roles.get("Weight"))
        name = weight["name"]
        down = name.endswith(
            ("hc_attn_down.weight_value", "hc_ffn_down.weight_value", "output_hc_down.weight_value")
        )
        router = name.endswith("ffn_gate_inp.weight_value")
        if not down and not router:
            continue
        w = buffers[name]
        groups = 4 if down else 1
        if (
            w["dtype"] != "bf16"
            or len(w["shape"]) != 2
            or w["shape"][1] != groups * 2560
            or not w.get("data")
            or w["layout"] != "native_contiguous"
            or w["shape"][0] not in ((256, 320) if down else (512,))
        ):
            raise ValueError("Unsupported BF16 projection weights")
        n = w["shape"][0]
        pointers = dict(
            A=roles["A"], Weight=weight, Output=roles["Partials"] if down else roles["Output"]
        )
        for role, dtype, size in [("A", "f16", groups * 2560), ("Output", "f32", groups * n)]:
            b = buffers[pointers[role]["name"]]
            if (
                b["dtype"] != dtype
                or b.get("data")
                or math.prod(b["shape"]) != size
                or pointers[role].get("offset", 0)
            ):
                raise ValueError("Invalid BF16 projection workspace")
        key = f"n{n}-g{groups}"
        if key not in compiled:
            export_kernel(bf16_single(n, 2560, groups), report / "aot" / key)
            assets = {}
            for field, filename in [
                ("module", "kernel.cubin"),
                ("source", "kernel.cu"),
                ("host_abi", "host.txt"),
            ]:
                target = root / "bf16-single" / key / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                link_or_copy(report / "aot" / key / filename, target)
                assets[field] = dict(file=str(target.relative_to(root)), sha256=file_hash(target))
            host = parse_host((report / "aot" / key / "host.txt").read_text())[0]
            validate_parameter_count((report / "aot" / key / "kernel.cu").read_text(), host)
            compiled[key] = (host, assets)
        host, assets = compiled[key]
        replacement = replace_binding(k, host, assets, pointers)
        # Exported A, Output, Weight ABI: only Output is mutable.
        replacement["readonly_arguments"] = [True, False, True]
        package["kernels"][position] = replacement
        changed.append(k["name"])
    if len(changed) != 145:
        raise ValueError(f"Expected 145 BF16 target projections, got {len(changed)}")
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
        upgrade(a.model.resolve(strict=True), staging, a.output)
    print("BF16 SINGLE READY", a.model_output, flush=True)


if __name__ == "__main__":
    main()
