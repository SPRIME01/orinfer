"""Write single-token GDN convolution history directly to its private state."""

import argparse
import re
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
    from kernels.operators.op08_gdn_conv_prep import gdn_conv_prep
    from tools.operators.common import configure, export_kernel

    configure()
    data, origin, package = load_model(model)
    if data["architecture"] != "flash_next":
        raise ValueError("Expected Flash Next")
    root = clone_model(model, destination, origin)
    reader = AssetReader(origin)
    old_roles = {"HI", "HO", "K", "Q", "V", "W", "X", "lengths", "positions", "positions_out"}
    compiled, changed = None, []
    for position, k in enumerate(package["kernels"]):
        if not k["name"].startswith("decode/") or not any(
            a.get("name", "").endswith("ssm_conv1d.weight_value") for a in k["args"]
        ):
            continue
        roles = pointer_bindings(k, parse_host(reader.read(k["host_abi"]).decode())[0])
        if set(roles) != old_roles or k["grid"] != [1, 80, 1] or k["block"] != [16, 1, 1]:
            raise ValueError("Expected compact single-token convolution")
        if compiled is None:
            export_kernel(
                gdn_conv_prep(
                    B=1,
                    tokens=1,
                    compact_decode=True,
                    inplace_history=True,
                    normalize_round_fp16=False,
                    conv_product_round_fp16=False,
                    weight_dtype="float32",
                ),
                report / "aot",
            )
            assets = {}
            for field, filename in [
                ("module", "kernel.cubin"),
                ("source", "kernel.cu"),
                ("host_abi", "host.txt"),
            ]:
                target = root / "conv-history" / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                link_or_copy(report / "aot" / filename, target)
                assets[field] = dict(file=str(target.relative_to(root)), sha256=file_hash(target))
            host = parse_host((report / "aot/host.txt").read_text())[0]
            source = (report / "aot/kernel.cu").read_text()
            validate_parameter_count(source, host)
            compiled = (host, assets, source)
        host, assets, source = compiled
        roles["History"] = roles.pop("HI")
        del roles["HO"]
        replacement = replace_binding(k, host, assets, roles)
        declarations = re.search(r"\b" + re.escape(host["symbol"]) + r"\(([^)]*)\)", source)[
            1
        ].split(",")
        replacement["readonly_arguments"] = [bool(re.search(r"\bconst\b", d)) for d in declarations]
        package["kernels"][position] = replacement
        changed.append(k["name"])
    if len(changed) != 36:
        raise ValueError(f"Expected 36 target GDN convolutions, got {len(changed)}")
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
            copies_removed=36,
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
    print("CONV HISTORY READY", a.model_output, flush=True)


if __name__ == "__main__":
    main()
