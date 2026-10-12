"""Publish generated argument access for model-owned decode fork/join graphs."""

import argparse
import re
import shutil
from pathlib import Path

from tools.model.compact import compact_package
from tools.model.package import model_library
from tools.model.publication import (
    AssetReader,
    atomic_model,
    clone_model,
    file_hash,
    load_model,
    write_json,
)
from tools.operators.abi import parse_host, validate_parameter_count


def upgrade(model, destination, report):
    report.mkdir(parents=True, exist_ok=True)
    data, origin, package = load_model(model)
    if data["architecture"] != "flash_next":
        raise ValueError("Expected Flash Next")
    root = clone_model(model, destination, origin)
    reader = AssetReader(origin)
    contracts = set(package.get("plan_kernel_contracts", []))
    changed = []
    for k in package["kernels"]:
        if not k["name"].startswith("decode/"):
            continue
        source = reader.read(k["source"]).decode()
        host = parse_host(reader.read(k["host_abi"]).decode())[0]
        validate_parameter_count(source, host)
        prototype = re.search(r"\b" + re.escape(k["symbol"]) + r"\s*\(([^)]*)\)", source)
        if prototype is None:
            raise ValueError("Missing CUDA access prototype")
        declarations = prototype[1].split(",")
        if len(declarations) != len(k["args"]):
            raise ValueError("Mismatched CUDA access prototype")
        k["readonly_arguments"] = [
            "*" not in declaration or bool(re.search(r"\bconst\b", declaration))
            for declaration in declarations
        ]
        contracts.add(k["name"])
        changed.append(k["name"])
    if not changed:
        raise ValueError("Missing decode kernels")
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
            kernels_unchanged=True,
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
    print("GRAPH BRANCHES READY", a.model_output, flush=True)


if __name__ == "__main__":
    main()
