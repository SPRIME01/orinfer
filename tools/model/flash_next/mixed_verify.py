"""Add native MTP target verification over the existing joined Flash body.

The shared HC/projection/MoE rows reuse published cubins. Causal verification
segments retain the exact compact GDN, convolution, PLE and QSA commit factors.
No new drafter, weight conversion or online compilation is required.
"""

import argparse
import ast
import copy
import math
import re
from pathlib import Path

from tools.model.publication import (
    AssetReader,
    atomic_model,
    clone_model,
    commit_package,
    file_hash,
    link_or_copy,
    load_model,
)
from tools.model.host_compiler import pointer_bindings, replace_binding
from tools.operators.abi import parse_host


def share_verification_state(data, package):
    """One request owns at most one live verification profile, including commit.

    Share temporary prefix/factor storage across widths, never across owners.
    Kernel shapes and copy lengths retain their original profile-specific ABI.
    """
    spec = data["metadata"]["mtp"]
    if spec.get("shared_verification_state"):
        return
    groups = {}
    aliases = {}
    for buffer in data["metadata"]["buffers"]:
        match = re.fullmatch(r"VerifyM[2-8]_(Prefix|Saved)_(.+)", buffer["name"])
        if match:
            name = f"VerifyShared_{match[1]}_{match[2]}"
            groups.setdefault(name, []).append(buffer)
            aliases[buffer["name"]] = name
    if not groups:
        raise ValueError("Missing verification prefix/factor storage")
    shared = []
    for name, variants in groups.items():
        if len({b["dtype"] for b in variants}) != 1 or any(b.get("data") for b in variants):
            raise ValueError("Incompatible verification state variants")
        largest = max(variants, key=lambda b: math.prod(b["shape"]))
        shared.append(dict(largest, name=name))
        data["buffer_scopes"][name] = "sequence"
    data["metadata"]["buffers"] = [
        b for b in data["metadata"]["buffers"] if b["name"] not in aliases
    ] + shared
    for name in aliases:
        del data["buffer_scopes"][name]
    data["metadata"]["reset_buffers"] = list(
        dict.fromkeys(aliases.get(name, name) for name in data["metadata"]["reset_buffers"])
    )
    for kernel in package["kernels"]:
        for arg in kernel["args"]:
            if arg.get("name") in aliases:
                arg["name"] = aliases[arg["name"]]
    spec["shared_verification_state"] = True


class Interfaces:
    def __init__(self, origin):
        self.reader = AssetReader(origin)
        self.cache = {}

    def get(self, kernel):
        key = kernel["host_abi"]["sha256"]
        if key not in self.cache:
            raw = self.reader.read(kernel["host_abi"]).decode()
            node = next(
                n
                for n in ast.parse(raw).body
                if isinstance(n, ast.FunctionDef) and n.name == "call"
            )
            (host,) = parse_host(raw)
            self.cache[key] = (
                [a.arg for a in node.args.args if a.arg not in ("stream", "kernels")],
                host,
            )
        return self.cache[key]

    def pointers(self, kernel):
        return pointer_bindings(kernel, self.get(kernel)[1], allow_scalars=True)


def upgrade(model, destination, output):
    import tilelang
    from kernels.model.speculative_factors import save_gdn_factors
    from tools.operators.common import configure, export_kernel

    configure()
    data, origin, package = load_model(model)
    metadata = data["metadata"]
    spec = metadata.get("mtp")
    if data["architecture"] != "flash_next" or not spec or not metadata["batch_layout"]["profiles"]:
        raise ValueError("Expected joined Flash execution with native MTP")
    if spec.get("batch_verification_tokens"):
        raise ValueError("Batched verification already installed")
    operator = clone_model(model, destination, origin)
    interface = Interfaces(origin)
    original = {k["name"]: k for k in package["kernels"]}
    additions = []
    stride = metadata["batch_layout"]["row_strides"]

    def private(name):
        if data["buffer_scopes"][name] == "workspace":
            data["buffer_scopes"][name] = "sequence"

    for tokens in range(2, 9):
        program = f"verify_m{tokens}"
        mapping = {f"VerifyM{tokens}_HC": "FlashMixedResidual"}
        asset = output / "aot" / f"save-factors-{tokens}"
        kernel = tilelang.compile(
            save_gdn_factors.__wrapped__(tokens),
            out_idx=[],
            execution_backend="nvrtc",
            target={"kind": "cuda", "arch": "sm_87"},
        )
        export_kernel(kernel, asset)
        assets = {}
        for field, filename in [
            ("module", "kernel.cubin"),
            ("source", "kernel.cu"),
            ("host_abi", "host.txt"),
        ]:
            target = operator / "verify-factors" / f"m{tokens}" / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            link_or_copy(asset / filename, target)
            assets[field] = dict(file=str(target.relative_to(operator)), sha256=file_hash(target))
        (factor_host,) = parse_host((asset / "host.txt").read_text())
        for layer in range(48):
            templates = [
                k
                for k in original.values()
                if k["name"].startswith(f"flash_batch_m2/layer{layer}/private/")
            ]
            signatures = {tuple(interface.get(k)[0]) for k in templates}
            source = [
                k for k in original.values() if k["name"].startswith(f"{program}/layer{layer}/")
            ]
            selected = [
                k
                for k in source
                if tuple(interface.get(k)[0]) in signatures
                or any(a.get("name", "").startswith(f"VerifyM{tokens}_Prefix_") for a in k["args"])
            ]
            boundaries = {tuple(interface.get(k)[0]): k for k in templates}
            for old in selected:
                template = boundaries.get(tuple(interface.get(old)[0]))
                if template:
                    old_pointers = interface.pointers(old)
                    for argument, value in interface.pointers(template).items():
                        name = value["name"]
                        if name.startswith("BatchM2_"):
                            role = name.removeprefix("BatchM2_")
                            if role not in stride:
                                raise ValueError("Unknown verification row boundary")
                            target = "FlashMixed_" + role
                            source_name = old_pointers[argument]["name"]
                            if source_name in mapping and mapping[source_name] != target:
                                raise ValueError("Conflicting verification row aliases")
                            mapping[source_name] = target
            counter = {"ple": 0, "private": 0}
            recurrence = None
            for old in selected:
                names, _ = interface.get(old)
                ple = layer == 1 and (
                    "Normed" in names
                    or any(a.get("name") == f"VerifyM{tokens}_Prefix_ple" for a in old["args"])
                )
                section = "ple" if ple else "private"
                k = copy.deepcopy(old)
                k["name"] = (
                    f"flash_verify_segment_m{tokens}/layer{layer}/{section}/k{counter[section]}"
                )
                counter[section] += 1
                for a in k["args"]:
                    if "name" not in a:
                        continue
                    name = a["name"]
                    if name in mapping:
                        a["name"] = mapping[name]
                    elif name.startswith((f"VerifyM{tokens}_Prefix_", f"VerifyM{tokens}_Saved_")):
                        private(name)
                additions.append(k)
                if names == ["Q", "K", "V", "G", "Beta", "State", "Prefix", "Out"]:
                    recurrence = k
            if layer % 4 != 3:
                if not recurrence or counter["private"] != 3:
                    raise ValueError(
                        "Expected convolution, prefix capture and compact GDN verification"
                    )
                source_pointers = interface.pointers(recurrence)
                bindings = {"K": source_pointers["K"], "G": source_pointers["G"]}
                for argument in ("SavedK", "SavedG"):
                    name = f"VerifyM{tokens}_Saved_{layer}_gdn_{argument[-1]}"
                    private(name)
                    bindings[argument] = dict(kind="buffer", name=name)
                k = replace_binding(
                    dict(name=f"flash_verify_segment_m{tokens}/layer{layer}/private/k3"),
                    factor_host,
                    assets,
                    bindings,
                )
                additions.append(k)
                counter["private"] += 1
            if counter["private"] != (20 if layer % 4 == 3 else 4) or counter["ple"] != (
                3 if layer == 1 else 0
            ):
                raise ValueError(f"Unexpected verification stages {tokens}/{layer}: {counter}")
        for role in (
            f"M{tokens}_Embedding",
            f"M{tokens}_Ple",
            "VerificationPairs",
            spec["verification_logits"],
        ):
            private(role)
        old = original[f"mtp_capture_m{tokens}/body/k0"]
        k = copy.deepcopy(old)
        k["name"] = f"flash_verify_capture_m{tokens}/body/k0"
        for a in k["args"]:
            if a.get("name") in mapping:
                a["name"] = mapping[a["name"]]
        additions.append(k)
    spec["batch_verification_tokens"] = list(range(2, 9))
    spec["batch_programs"] = sorted(
        {
            f"flash_verify_segment_m{tokens}/layer{layer}/private"
            for tokens in range(2, 9)
            for layer in range(48)
        }
        | {
            name
            for tokens in range(2, 9)
            for name in (
                f"flash_verify_segment_m{tokens}/layer1/ple",
                f"flash_verify_capture_m{tokens}",
            )
        }
    )
    package["kernels"].extend(additions)
    share_verification_state(data, package)
    package["buffer_contracts"] = [
        {k: v for k, v in b.items() if k != "data"} for b in metadata["buffers"]
    ]
    commit_package(destination, operator, data, package)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    args = parser.parse_args()
    with atomic_model(args.model_output) as staging:
        upgrade(args.model, staging, args.output)


if __name__ == "__main__":
    main()
