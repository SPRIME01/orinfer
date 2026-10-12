"""Publish shared Flash prefill/decode bodies and request-private causal segments.

Weights are immutable. Dense projections, hyperconnections and routed experts
see the concatenated rows; GDN/QSA/PLE see exactly one request's causal chunk.
Only request endpoints enter the vocabulary head. Compilation is offline.
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
    load_model,
    write_json,
)
from tools.model.flash_next.batching import INTERFACES, kind, shape_planner

CAPACITIES = (32, 64, 128, 256, 1024)
SEGMENTS = (1, 16, 128, 512)
RESIDUAL = "FlashMixedResidual"
HEAD_RESIDUAL = "FlashMixedHeadResidual"


def upgrade(model, destination, output):
    import torch
    import tilelang
    from kernels.model.rows import explicit_rows
    from tools.model.flash_next.dynamic_batch import factory
    from tools.model.flash_next.prepare import DTYPES, Publisher
    from tools.model.upgrade_dynamic_batch import contract
    from tools.operators.abi import parse_host
    from tools.operators.common import configure

    configure()
    data, origin, package = load_model(model)
    metadata = data["metadata"]
    layout = metadata.get("batch_layout", {})
    if (
        data["architecture"] != "flash_next"
        or not layout.get("state_columns")
        or layout["profiles"]
    ):
        raise ValueError("Expected a native parallel Flash decode package")
    operator = clone_model(model, destination, origin)
    buffers = {b["name"]: b for b in metadata["buffers"]}
    original = {k["name"]: k for k in package["kernels"]}
    strides = layout["row_strides"]
    reader = AssetReader(origin)
    hosts = {}

    def api(kernel):
        key = kernel["host_abi"]["sha256"]
        if key not in hosts:
            raw = reader.read(kernel["host_abi"]).decode()
            node = next(
                n
                for n in ast.parse(raw).body
                if isinstance(n, ast.FunctionDef) and n.name == "call"
            )
            (host,) = parse_host(raw)
            hosts[key] = (
                [a.arg for a in node.args.args if a.arg not in ("kernels", "stream")],
                host,
            )
        return hosts[key]

    def pointer(kernel, argument):
        _, host = api(kernel)
        values = [
            v
            for v, a in zip(kernel["args"], host["ordered_arguments"])
            if a["value"] == argument + ".data_ptr()"
        ]
        if len(values) != 1 or values[0]["kind"] != "buffer":
            raise ValueError("Expected one whole private interface buffer")
        return values[0]["name"]

    initialize = original[f"decode/begin/k{int(metadata.get('vision') is not None)}"]
    residual_role = pointer(initialize, api(initialize)[0][1])

    def mixed_role(role):
        return RESIDUAL if role == residual_role else "FlashMixed_" + role

    def arena(name, dtype, size):
        if name in buffers:
            if math.prod(buffers[name]["shape"]) != size or buffers[name]["dtype"] != dtype:
                raise ValueError("Incompatible shared row arena")
            return
        buffers[name] = dict(
            name=name,
            dtype=dtype,
            shape=[size],
            layout="native_contiguous",
            alignment=256,
            access="read_write",
        )
        data["buffer_scopes"][name] = "workspace"

    element_bytes = {
        v[0]: torch.empty((), dtype=getattr(torch, k)).element_size() for k, v in DTYPES.items()
    }
    for role, stride in strides.items():
        if role in ("Logits", "Selected"):
            continue
        dtype = buffers[f"BatchM128_{role}"]["dtype"]
        arena(mixed_role(role), dtype, max(CAPACITIES) * stride // element_bytes[dtype])
    arena(HEAD_RESIDUAL, "f16", 128 * 4 * 2560)

    # Formal parameter identities survive catalog bundling. Never infer a
    # private mixer or its tensor layout from an unstable source filename.
    templates = {}
    for layer in range(48):
        prefix = f"flash_batch_m2/layer{layer}/private/"
        templates[layer] = [k for k in original.values() if k["name"].startswith(prefix)]
    boundaries = {}
    additions = []
    private_counts = {}
    for tokens in SEGMENTS:
        program = "decode" if tokens == 1 else f"prefill_m{tokens}"
        by_role = {}
        residual = pointer(
            original[f"{program}/begin/k{int(metadata.get('vision') is not None)}"],
            api(initialize)[0][1],
        )
        by_role[residual] = RESIDUAL
        for layer in range(48):
            prefix = f"{program}/layer{layer}/"
            source = (
                templates[layer]
                if tokens == 1
                else [k for k in original.values() if k["name"].startswith(prefix)]
            )
            signatures = {tuple(api(k)[0]) for k in templates[layer]}
            private = [k for k in source if tuple(api(k)[0]) in signatures]
            if [api(k)[0] for k in private] != [api(k)[0] for k in templates[layer]]:
                raise ValueError(f"Private causal segment differs at {tokens}/{layer}")
            for source_k, template in zip(private, templates[layer]):
                names, host = api(template)
                for a, v in zip(host["ordered_arguments"], template["args"]):
                    if not v.get("name", "").startswith("BatchM2_"):
                        continue
                    role = v["name"].removeprefix("BatchM2_")
                    name = pointer(source_k, a["value"].removesuffix(".data_ptr()"))
                    target = mixed_role(role)
                    if name in by_role and by_role[name] != target:
                        raise ValueError("Private interface aliases incompatible shared rows")
                    by_role[name] = target
            section = "ple" if layer == 1 else "private"
            counters = {}
            for i, old in enumerate(private):
                section = "ple" if layer == 1 and i < 2 else "private"
                k = copy.deepcopy(old)
                slot = counters.get(section, 0)
                k["name"] = f"flash_segment_m{tokens}/layer{layer}/{section}/k{slot}"
                counters[section] = slot + 1
                for a in k["args"]:
                    if a.get("name") in by_role:
                        a["name"] = by_role[a["name"]]
                additions.append(k)
            private_counts[tokens, layer] = counters
        boundaries[tokens] = by_role
        width = 4096 if tokens >= 256 else tokens
        for name in (f"M{width}_Embedding", f"M{width}_Ple"):
            data["buffer_scopes"][name] = "sequence"
        # MTP ring capture must retain each segment's own hidden history.
        if metadata.get("mtp"):
            for old in original.values():
                if old["name"].startswith(f"mtp_capture_m{tokens}/"):
                    k = copy.deepcopy(old)
                    k["name"] = k["name"].replace("mtp_capture_", "flash_mixed_capture_", 1)
                    for a in k["args"]:
                        if a.get("name") in by_role:
                            a["name"] = by_role[a["name"]]
                    additions.append(k)

    # Fixed small bodies reuse the proven cubins, but share the same row arena
    # with large bodies and private causal segments.
    counts = {}
    contracts = []
    dynamic = {k["name"]: k for k in package["dynamic_batch_kernels"]}
    for rows in metadata["batch_profiles"]:
        for old in original.values():
            if not old["name"].startswith(f"flash_batch_m{rows}/head/"):
                continue
            k = copy.deepcopy(old)
            k["name"] = k["name"].replace("flash_batch_", "flash_mixed_head_", 1)
            for a in k["args"]:
                if a.get("name") == f"BatchM{rows}_{residual_role}":
                    a["name"] = HEAD_RESIDUAL
            additions.append(k)
    for rows in CAPACITIES:
        if rows > 128:
            continue
        for old in original.values():
            prefix = f"flash_dynamic_m{rows}/"
            if (
                not old["name"].startswith(prefix)
                or "/private/" in old["name"]
                or "/head/" in old["name"]
            ):
                continue
            k = copy.deepcopy(old)
            k["name"] = k["name"].replace("flash_dynamic_", "flash_mixed_", 1)
            for a in k["args"]:
                name = a.get("name", "")
                if name.startswith(f"BatchM{rows}_"):
                    role = name.removeprefix(f"BatchM{rows}_")
                    if role in strides:
                        a["name"] = mixed_role(role)
            additions.append(k)
            launch = copy.deepcopy(dynamic[old["name"]])
            launch["name"] = k["name"]
            contracts.append(launch)
            section = k["name"].split("/", 1)[1].rsplit("/", 1)[0]
            counts.setdefault(rows, {}).setdefault(section, 0)
            counts[rows][section] += 1

    def compile_shared(key):
        if key.startswith(("mixed-", "greedy-", "dense-gemv-")):
            return None
        fn = factory(key, planner.mixed_capacity)
        if fn is None:
            return None
        return tilelang.compile(
            explicit_rows(fn),
            out_idx=[],
            execution_backend="nvrtc",
            target={"kind": "cuda", "arch": "sm_87"},
        )

    planner, logical = shape_planner(metadata, buffers, original, output, compile_shared)
    planner.expert_profile_rows = {256: 512, 1024: 512}
    planner.exact_mixed_experts = True
    nibble = [t for n, t in planner.weight_tensors.items() if n.startswith("W_e8p_nibble_")]
    if planner.expert_packed_k != 64 or len(nibble) != 1:
        raise ValueError("Mixed experts require the validated K64/nibble book")

    class MixedPublisher(Publisher):
        planes = None

        def buffer(self, tensor, name=None, scope="workspace"):
            pointer = tensor.untyped_storage()._cdata
            offset = tensor.storage_offset() * tensor.element_size()
            if pointer in logical:
                return logical[pointer], offset
            if pointer not in self.storage:
                role = self.interfaces.get(pointer)
                name = mixed_role(role) if role else f"FlashMixedM{self.rows}_Scratch{self.number}"
                self.number += 1
                self.storage[pointer] = name
                if role is None:
                    arena(
                        name,
                        DTYPES[str(tensor.dtype).removeprefix("torch.")][0],
                        tensor.untyped_storage().nbytes() // tensor.element_size(),
                    )
            return self.storage[pointer], offset

        def bind(self, *args):
            program, section, label, values, rows = args
            if label.startswith(("expert-gu-", "expert-down-")):
                return self.expert(program, section, values, rows)
            bound_rows = rows * 10 if label.startswith("rotate-a8-ffn-") else rows
            op = super().bind(program, section, label, values, bound_rows)
            for field in ("module", "source", "host_abi"):
                asset = self.kernels[-1][field]
                if not asset["file"].startswith("flash-mixed/"):
                    asset["file"] = "flash-mixed/" + asset["file"]
            return op

        def expert(self, program, section, values, rows):
            from kernels.model.e8p_prefill import split_prefill_a8, e8p_prefill_grouped

            a, p, _book, _patch, ws, scale, counts, offsets, te, tr, tc, out = values
            assignments, k = a.shape
            n, e, tiles = out.shape[1], p.shape[0], te.numel()
            if self.planes is None:
                self.planes = {}
            if (rows, k) not in self.planes:
                self.planes[rows, k] = (
                    torch.empty((k // 64, assignments, 8), device="cuda", dtype=torch.uint32),
                    torch.empty((k // 64, assignments, 8), device="cuda", dtype=torch.uint32),
                    torch.empty((assignments,), device="cuda", dtype=torch.int32),
                )
            low, high, sums = self.planes[rows, k]
            split = f"mixed-split-{assignments}-{k}"
            planner.kernel(split, lambda: split_prefill_a8(assignments, k))
            self.bind(program, section, split, (a, low, high, sums), rows)
            # Preserve the registered projection slot. The Rust expert recipe
            # inserts its extra split by the validated /a8_split ABI contract.
            self.kernels[-1]["name"] += "/a8_split"
            exact = f"mixed-exact-{assignments}-{tiles}-{n}-{k}"
            planner.kernel(exact, lambda: e8p_prefill_grouped(assignments, e, tiles, n, k))
            return self.bind(
                program,
                section,
                exact,
                (
                    low,
                    high,
                    sums,
                    p,
                    nibble[0],
                    ws,
                    scale.reshape(-1),
                    counts,
                    offsets,
                    te,
                    tr,
                    tc,
                    out,
                ),
                rows,
            )

    publish = MixedPublisher(planner, destination)
    publish.cache = operator / "flash-mixed"
    (publish.cache / "kernels").mkdir(parents=True)
    for rows in (256, 1024):
        publish.rows, publish.storage, publish.interfaces = rows, {}, {}
        planner.mixed_capacity = rows
        planner.position = planner.capacity - rows
        plan = planner.plan(rows)
        publish.interfaces.update(
            {
                plan[n].untyped_storage()._cdata: r
                for n, r in [
                    ("embedding", "M1_Embedding"),
                    ("ple_embedding", "M1_Ple"),
                    ("residual", residual_role),
                ]
            }
        )
        entries = list(zip(plan["ops"], plan["labels"]))[: plan["body_count"]]
        layers, layer = [], 0
        for (kernel, args), label in entries:
            weight_layers = [
                int(match[1])
                for a in args
                if isinstance(a, torch.Tensor)
                and (
                    match := re.match(
                        r"W_blk\.(\d+)\.", logical.get(a.untyped_storage()._cdata, "")
                    )
                )
            ]
            if weight_layers:
                layer = min(weight_layers)
            if label.startswith("dense-quant-") and args[0] is plan["ple_embedding"]:
                layer = 1
            layers.append(layer)
        for ((_, args), label), layer in zip(entries, layers):
            knd = kind(label)
            if knd not in INTERFACES:
                continue
            template = next(
                k
                for k in templates[layer]
                if tuple(api(k)[0])
                == {
                    "gdn-conv": (
                        "X",
                        "W",
                        "HI",
                        "lengths",
                        "positions",
                        "Q",
                        "K",
                        "V",
                        "HO",
                        "positions_out",
                    ),
                    "gdn-sequence": ("Q", "K", "V", "G", "Beta", "State", "Prefix", "Out"),
                    "ple-conv": ("Normed", "Gated", "State", "Weight", "Output"),
                    "ple-history": ("Normed", "State", "NextState"),
                    "qsa-prepare": tuple(api(templates[layer][0])[0]),
                    "index-query": ("QK", "Weight", "Position", "Query", "Coordinates"),
                    "index-compress": (
                        "QK",
                        "Pending",
                        "Weight",
                        "Position",
                        "Cache",
                        "Coordinates",
                    ),
                    "index-pending": ("QK", "Position", "Pending"),
                    "qsa-merge": ("Max", "Den", "Out", "Gate", "Output"),
                }[knd]
            )
            names, _ = api(template)
            for index in INTERFACES[knd]:
                role = pointer(template, names[index]).removeprefix("BatchM2_")
                tensor = args[index]
                if tensor.numel() * tensor.element_size() != rows * strides[role]:
                    raise ValueError("Shared interface is not token-major")
                publish.interfaces[tensor.untyped_storage()._cdata] = role
        stage = "pre"
        for index, (((_, args), label), layer) in enumerate(zip(entries, layers)):
            if index == 0:
                section = "begin"
            else:
                if index == 1 or layers[index - 1] != layer:
                    stage = "ple" if layer == 1 else "pre"
                knd = kind(label)
                if knd:
                    stage = "pre" if knd.startswith("ple-") else "post"
                    continue
                if label == "gdn-history-copy":
                    continue
                section = (
                    f"layer{layer}/pre"
                    if label.startswith("index-qk:")
                    else f"layer{layer}/{stage}"
                )
            op = publish.bind(f"flash_mixed_m{rows}", section, label, args, rows)
            publish.groups[f"flash_mixed_m{rows}"][section].append(op)
            for a in publish.kernels[-1]["args"]:
                if data["buffer_scopes"].get(a.get("name")) == "sequence":
                    raise ValueError("Shared stage reads private causal state")
        counts[rows] = {
            section: len(ops) for section, ops in publish.groups[f"flash_mixed_m{rows}"].items()
        }
    additions.extend(publish.kernels)
    for kernel in publish.kernels:
        (host,) = parse_host(AssetReader(operator).read(kernel["host_abi"]).decode())
        rows = int(kernel["name"].split("/")[0].rsplit("m", 1)[1])
        # Rotation of expert FFN rows uses ten assignments per token. Exact
        # expert planes keep capacity strides; Counts select live assignments.
        scale = 10 if "rotate-a8-ffn-" in kernel["host_abi"]["file"] else 1
        try:
            contracts.append(contract(kernel, host, rows, scale))
        except ValueError as error:
            raise ValueError(f"{kernel['name']}: {error}") from error
    metadata["buffers"] = list(buffers.values())
    layout["profiles"] = {str(n): "flash_recurrent" for n in SEGMENTS if n > 1}
    layout["small_mixed_shapes"] = [16, 128, 512]
    metadata["prefill_batch_profiles"] = [
        dict(tokens=n, kind="flash_mixed") for n in CAPACITIES if n > 128
    ]
    package["prefill_batch_profiles"] = copy.deepcopy(metadata["prefill_batch_profiles"])
    package["kernels"].extend(additions)
    package["dynamic_batch_kernels"].extend(contracts)
    package["plan_kernel_contracts"] = sorted(
        set(package.get("plan_kernel_contracts", []))
        | {
            k["name"]
            for k in additions
            if k["name"].endswith("/a8_split")
            or any(a.get("name", "").endswith("_exps.weight_packed") for a in k["args"])
        }
    )
    package["buffer_contracts"] = [
        {k: v for k, v in b.items() if k != "data"} for b in metadata["buffers"]
    ]
    write_json(output / "stages.json", counts)
    write_json(
        output / "private-stages.json", {f"{n}/{l}": v for (n, l), v in private_counts.items()}
    )
    assets = AssetReader(operator)
    checked = set()
    for kernel in additions:
        for field in ("module", "source", "host_abi"):
            asset = kernel[field]
            identity = (asset["file"], asset.get("tensor"), asset["sha256"])
            if identity not in checked:
                assets.read(asset)
                checked.add(identity)
    commit_package(destination, operator, data, package)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-cache", type=Path)
    args = parser.parse_args()
    if args.compile_cache:
        from tools.model.publication import seed_compile_cache

        seed_compile_cache(args.compile_cache, args.output / "cache/0.1.15")
    with atomic_model(args.model_output) as staging:
        upgrade(args.model.resolve(), staging, args.output)


if __name__ == "__main__":
    main()
