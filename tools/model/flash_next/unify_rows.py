"""Publish shared token-row computation for target, mixed batches and MTP.

Request-owned causal segments, capture and accepted-prefix restoration remain
separate. Shapes select local tiles, never a different model execution recipe.
"""

import argparse
import copy
import math
import re
from pathlib import Path

from tools.model.flash_next.optimize_decode import pointer_bindings, replace_binding
from tools.model.publication import (
    AssetReader,
    atomic_model,
    clone_model,
    commit_package,
    file_hash,
    link_or_copy,
    load_model,
    write_json,
)
from tools.model.upgrade_dynamic_batch import contract
from tools.operators.abi import parse_host, validate_parameter_count

CAPACITY = 1024
REQUESTS = 128
RESIDUAL = "FlashDecode_Residual"


def obsolete_computation(name, *, draft=True):
    """Keep causal controls/capture, discard replaced scene-specific math."""
    if name.startswith(("flash_mixed_m", "flash_mixed_head_m")):
        return True
    if re.match(r"verify_m[2-8]/layer", name):
        return True
    if re.match(r"verify_m[2-8]/head/", name):
        return name.rsplit("/", 1)[-1] != "k8"
    if re.match(r"verify_m[2-8]/begin/", name):
        return name.rsplit("/", 1)[-1] != "k0"
    if re.match(r"mtp_capture_m[2-8]/", name):
        return True
    if not draft:
        return False
    if name.startswith(("flash_draft_m", "mtp_head_m")):
        return True
    if re.match(r"mtp_warm_m[0-9]+/layer48/", name):
        return True
    match = re.fullmatch(r"mtp_warm_m[0-9]+/begin/k([0-9]+)", name)
    return bool(match and int(match[1]) >= 3)


def prune_computation(package, *, draft=True):
    removed = {
        k["name"] for k in package["kernels"] if obsolete_computation(k["name"], draft=draft)
    }
    package["kernels"] = [k for k in package["kernels"] if k["name"] not in removed]
    package["dynamic_batch_kernels"] = [
        k for k in package["dynamic_batch_kernels"] if k["name"] not in removed
    ]
    package["plan_kernel_contracts"] = sorted(set(package["plan_kernel_contracts"]) - removed)
    return removed


class Compiler:
    def __init__(self, origin, root, output, buffers):
        self.reader, self.root, self.output, self.buffers = (
            AssetReader(origin),
            root,
            output,
            buffers,
        )
        self.hosts, self.compiled = {}, {}

    def read(self, asset):
        reader = (
            self.reader if (self.reader.base / asset["file"]).exists() else AssetReader(self.root)
        )
        return reader.read(asset)

    def host(self, kernel):
        identity = kernel["host_abi"]["sha256"]
        if identity not in self.hosts:
            self.hosts[identity] = parse_host(self.read(kernel["host_abi"]).decode())[0]
        return self.hosts[identity]

    def pointers(self, kernel):
        return pointer_bindings(kernel, self.host(kernel), allow_scalars=True)

    def emit(
        self,
        old,
        name,
        bindings,
        rows,
        *,
        build=None,
        key=None,
        multiplier=1,
        symbolic=True,
        constants=None,
        kernel=None,
        artifact=None,
    ):
        if build is None and kernel is None and artifact is None:
            host, assets = self.host(old), {f: old[f] for f in ("module", "source", "host_abi")}
            source = self.read(old["source"]).decode()
        else:
            from kernels.model.rows import explicit_rows
            import tilelang
            from tools.operators.common import export_kernel

            if key not in self.compiled:
                print("COMPILE", key, flush=True)
                passes = {}
                if key.startswith("causal-gdn-"):
                    passes = {
                        "tl.disable_thread_storage_sync": True,
                        "tl.disable_shared_memory_reuse": True,
                    }
                if key.startswith("expert-"):
                    from tools.model.flash_next.decode_tuning import EXACT_PASSES

                    passes = EXACT_PASSES
                if artifact is None and kernel is None:
                    fn = explicit_rows(build()) if symbolic else build()
                    kernel = tilelang.compile(
                        fn,
                        out_idx=[],
                        execution_backend="nvrtc",
                        target={"kind": "cuda", "arch": "sm_87"},
                        pass_configs=passes,
                    )
                folder = artifact or self.output / "aot" / key
                if artifact is None:
                    export_kernel(kernel, folder)
                assets = {}
                for field, filename in [
                    ("module", "kernel.cubin"),
                    ("source", "kernel.cu"),
                    ("host_abi", "host.txt"),
                ]:
                    target = self.root / "shared-rows" / key / filename
                    target.parent.mkdir(parents=True, exist_ok=True)
                    link_or_copy(folder / filename, target)
                    assets[field] = dict(
                        file=str(target.relative_to(self.root)), sha256=file_hash(target)
                    )
                host = parse_host((folder / "host.txt").read_text())[0]
                source = (folder / "kernel.cu").read_text()
                validate_parameter_count(source, host)
                self.compiled[key] = host, assets, source
            host, assets, source = self.compiled[key]
        self.hosts[assets["host_abi"]["sha256"]] = host
        result = replace_binding(
            dict(old, name=name),
            host,
            assets,
            bindings,
            dimensions=dict(m=rows * multiplier, rows=rows * multiplier, **(constants or {})),
        )
        prototype = re.search(r"\b" + re.escape(result["symbol"]) + r"\s*\(([^)]*)\)", source)
        if prototype is None or len(prototype[1].split(",")) != len(result["args"]):
            raise ValueError("Missing or inconsistent shared-row CUDA argument access")
        result["readonly_arguments"] = [
            "*" not in value or bool(re.search(r"\bconst\b", value))
            for value in prototype[1].split(",")
        ]
        launch = contract(result, host, rows, multiplier, constants)
        return result, launch


def target_rows(data, package, compiler):
    """Lift the validated target template's bound, retaining its arithmetic."""
    from kernels.model.e8p_exact import e8p_exact_grouped
    from kernels.model.moe import route_metadata

    buffers = compiler.buffers
    original = {k["name"]: k for k in package["kernels"]}
    entry = original.get("flash_decode/begin/k0")
    if entry is None:
        raise ValueError("Finalize the independent-request decode template first")
    residual = compiler.pointers(entry)["Residual"]["name"]
    aliases = {residual: RESIDUAL}
    layout = data["metadata"]["batch_layout"]
    role = residual.removeprefix("FlashDecode_")
    layout["row_strides"]["Residual"] = layout["row_strides"].pop(role)
    buffers[RESIDUAL] = dict(buffers.pop(residual), name=RESIDUAL)
    data["buffer_scopes"][RESIDUAL] = data["buffer_scopes"].pop(residual)

    fixed, tiles, route_tiles = set(), set(), {}
    body_buffers, head_buffers = set(), set()
    for k in original.values():
        if k["name"].startswith("flash_decode/") and "/packet/" not in k["name"]:
            names = {a["name"] for a in k["args"] if "name" in a}
            (head_buffers if "/head/" in k["name"] else body_buffers).update(names)
    fixed.update(head_buffers - body_buffers)
    for k in original.values():
        if not k["name"].startswith("flash_decode/"):
            continue
        p = compiler.pointers(k)
        if "Counts" in p:
            fixed.update(
                p[a]["name"]
                for a in ("Counts", "Offsets", "RowOffsets", "TileOffsets", "TC", "TileCount")
                if a in p
            )
        tiles.update(p[a]["name"] for a in ("TE", "TR", "TileExpert", "TileRow") if a in p)
        if {"IDs", "Counts", "RelativeSlot", "TileExpert"} <= p.keys():
            source = compiler.reader.read(k["source"]).decode()
            matches = re.findall(r"TileRow\[[^;]+?= \(tile \* (8|16)\)", source)
            if len(set(matches)) != 1:
                raise ValueError("Missing certified expert route tile")
            section = k["name"].rsplit("/k", 1)[0]
            choice = k["name"].split("/rows/", 1)[1] if "/rows/" in k["name"] else None
            route_tiles[section, choice] = int(matches[0])
    for name, b in buffers.items():
        if not name.startswith(
            ("FlashDecode_", "FlashDecodeQ4_", "FlashDecodeE8P_", "FlashDecodeHC")
        ):
            continue
        if b.get("data") or name in fixed:
            continue
        if name in tiles:
            b["shape"] = [(CAPACITY * 10 + 7) // 8 + 512]
        elif name == "FlashDecodeHCPartials":
            b["shape"][1] = CAPACITY
        elif math.prod(b["shape"]) % REQUESTS == 0:
            b["shape"][0] = b["shape"][0] * CAPACITY // REQUESTS
        else:
            raise ValueError(f"Unknown target row workspace geometry: {name}")

    additions, launches = [], []
    for k in package["kernels"]:
        for arg in k["args"]:
            if arg.get("name") in aliases:
                arg["name"] = aliases[arg["name"]]
        if not k["name"].startswith("flash_decode/") or "/packet/" in k["name"]:
            continue
        capacity = REQUESTS if "/head/" in k["name"] else CAPACITY
        name = k["name"].replace("/33-128", f"/33-{capacity}").replace("/1-128", f"/1-{capacity}")
        host = compiler.host(k)
        symbolic = any(
            a["ctype"] == "ctypes.c_int32" and a["value"] in ("m", "rows")
            for a in host["ordered_arguments"]
        )
        p = compiler.pointers(k)
        multiplier, build, key = 1, None, None
        source = compiler.reader.read(k["source"]).decode()
        if symbolic and {"Low", "High", "P", "Counts", "Out"} <= p.keys():
            multiplier = 10
            e, groups, n, words = buffers[p["P"]["name"]]["shape"]
            section = k["name"].rsplit("/k", 1)[0]
            choice = k["name"].split("/rows/", 1)[1] if "/rows/" in k["name"] else None
            bm = route_tiles[section, choice]
            count = (CAPACITY * 10 + bm - 1) // bm + 512

            def build(e=e, n=n, groups=groups, words=words, bm=bm, count=count):
                return e8p_exact_grouped.__wrapped__(
                    CAPACITY * 10, e, count, n, groups * words * 8, block_m=bm, dynamic_rows=True
                )

            key = f"expert-m{bm}-n{n}-k{groups * words * 8}"
        elif symbolic and {"IDs", "Counts", "RelativeSlot", "TileExpert"} <= p.keys():
            matches = re.findall(r"TileRow\[[^;]+?= \(tile \* (8|16)\)", source)
            if len(set(matches)) != 1:
                raise ValueError("Missing certified expert route tile")
            bm = int(matches[0])

            def build(bm=bm):
                return route_metadata.__wrapped__(CAPACITY, block_m=bm, dynamic_rows=True)

            key = f"route-m{bm}"
        elif any(
            a["ctype"] == "ctypes.c_int32"
            and a["value"] in ("m", "rows")
            and k["args"][i].get("value") == REQUESTS * 10
            for i, a in enumerate(host["ordered_arguments"])
        ):
            multiplier = 10
        replacement, launch = compiler.emit(
            k, name, p, capacity, build=build, key=key, multiplier=multiplier
        )
        additions.append(replacement)
        launches.append(launch)
    replaced = {
        x["name"].replace(f"/33-{CAPACITY}", "/33-128").replace(f"/1-{CAPACITY}", "/1-128")
        for x in additions
    }
    package["kernels"] = [k for k in package["kernels"] if k["name"] not in replaced] + additions
    package["dynamic_batch_kernels"] = [
        k for k in package["dynamic_batch_kernels"] if k["name"] not in replaced
    ] + launches
    package["plan_kernel_contracts"] = sorted(
        (set(package["plan_kernel_contracts"]) - replaced) | {k["name"] for k in additions}
    )


def wide_tiles(package, compiler, prefix="flash_decode", body_capacity=CAPACITY):
    """Retain small-row AOT choices; cache BF16 weights across wider rows."""
    from kernels.model.bf16_single import bf16_rows
    from kernels.model.hc_bf16_cache import hc_up13_mix
    from kernels.model.hyperconnection import hc_projection

    additions, contracts, renamed = [], [], {}
    names = {k["name"] for k in package["kernels"]}
    for kernel in list(package["kernels"]):
        if not kernel["name"].startswith(prefix + "/") or "/packet/" in kernel["name"]:
            continue
        p = compiler.pointers(kernel)
        capacity = REQUESTS if "/head/" in kernel["name"] else body_capacity
        if kernel["name"].split("/rows/", 1)[0] + f"/rows/33-{capacity}" in names:
            continue
        if (
            set(p) == {"A", "W", "Out"}
            and "/rows/" not in kernel["name"]
            and p["W"]["name"].endswith("_inject.weight_value")
        ):
            n, k = compiler.buffers[p["W"]["name"]]["shape"]
            if (n, k) != (4, 10240):
                raise ValueError("Invalid HC injection geometry")
            result, launch = compiler.emit(
                kernel,
                kernel["name"] + f"/rows/33-{capacity}",
                dict(A=p["A"], W=p["W"], C=p["Out"]),
                capacity,
                build=lambda capacity=capacity: hc_projection.__wrapped__(
                    capacity, 4, 10240, block_m=16, block_n=16, dtype="float16", dynamic_rows=True
                ),
                key="hc-injection-wide",
            )
        elif kernel["symbol"] == "bf16_single_kernel" and "/rows/" not in kernel["name"]:
            n, total = compiler.buffers[p["Weight"]["name"]]["shape"]
            elements = math.prod(compiler.buffers[p["Output"]["name"]]["shape"])
            groups = 4 if total == 10240 and n in (256, 320) else 1
            if elements < capacity * n * groups:
                raise ValueError("Invalid shared BF16 partial grouping")
            k = total // groups

            def build(n=n, k=k, groups=groups):
                return bf16_rows.__wrapped__(
                    n, k, groups, rows=capacity, block_m=16, dynamic_rows=True, weight_left=False
                )

            result, launch = compiler.emit(
                kernel,
                kernel["name"] + f"/rows/33-{capacity}",
                p,
                capacity,
                build=build,
                key=f"bf16-rows-{n}-{k}-{groups}",
            )
        elif {"A", "Hi", "Lo", "Normed", "Out"} <= p.keys() and (
            "/rows/" not in kernel["name"]
            or kernel["name"].endswith(f"/rows/1-{capacity}")
            or kernel["name"].endswith("/rows/1-32")
        ):
            if "/rows/" not in kernel["name"] and any(
                name.startswith(kernel["name"] + "/rows/") for name in names
            ):
                continue
            source = compiler.read(kernel["source"]).decode()
            bases = set(re.findall(r"hc_expand13_four\([^;]+?,\s*\(uint\)(\d+)\)", source))
            if len(bases) != 1:
                raise ValueError("Missing BF16 up cache exponent")
            base = int(next(iter(bases)))
            rank = compiler.buffers[p["Hi"]["name"]]["shape"][1] * 64
            patches = (
                math.prod(compiler.buffers[p["Patches"]["name"]]["shape"]) if "Patches" in p else 0
            )
            # A declared one-element zero escape placeholder is not a patch.
            patch_calls = re.findall(r"for \(int patch = 0; patch < (\d+)", source)
            # The primitive's source uses a literal patch loop only when needed.
            if (
                "hc_expand13_four" in source
                and "Patches[(" not in source
                and "Patches[" not in source
            ):
                patches = 0
            if patch_calls:
                patches = int(patch_calls[0])

            def build(rank=rank, base=base, patches=patches, capacity=capacity):
                return hc_up13_mix.__wrapped__(
                    capacity,
                    2560,
                    rank,
                    dynamic_rows=True,
                    base_exp=base,
                    patch_count=patches,
                    block_m=32,
                    weight_left=False,
                )

            old_name = kernel["name"]
            if old_name.endswith("/rows/1-32"):
                variant = old_name.replace("/1-32", f"/33-{capacity}")
            elif "/rows/" in old_name:
                kernel["name"] = old_name.replace(f"/1-{capacity}", "/1-32")
                renamed[old_name] = kernel["name"]
                variant = old_name.replace(f"/1-{capacity}", f"/33-{capacity}")
            else:
                variant = old_name + f"/rows/33-{capacity}"
            result, launch = compiler.emit(
                kernel,
                variant,
                p,
                capacity,
                build=build,
                key=f"hc13-wide-{rank}-{base}-{patches}",
            )
        else:
            continue
        additions.append(result)
        contracts.append(launch)
    for launch in package["dynamic_batch_kernels"]:
        launch["name"] = renamed.get(launch["name"], launch["name"])
    package["kernels"].extend(additions)
    package["dynamic_batch_kernels"].extend(contracts)
    package["plan_kernel_contracts"] = sorted(
        {renamed.get(n, n) for n in package["plan_kernel_contracts"]}
        | {k["name"] for k in additions}
    )


def verification_state(package, compiler):
    """Rebind causal verification and capture to the common target row frame."""
    additions = []
    for kernel in list(package["kernels"]):
        if kernel["name"].startswith("flash_verify_segment_m"):
            clone = copy.deepcopy(kernel)
            clone["name"] = clone["name"].replace("flash_verify_segment_m", "verify_rows_m", 1)
        elif kernel["name"].startswith("flash_verify_capture_m"):
            clone = copy.deepcopy(kernel)
            clone["name"] = clone["name"].replace(
                "flash_verify_capture_m", "verify_rows_capture_m", 1
            )
        else:
            continue
        for arg in clone["args"]:
            name = arg.get("name", "")
            common = (
                RESIDUAL
                if name == "FlashMixedResidual"
                else name.replace("FlashMixed_", "FlashDecode_", 1)
            )
            if common in compiler.buffers:
                arg["name"] = common
        additions.append(clone)
    package["kernels"].extend(additions)
    package["plan_kernel_contracts"] = sorted(
        set(package["plan_kernel_contracts"]) | {k["name"] for k in additions}
    )


def wide_integer_tiles(package, compiler):
    """Bulk rows reuse projection primitives with weight-reusing matrix tiles."""
    from kernels.model.q4_projection import q4_decode
    from kernels.model.int8_projection import int8_projection
    from kernels.model.e8p_prefill import e8p_prefill_grouped
    from kernels.model.moe import route_metadata

    additions, contracts, renamed = [], [], {}
    for kernel in package["kernels"]:
        if (
            not kernel["name"].startswith(("flash_decode/", "draft_rows/"))
            or "/head/" in kernel["name"]
        ):
            continue
        match = re.search(r"/rows/33-(512|1024)$", kernel["name"])
        if not match:
            continue
        capacity = int(match[1])
        p = compiler.pointers(kernel)
        multiplier = 1
        if set(p) == {"A", "W", "S", "AS", "Out"}:
            n, words = compiler.buffers[p["W"]["name"]]["shape"]
            k = words * 4
            dtype = {"f16": "float16", "f32": "float32"}[
                compiler.buffers[p["Out"]["name"]]["dtype"]
            ]

            def build(n=n, k=k, dtype=dtype, capacity=capacity):
                return q4_decode(
                    capacity,
                    n,
                    k,
                    algorithm="tensor_core",
                    block_m=64,
                    block_n=128,
                    output_dtype=dtype,
                    dynamic_rows=True,
                )

            key = f"q4-bulk-{n}-{k}-{dtype}"
        elif set(p) == {"A", "Weight", "WeightScale", "TokenScale", "Output"}:
            n, k = compiler.buffers[p["Weight"]["name"]]["shape"]
            dtype = {"f16": "float16", "f32": "float32"}[
                compiler.buffers[p["Output"]["name"]]["dtype"]
            ]

            def build(n=n, k=k, dtype=dtype, capacity=capacity):
                return int8_projection.__wrapped__(
                    capacity,
                    n,
                    k,
                    dtype,
                    block_m=64,
                    block_n=64 if n < 128 else 128,
                    block_k=128,
                    dynamic_rows=True,
                )

            key = f"w8-bulk-{n}-{k}-{dtype}"
        elif {"Low", "High", "P", "Counts", "Out"} <= p.keys():
            multiplier = 10
            e, groups, n, words = compiler.buffers[p["P"]["name"]]["shape"]
            k = groups * words * 8
            tiles = (capacity * 10 + 15) // 16 + 512

            def build(e=e, n=n, k=k, tiles=tiles, capacity=capacity):
                return e8p_prefill_grouped.__wrapped__(
                    capacity * 10, e, tiles, n, k, dynamic_rows=True, row_major=True
                )

            key = f"expert-bulk-pipeline-n{n}-k{k}"
        elif {"IDs", "Counts", "RelativeSlot", "TileExpert"} <= p.keys():

            def build(capacity=capacity):
                return route_metadata.__wrapped__(capacity, block_m=16, dynamic_rows=True)

            key = f"route-bulk-m16-{capacity}"
        else:
            continue
        old_name = kernel["name"]
        result, launch = compiler.emit(
            kernel,
            old_name.replace("/33-", "/129-"),
            p,
            capacity,
            build=build,
            key=key,
            multiplier=multiplier,
        )
        kernel["name"] = old_name.replace(f"/33-{capacity}", "/33-128")
        renamed[old_name] = kernel["name"]
        additions.append(result)
        contracts.append(launch)
    for launch in package["dynamic_batch_kernels"]:
        launch["name"] = renamed.get(launch["name"], launch["name"])
    package["kernels"].extend(additions)
    package["dynamic_batch_kernels"].extend(contracts)
    package["plan_kernel_contracts"] = sorted(
        {renamed.get(n, n) for n in package["plan_kernel_contracts"]}
        | {k["name"] for k in additions}
    )


def wide_raw_hc(package, compiler):
    """Use the resident BF16 HC weight for bulk rows, without unpacking a cache."""
    from kernels.model.hyperconnection import hc_up_mix

    additions, contracts, renamed = [], [], {}
    for kernel in package["kernels"]:
        match = re.search(r"/rows/33-(512|1024)$", kernel["name"])
        if not match or not kernel["name"].startswith(("flash_decode/", "draft_rows/")):
            continue
        p = compiler.pointers(kernel)
        if not {"A", "Hi", "Lo", "Normed", "Out"} <= p.keys():
            continue
        capacity = int(match[1])
        hi = p["Hi"]["name"]
        weight = hi.removeprefix("HC13_").removesuffix("_Hi")
        if weight not in compiler.buffers:
            raise ValueError(f"Missing resident HC weight: {weight}")
        rank = compiler.buffers[weight]["shape"][1]
        bindings = {n: p[n] for n in ("A", "Normed", "Out")}
        bindings["W"] = dict(p["Hi"], name=weight)

        def build(rank=rank, capacity=capacity):
            return hc_up_mix.__wrapped__(capacity, 2560, rank, dynamic_rows=True, block_m=128)

        old_name = kernel["name"]
        result, launch = compiler.emit(
            kernel,
            old_name.replace("/33-", "/129-"),
            bindings,
            capacity,
            build=build,
            key=f"hc-raw-bulk-{rank}",
        )
        kernel["name"] = old_name.replace(f"/33-{capacity}", "/33-128")
        renamed[old_name] = kernel["name"]
        additions.append(result)
        contracts.append(launch)
    for launch in package["dynamic_batch_kernels"]:
        launch["name"] = renamed.get(launch["name"], launch["name"])
    package["kernels"].extend(additions)
    package["dynamic_batch_kernels"].extend(contracts)
    package["plan_kernel_contracts"] = sorted(
        {renamed.get(n, n) for n in package["plan_kernel_contracts"]}
        | {k["name"] for k in additions}
    )


def upgrade(model, destination, output, *, tuning_cache=None):
    from tools.operators.common import configure

    configure()
    data, origin, package = load_model(model)
    root = clone_model(model, destination, origin)
    buffers = {b["name"]: copy.deepcopy(b) for b in data["metadata"]["buffers"]}
    compiler = Compiler(origin, root, output, buffers)
    draft = data["metadata"].get("mtp") is not None
    compiler.tuning_cache = tuning_cache
    target_rows(data, package, compiler)
    wide_tiles(package, compiler)
    verification_state(package, compiler)
    if draft:
        from tools.model.flash_next.draft_rows import publish

        publish(data, package, compiler)
        wide_tiles(package, compiler, "draft_rows", 512)
    wide_integer_tiles(package, compiler)
    wide_raw_hc(package, compiler)
    removed = prune_computation(package, draft=draft)
    state_removed = 0
    if draft:
        from tools.model.flash_next.causal_rows import publish as causal_states

        state_removed = causal_states(data, package, compiler)
    data["metadata"]["buffers"] = list(buffers.values())
    package["buffer_contracts"] = [
        {k: v for k, v in b.items() if k != "data"} for b in buffers.values()
    ]
    digest = commit_package(destination, root, data, package)
    write_json(
        output / "results.json",
        dict(
            capacity=CAPACITY,
            compiled=len(compiler.compiled),
            removed=len(removed) + state_removed,
            execution_package=digest,
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--model-output", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--compile-cache", type=Path)
    p.add_argument("--tuning-cache", type=Path)
    args = p.parse_args()
    if args.compile_cache:
        from tools.model.publication import seed_compile_cache

        seed_compile_cache(args.compile_cache, args.output / "cache/0.1.15")
    with atomic_model(args.model_output) as staging:
        upgrade(
            args.model.resolve(),
            staging,
            args.output,
            tuning_cache=args.tuning_cache,
        )


if __name__ == "__main__":
    main()
