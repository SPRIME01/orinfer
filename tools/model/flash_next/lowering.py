"""Lower shared Flash operators by operands, independent of program names.

The registered model recipe owns causality and section boundaries. This pass
owns selected weight representations and their prerequisite kernels. Fixed,
symbolic and mixed bodies all pass through the same lowering before sealing.
"""

import argparse
import math
from pathlib import Path

from tools.model.flash_next.kernel_policy import expert_implementation, projection_policies
from tools.model.flash_next.optimize_decode import pointer_bindings, replace_binding
from tools.model.flash_next.optimize_experts import row_tile
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


def operand_key(argument):
    return argument["name"], argument.get("offset", 0)


def lower(data, package, operator, output):
    from kernels.model.e8p_exact import e8p_exact_grouped, split_a8
    from kernels.model.bf16_single import bf16_single
    from tools.model.flash_next.decode_tuning import DecodeTuner
    from kernels.model.rotation_a8 import rotate_activation_a8
    from kernels.model.rows import explicit_rows
    from tools.model.upgrade_dynamic_batch import contract
    from tools.operators.common import export_kernel
    import tilelang

    reader, hosts, compiled = AssetReader(operator), {}, {}
    tuner = DecodeTuner(output / "integer-tuning")
    buffers = {b["name"]: b for b in data["metadata"]["buffers"]}
    policies = projection_policies(buffers)
    dynamics = {d["name"]: d for d in package.get("dynamic_batch_kernels", [])}
    plan_contracts = set(package.get("plan_kernel_contracts", []))
    additions, changed, prerequisites = [], [], {}

    def host(kernel):
        digest = kernel["host_abi"]["sha256"]
        if digest not in hosts:
            hosts[digest] = parse_host(reader.read(kernel["host_abi"]).decode())[0]
        return hosts[digest]

    def pointers(kernel):
        return pointer_bindings(kernel, host(kernel), allow_scalars=True)

    exact_books = {}
    for kernel in package["kernels"]:
        if kernel["symbol"] in ("exact_e8p_int4_kernel", "exact_e8p_shared_input_kernel"):
            p = pointers(kernel)
            exact_books[p["P"]["name"]] = p["Book"]["name"]
        if kernel["name"].endswith(("/q4_rotation", "/a8_split")):
            plan_contracts.update((kernel["name"], kernel["name"].rsplit("/", 1)[0]))

    def compile_kernel(key, factory, symbolic=False):
        if key not in compiled:
            print("COMPILE", key, flush=True)
            kernel = factory()
            if symbolic and not hasattr(kernel, "adapter"):
                kernel = tilelang.compile(
                    explicit_rows(kernel),
                    out_idx=[],
                    execution_backend="nvrtc",
                    target={"kind": "cuda", "arch": "sm_87"},
                )
            exported = output / "aot" / key
            export_kernel(kernel, exported)
            assets = {}
            for field, filename in (
                ("module", "kernel.cubin"),
                ("source", "kernel.cu"),
                ("host_abi", "host.txt"),
            ):
                path = operator / "compute-policy" / key / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                # Cloned assets may share an inode with a frozen package.
                # Replace its link, never overwrite that shared inode.
                path.unlink(missing_ok=True)
                link_or_copy(exported / filename, path)
                assets[field] = dict(file=str(path.relative_to(operator)), sha256=file_hash(path))
            abi = parse_host((exported / "host.txt").read_text())[0]
            validate_parameter_count((exported / "kernel.cu").read_text(), abi)
            compiled[key] = abi, assets
        return compiled[key]

    def scratch(name, dtype, shape):
        expected = dict(
            name=name,
            dtype=dtype,
            shape=shape,
            layout="native_contiguous",
            alignment=256,
            access="read_write",
        )
        if name not in buffers:
            buffers[name] = expected
            data["metadata"]["buffers"].append(expected)
            data["buffer_scopes"][name] = "workspace"
        elif any(buffers[name].get(k) != v for k, v in expected.items()):
            raise ValueError(f"Incompatible compute workspace: {name}")
        return dict(kind="buffer", name=name)

    def bind(original, key, factory, bound, rows, symbolic=False):
        abi, assets = compile_kernel(key, factory, symbolic)
        k = replace_binding(original, abi, assets, bound, dimensions=dict(m=rows, rows=rows))
        if symbolic:
            dynamics[k["name"]] = contract(k, abi, rows)
        return k

    # Quantization nodes identify FP16 inputs and exact live row extents even
    # when a profile is a slice of a larger persistent arena.
    quantizers = {}
    for k in package["kernels"]:
        p = pointers(k)
        if set(p) == {"Q", "S", "X"} and buffers[p["Q"]["name"]]["dtype"] == "i8":
            key = (k["name"].rsplit("/", 1)[0], operand_key(p["Q"]), operand_key(p["S"]))
            quantizers.setdefault(key, []).append((k, p))

    for position, original in enumerate(package["kernels"]):
        p = pointers(original)
        down = set(p) == {"A", "Partials", "W"}
        router = set(p) == {"A", "Output", "Weight"}
        if down or router:
            weight = p["W" if down else "Weight"]
            w = buffers[weight["name"]]
            if w["dtype"] == "bf16" and (
                (down and w["shape"] in ([320, 10240], [256, 10240]))
                or (router and w["shape"] == [512, 2560])
            ):
                groups = 4 if down else 1
                n = w["shape"][0]
                out = p["Partials" if down else "Output"]
                rows = math.prod(buffers[out["name"]]["shape"]) // (groups * n)
                symbolic = original["name"] in dynamics
                if symbolic:
                    rows = dynamics[original["name"]]["capacity"]
                if rows <= 8 and original["symbol"] != "bf16_single_kernel":
                    key = f"bf16-r{rows}-n{n}-g{groups}-d{int(symbolic)}"

                    def factory():
                        return (
                            bf16_single.__wrapped__(n, 2560, groups, rows=rows, dynamic_rows=True)
                            if symbolic
                            else bf16_single(n, 2560, groups, rows=rows)
                        )

                    package["kernels"][position] = bind(
                        original,
                        key,
                        factory,
                        dict(A=p["A"], Weight=weight, Output=out),
                        rows,
                        symbolic,
                    )
                    changed.append(dict(name=original["name"], strategy="bf16_direct", rows=rows))
        selected = [a["name"] for a in original["args"] if a.get("name") in policies]
        if selected:
            if len(selected) != 1:
                raise ValueError("Ambiguous selected projection")
            policy = policies[selected[0]]
            p = pointers(original)
            if set(p) != {"A", "Weight", "WeightScale", "TokenScale", "Output"}:
                raise ValueError(f"Unsupported selected projection ABI: {original['name']}")
            program = original["name"].split("/")[0]
            scope, slot = original["name"].rsplit("/", 1)
            producers = quantizers[(scope, operand_key(p["A"]), operand_key(p["TokenScale"]))]
            preceding = [
                (k, q)
                for k, q in producers
                if int(k["name"].rsplit("/k", 1)[1]) < int(slot.removeprefix("k"))
            ]
            if not preceding:
                raise ValueError(f"Missing preceding quantizer: {original['name']}")
            quant, q = max(preceding, key=lambda item: int(item[0]["name"].rsplit("/k", 1)[1]))
            rows = quant["grid"][0]
            symbolic = original["name"] in dynamics
            if symbolic:
                rows = dynamics[original["name"]]["capacity"]
            identity = (program, operand_key(q["X"]), policy.signs, rows, symbolic)
            # QKV and Z share the same input transform. Reuse its workspace
            # and execute the transform once, immediately before its first use.
            if identity not in prerequisites:
                # Layer programs run in order. Storage belongs to the shape,
                # not each layer/consumer; captures keep these addresses stable.
                prefix = f"ComputeQ4_R{rows}_K{policy.inputs}"
                a = scratch(prefix + "_Activation", "i8", [rows, policy.inputs])
                s = scratch(prefix + "_Scale", "f16", [rows])
                name = original["name"] + "/q4_rotation"
                key = f"rotate-r{rows}-k{policy.inputs}-d{int(symbolic)}"

                def factory():
                    return (
                        rotate_activation_a8.__wrapped__(rows, policy.inputs, dynamic_rows=True)
                        if symbolic
                        else rotate_activation_a8(rows, policy.inputs)
                    )

                rotation = bind(
                    dict(name=name),
                    key,
                    factory,
                    dict(X=q["X"], Signs=dict(kind="buffer", name=policy.signs), Q=a, S=s),
                    rows,
                    symbolic,
                )
                additions.append(rotation)
                prerequisites[identity] = a, s
                plan_contracts.update((name, original["name"]))
            a, s = prerequisites[identity]
            dtype = {"f16": "float16", "f32": "float32"}[buffers[p["Output"]["name"]]["dtype"]]
            kernel, config = tuner.compile(
                "q4", rows, policy.outputs, policy.inputs, dtype, dynamic=symbolic
            )
            key = f"q4-r{rows}-n{policy.outputs}-k{policy.inputs}-{dtype}-d{int(symbolic)}"

            def factory(kernel=kernel):
                return kernel

            bound = dict(
                A=a,
                AS=s,
                W=dict(kind="buffer", name=policy.packed),
                S=dict(kind="buffer", name=policy.scale),
                Out=p["Output"],
            )
            package["kernels"][position] = bind(original, key, factory, bound, rows, symbolic)
            changed.append(
                dict(name=original["name"], strategy="q4a8", rows=rows, dynamic=symbolic)
            )

        packed = next(
            (a for a in original["args"] if a.get("name", "").endswith("_exps.weight_packed")), None
        )
        if packed is None or original["symbol"] in (
            "exact_e8p_int4_kernel",
            "exact_e8p_shared_input_kernel",
        ):
            continue
        p = pointers(original)
        if set(p) != {
            "A",
            "AS",
            "Book",
            "C",
            "Counts",
            "Offsets",
            "P",
            "TileCount",
            "TileExpert",
            "TileRow",
            "WS",
        }:
            raise ValueError(f"Unsupported grouped expert ABI: {original['name']}")
        e, groups, n, words = buffers[packed["name"]]["shape"]
        k = groups * words * 8
        assignments = math.prod(buffers[p["AS"]["name"]]["shape"])
        if assignments % 10:
            raise ValueError("Invalid routed row capacity")
        # Certification follows the actual bank, not a global LUT shape.
        book = exact_books.get(packed["name"])
        if (
            expert_implementation(assignments // 10, words * 8, nibble_book=book is not None)
            != "exact_a8"
        ):
            continue
        bm = row_tile(reader.read(original["source"]).decode())
        if bm not in (8, 16):
            # The coupled route/expert tuner installs a legal exact tile later.
            continue
        tiles = math.prod(buffers[p["TileExpert"]["name"]]["shape"])
        prefix = f"ComputeE8P_R{assignments}_K{k}"
        args = {
            role: scratch(prefix + "_" + role, dtype, shape)
            for role, dtype, shape in (
                ("Low", "u32", [assignments, k // 8]),
                ("High", "u32", [assignments, k // 8]),
                ("Sum", "i32", [assignments]),
            )
        }
        split = bind(
            dict(name=original["name"] + "/a8_split"),
            f"split-r{assignments}-k{k}",
            lambda: split_a8(assignments, k),
            dict(A=p["A"], **args),
            assignments,
        )
        additions.append(split)
        bound = dict(
            **args,
            P=p["P"],
            Book=dict(kind="buffer", name=book),
            WS=p["WS"],
            AS=p["AS"],
            Counts=p["Counts"],
            Offsets=p["Offsets"],
            TE=p["TileExpert"],
            TR=p["TileRow"],
            TC=p["TileCount"],
            Out=p["C"],
        )
        package["kernels"][position] = bind(
            original,
            f"exact-r{assignments}-bm{bm}-n{n}-k{k}-t{tiles}",
            lambda: e8p_exact_grouped(assignments, e, tiles, n, k, block_m=bm),
            bound,
            assignments,
        )
        # Planes are capacity-bounded; routing counts exclude inactive rows.
        if original["name"] in dynamics:
            for kernel in (split, package["kernels"][position]):
                dynamics[kernel["name"]] = dict(
                    name=kernel["name"],
                    capacity=assignments // 10,
                    grid=[dict(op="constant", value=v) for v in kernel["grid"]],
                    arguments=[],
                )
        plan_contracts.update((split["name"], original["name"]))
        changed.append(dict(name=original["name"], strategy="exact_e8p_a8", rows=assignments // 10))

    package["kernels"].extend(additions)
    package["dynamic_batch_kernels"] = list(dynamics.values())
    package["plan_kernel_contracts"] = sorted(plan_contracts)
    package["buffer_contracts"] = [
        {k: v for k, v in b.items() if k != "data"} for b in data["metadata"]["buffers"]
    ]
    write_json(
        output / "compute-policy.json",
        dict(changed=changed, compiled=len(compiled), prerequisites=len(additions)),
    )


def upgrade(model, destination, output):
    import shutil
    from tools.model.compact import compact_package
    from tools.model.package import model_library
    from tools.operators.common import configure

    configure()
    data, origin, package = load_model(model)
    if data["architecture"] != "flash_next" or data["compute_policy"] != "int8_quality":
        raise ValueError("Expected Flash Next INT8 quality execution")
    operator = clone_model(model, destination, origin)
    lower(data, package, operator, output)
    path = operator / package["execution"]["library"]["file"]
    path.unlink()
    shutil.copyfile(model_library(), path)
    package["execution"]["library"]["sha256"] = file_hash(path)
    digest = compact_package(operator, package, destination / "cache/packages")
    shutil.rmtree(operator)
    data["execution_package"] = digest
    write_json(destination / "cache/model.json", data)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--model-output", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--compile-cache", type=Path)
    a = p.parse_args()
    if a.compile_cache:
        from tools.model.publication import seed_compile_cache

        seed_compile_cache(a.compile_cache, a.output / "cache/0.1.15")
    with atomic_model(a.model_output) as staging:
        upgrade(a.model.resolve(strict=True), staging, a.output)


if __name__ == "__main__":
    main()
