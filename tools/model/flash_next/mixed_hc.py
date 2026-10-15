"""Reuse BF16 HC prefill operands in concatenated Flash row programs."""

import argparse
import copy
from pathlib import Path

from tools.model.publication import (
    AssetReader,
    atomic_model,
    clone_model,
    commit_package,
    load_model,
    file_hash,
    link_or_copy,
)
from tools.model.flash_next.compiler import pointer_bindings, replace_binding
from tools.model.upgrade_dynamic_batch import contract
from tools.operators.abi import parse_host


def reduction_groups(n, k):
    if (n, k) == (320, 10240):
        return 4
    if (n, k) == (10240, 320):
        return 1
    raise ValueError("Unexpected joined HC projection")


def upgrade(model, destination, output):
    import tilelang
    from kernels.model.hc_prefill import hc_input_bf16, hc_projection_prefill
    from kernels.model.rows import explicit_rows
    from tools.operators.common import configure, export_kernel

    configure()
    data, origin, package = load_model(model)
    if data["architecture"] != "flash_next" or not data["metadata"]["batch_layout"]["profiles"]:
        raise ValueError("Expected joined Flash row execution")
    operator = clone_model(model, destination, origin)
    buffers = {b["name"]: b for b in data["metadata"]["buffers"]}
    reader = AssetReader(origin)
    assets = {}
    changes = []
    added = []
    dynamic = {k["name"]: k for k in package["dynamic_batch_kernels"]}
    original = {k["name"]: k for k in package["kernels"]}

    def compile_kernel(key, fn):
        if key not in assets:
            kernel = tilelang.compile(
                explicit_rows(fn),
                out_idx=[],
                execution_backend="nvrtc",
                target={"kind": "cuda", "arch": "sm_87"},
            )
            export_kernel(kernel, output / "aot" / key)
            identity = {}
            for field, filename in [
                ("module", "kernel.cubin"),
                ("source", "kernel.cu"),
                ("host_abi", "host.txt"),
            ]:
                src = output / "aot" / key / filename
                dst = operator / "mixed-hc" / key / filename
                dst.parent.mkdir(parents=True, exist_ok=True)
                link_or_copy(src, dst)
                identity[field] = dict(file=str(dst.relative_to(operator)), sha256=file_hash(dst))
            assets[key] = (parse_host((output / "aot" / key / "host.txt").read_text())[0], identity)
        return assets[key]

    for index, kernel in enumerate(package["kernels"]):
        if not kernel["name"].startswith("flash_mixed_m"):
            continue
        if not any(
            a.get("name", "").endswith(
                (
                    ".hc_attn_down.weight_value",
                    ".hc_ffn_down.weight_value",
                    ".hc_attn_up.weight_value",
                    ".hc_ffn_up.weight_value",
                )
            )
            for a in kernel["args"]
        ):
            continue
        host = parse_host(reader.read(kernel["host_abi"]).decode())[0]
        pointers = pointer_bindings(kernel, host, allow_scalars=True)
        if set(pointers) != {"A", "W", "C"}:
            continue
        w = buffers[pointers["W"]["name"]]
        if not w["name"].endswith(
            (
                ".hc_attn_down.weight_value",
                ".hc_ffn_down.weight_value",
                ".hc_attn_up.weight_value",
                ".hc_ffn_up.weight_value",
            )
        ):
            continue
        n, k = w["shape"]
        rows = int(kernel["name"].split("/")[0].rsplit("m", 1)[1])
        dimensions = dict(m=rows, M=rows, rows=rows, batch=rows)
        if w["dtype"] != "bf16" or (n, k) not in ((320, 10240), (10240, 320)):
            raise ValueError("Unexpected joined HC projection")
        role = f"FlashMixedHC_M{rows}_K{k}"
        if role not in buffers:
            b = dict(
                name=role,
                dtype="bf16",
                shape=[rows, k],
                layout="native_contiguous",
                alignment=256,
                access="read_write",
            )
            buffers[role] = b
            data["metadata"]["buffers"].append(b)
            data["buffer_scopes"][role] = "workspace"
            package["buffer_contracts"].append(copy.deepcopy(b))
        cast_name = kernel["name"] + "/bf16_input"
        if cast_name not in original:
            if buffers[pointers["A"]["name"]]["dtype"] != "f16":
                raise ValueError("HC conversion requires the original FP16 operand")
            cast_host, cast_assets = compile_kernel(
                f"cast-m{rows}-k{k}", hc_input_bf16.__wrapped__(rows, k, True)
            )
            cast = replace_binding(
                dict(name=cast_name),
                cast_host,
                cast_assets,
                dict(Input=pointers["A"], Output=dict(kind="buffer", name=role)),
                dimensions=dimensions,
            )
            added.append(cast)
            dynamic[cast_name] = contract(cast, cast_host, rows)
        elif pointers["A"] != dict(kind="buffer", name=role):
            raise ValueError("Existing HC operand cast/consumer differs")
        large = rows > 128
        bm = 128 if large and n > 320 else 64 if rows >= 64 else 16
        bn = 128 if large else 64
        gm = 4 if large else 0
        project_host, project_assets = compile_kernel(
            f"project-m{rows}-n{n}-k{k}-g{reduction_groups(n, k)}",
            hc_projection_prefill.__wrapped__(
                rows,
                n,
                k,
                block_m=bm,
                block_n=bn,
                dtype="float16",
                group_m=gm,
                silu=large and n == 320,
                dynamic_rows=True,
                reduction_groups=reduction_groups(n, k),
            ),
        )
        replacement = replace_binding(
            kernel,
            project_host,
            project_assets,
            dict(A=dict(kind="buffer", name=role), W=pointers["W"], C=pointers["C"]),
            dimensions=dimensions,
        )
        package["kernels"][index] = replacement
        dynamic[kernel["name"]] = contract(replacement, project_host, rows)
        changes.append(kernel["name"])
    if len(changes) != 5 * 48 * 4:
        raise ValueError(f"Incomplete joined HC conversion: {len(changes)}")
    package["kernels"].extend(added)
    package["dynamic_batch_kernels"] = list(dynamic.values())
    package["plan_kernel_contracts"] = sorted(
        set(package.get("plan_kernel_contracts", [])) | {k["name"] for k in added} | set(changes)
    )
    for profile in data["metadata"]["prefill_batch_profiles"]:
        profile["kind"] = "flash_mixed_grouped"
    package["prefill_batch_profiles"] = copy.deepcopy(data["metadata"]["prefill_batch_profiles"])
    commit_package(destination, operator, data, package)
    print("HC OPERANDS READY", len(changes), len(assets), flush=True)


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
        upgrade(a.model.resolve(), staging, a.output)


if __name__ == "__main__":
    main()
