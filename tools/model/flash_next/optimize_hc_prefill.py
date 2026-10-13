"""Compile HC prefill with reusable BF16 operands and a fused up/mix epilogue."""

import argparse
import copy
import json
import shutil
import subprocess
from pathlib import Path
from tools.model.publication import (
    ROOT,
    load_model,
    clone_model,
    AssetReader,
    atomic_model,
    link_or_copy,
    file_hash,
    write_json,
)
from tools.model.compact import compact_package
from tools.model.flash_next.optimize_decode import pointer_bindings, replace_binding
from tools.operators.abi import parse_host, validate_parameter_count


def upgrade(model, destination, report, engine=None):
    from kernels.model.hc_prefill import (
        hc_norm_prefill as hc_norm_bf_only,
        hc_combine_norm_prefill as hc_combine_norm_bf_only,
        hc_projection_prefill as projection_bf16_input,
        hc_up_mix_prefill as up_mix_prefill,
    )
    from tools.model.package import model_library
    from tools.operators.common import configure, export_kernel

    configure()
    cli = engine if engine is not None else ROOT / "target/release/orinfer"
    plan = json.loads(
        subprocess.check_output(
            [str(cli.resolve(strict=True)), "plan-model", str(model)], text=True
        )
    )["manifest"]
    operations = plan["programs"]["prefill_m512"]
    data, origin, package = load_model(model)
    if data["architecture"] != "flash_next" or data["compute_policy"] != "int8_quality":
        raise ValueError("Expected native Flash Next INT8 quality model")
    operator = clone_model(model, destination, origin)
    reader = AssetReader(origin)
    buffers = {b["name"]: b for b in data["metadata"]["buffers"]}
    specs = {k["name"]: k for k in package["kernels"]}
    roles = {}
    variants = {}
    groups = []
    changes = []

    def pointers(name):
        if name not in roles:
            roles[name] = pointer_bindings(
                specs[name], parse_host(reader.read(specs[name]["host_abi"]).decode())[0]
            )
        return copy.deepcopy(roles[name])

    def add(name, dt, shape):
        if name in buffers:
            raise ValueError(f"Duplicate HC prefill workspace: {name}")
        c = dict(
            name=name,
            dtype=dt,
            shape=shape,
            layout="native_contiguous",
            alignment=256,
            access="read_write",
        )
        data["metadata"]["buffers"].append(copy.deepcopy(c))
        data["buffer_scopes"][name] = "workspace"
        package["buffer_contracts"].append(c)
        buffers[name] = data["metadata"]["buffers"][-1]
        return dict(kind="buffer", name=name)

    B = add("HC_M512_NormBF", "bf16", [512, 4, 2560])
    D = add("HC_M512_ActivationBF", "bf16", [512, 320])
    Inv = add("HC_M512_InvRms", "f32", [512, 4])

    def replace(name, key, build, args):
        if key not in variants:
            export_kernel(build(), report / "aot" / key)
            assets = {}
            target = operator / "hc-bf16-prefill" / key
            for field, filename in [
                ("module", "kernel.cubin"),
                ("source", "kernel.cu"),
                ("host_abi", "host.txt"),
            ]:
                path = target / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                link_or_copy(report / "aot" / key / filename, path)
                assets[field] = dict(file=str(path.relative_to(operator)), sha256=file_hash(path))
            h = parse_host((target / "host.txt").read_text())[0]
            validate_parameter_count((target / "kernel.cu").read_text(), h)
            variants[key] = (h, assets)
        h, assets = variants[key]
        replacement = replace_binding(specs[name], h, assets, args)
        index = next(i for i, k in enumerate(package["kernels"]) if k["name"] == name)
        package["kernels"][index] = replacement
        changes.append(name)

    for index, op in enumerate(operations):
        if op.get("kind") != "kernel":
            continue
        name = op["name"]
        spec = specs[name]
        ws = [
            b
            for b in spec["args"]
            if b.get("name", "").endswith(
                (".hc_attn_norm.weight_value", ".hc_ffn_norm.weight_value")
            )
        ]
        if not ws:
            continue
        if spec["grid"] != [512, 4, 1]:
            raise ValueError(f"Unexpected HC prefill norm geometry: {name}")
        names = [operations[index + i]["name"] for i in range(5)]
        assert all(operations[index + i]["kind"] == "kernel" for i in range(5))
        norm, down, up, mix, inject = [pointers(n) for n in names]
        combined = "Block" in norm
        assert set(norm) == (
            {"Block", "Residual", "Inject", "Weight", "Output", "Normed"}
            if combined
            else {"Residual", "Weight", "Normed"}
        )
        assert buffers[norm["Weight"]["name"]]["shape"] == [4, 2560]
        assert (
            down["A"] == norm["Normed"]
            and up["A"] == down["C"]
            and mix["Normed"] == norm["Normed"]
            and mix["Up"] == up["C"]
            and inject["A"] == norm["Normed"]
        )
        assert (
            buffers[down["W"]["name"]]["shape"] == [320, 10240]
            and buffers[up["W"]["name"]]["shape"] == [10240, 320]
            and buffers[inject["W"]["name"]]["shape"] == [4, 10240]
        )
        residual = norm["Output"] if combined else norm["Residual"]
        normalized = norm.pop("Normed")
        norm.update(NormedBF=B, InvRms=Inv)
        factory = hc_combine_norm_bf_only if combined else hc_norm_bf_only
        replace(
            names[0],
            "combine-norm" if combined else "norm",
            lambda factory=factory: factory(512, 2560, dtype="float16"),
            norm,
        )
        replace(
            names[1],
            "down",
            lambda: projection_bf16_input(
                512,
                320,
                10240,
                dtype="float16",
                block_m=64,
                block_n=128,
                group_m=4,
                silu=True,
                output_dtype="bfloat16",
                reduction_groups=4,
            ),
            dict(A=B, W=down["W"], C=D),
        )
        replace(
            names[2],
            "up-mix",
            lambda: up_mix_prefill(512, 2560, 320),
            dict(
                A=D,
                W=up["W"],
                Residual=residual,
                Weight=norm["Weight"],
                InvRms=Inv,
                Out=mix["Mixed"],
            ),
        )
        replace(
            names[4],
            "inject",
            lambda: projection_bf16_input(
                512, 4, 10240, dtype="float16", block_m=16, block_n=16, stages=8
            ),
            dict(A=B, W=inject["W"], C=inject["C"]),
        )
        groups.append(dict(names=names, normalized_original=normalized, up_original=up["C"]))
    if len(groups) != 96:
        raise ValueError(f"Expected 96 HC prefill groups, got {len(groups)}")
    package["plan_kernel_contracts"] = sorted(
        set(package.get("plan_kernel_contracts", []))
        | {name for group in groups for name in group["names"]}
    )
    library = operator / package["execution"]["library"]["file"]
    library.unlink()
    shutil.copyfile(model_library(), library)
    package["execution"]["library"]["sha256"] = file_hash(library)
    digest = compact_package(operator, package, destination / "cache/packages")
    shutil.rmtree(operator)
    data["execution_package"] = digest
    write_json(destination / "cache/model.json", data)
    write_json(
        report / "results.json",
        dict(
            complete=True,
            model=str(destination),
            groups=groups,
            changed=changes,
            compiled_variants=len(variants),
            temporary_unused_mix_slots=False,
            new_workspace_bytes=512 * 4 * 2560 * 2 + 512 * 320 * 2 + 512 * 4 * 4,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine", type=Path)
    args = parser.parse_args()
    with atomic_model(args.model_output, args.engine) as staging:
        upgrade(args.model.resolve(strict=True), staging, args.output, args.engine)
    print("HC PREFILL PACKAGE READY", args.model_output, flush=True)


if __name__ == "__main__":
    main()
