"""Share HC cache exponents and QSA radix passes through immutable ABI scalars.

Tile geometry comes from the existing measured package. Scores stripe counts
are selected offline; no precision, reduction or causal partition is changed.
"""

import argparse
import math
import re
import shutil
from pathlib import Path

from tools.model.row_convergence import coalesce_adjacent_selections
from tools.model.host_compiler import Compiler
from tools.model.publication import atomic_model, clone_model, load_model, write_json
from tools.operators.abi import evaluate


def scalars(kernel, host):
    return {
        h["value"]: a["value"]
        for a, h in zip(kernel["args"], host["ordered_arguments"])
        if a["kind"] == "i32"
    }


def hc_geometry(kernel, compiler):
    p = compiler.pointers(kernel)
    source = compiler.read(kernel["source"]).decode()
    bases = set(re.findall(r"hc_expand13_four\([^;]+?,\s*\(uint\)(\d+)\)", source))
    bound = scalars(kernel, compiler.host(kernel))
    if not bases and "base_exp" in bound:
        bases.add(str(bound["base_exp"]))
    if len(bases) != 1:
        raise ValueError("Missing certified HC exponent")
    grid = compiler.host(kernel)["launch_expressions"]["gridDimX"]
    bm = next(m - 1 for m in range(2, 130) if evaluate(grid, dict(m=m, rows=m)) > 1)
    branches = set(map(int, re.findall(r"branch \* (\d+)", source)))
    if branches not in ({32}, {bm * 32}):
        raise ValueError("Missing certified HC matrix orientation")
    return dict(
        rank=compiler.buffers[p["Lo"]["name"]]["shape"][1] * 64,
        base_exp=int(bases.pop()),
        patch_count=math.prod(compiler.buffers[p["Patches"]["name"]]["shape"])
        if "Patches" in p
        else 0,
        block_m=bm,
        weight_left=branches != {32},
    )


def private_roles(source, columns, row_stride=None):
    """Recover certified pointer-table columns/byte views from exported wrappers."""
    loads = dict(re.findall(r"\b(\w+) = Pointers\[([^;]+)\];", source))
    roles, offsets = {}, {}
    for role, variable, offset in re.findall(
        r"void\* (\w+)_row = orin_row_pointer\(\(?([A-Za-z_][A-Za-z_0-9]*)"
        r"(?: \+ \(uint64_t\)(\d+))?\)?\);",
        source,
    ):
        expression = loads[variable]
        index = re.search(r"\+ \(int64_t\)(\d+)\)$", expression)
        if index is None and " + " in expression:
            raise ValueError("Unknown private-state column expression")
        if row_stride is not None and set(
            map(int, re.findall(r"\* \(int64_t\)(\d+)", expression))
        ) != {row_stride}:
            raise ValueError("Private-state row stride differs from model layout")
        column = int(index[1]) if index else 0
        if column >= len(columns):
            raise ValueError("Invalid private-state column")
        roles[role] = ("private", columns[column])
        offsets[role] = int(offset or 0)
    return roles, offsets


def qsa_kind(roles):
    if {"Cache", "Query", "Scores"} <= roles:
        return "scores"
    if {"Hist", "Scores"} <= roles:
        return "hist"
    if {"Hist", "Remaining"} <= roles:
        return "choose"
    return None


def radix_shift(kind, source):
    expression = (
        r">> (24|16|8)\) & 255" if kind == "hist" else r"(?:bucket|\(255 - b\)) << (24|16|8)\)"
    )
    found = set(map(int, re.findall(expression, source)))
    if len(found) > 1 or (
        not found and ("AtomicAdd" if kind == "hist" else "prefix |") not in source
    ):
        raise ValueError("Missing certified QSA radix byte")
    return found.pop() if found else 0


def publish(data, package, compiler, stripes):
    from kernels.model import qsa
    from kernels.model.hc_bf16_cache import hc_up13_mix
    from kernels.model.independent_rows import independent_rows

    dynamics = {k["name"]: k for k in package["dynamic_batch_kernels"]}
    columns = data["metadata"]["batch_layout"]["state_columns"]
    stride = len(columns) * len(data["metadata"]["batch_layout"]["layers"])
    context = data["metadata"]["max_context"]
    private, prefixes = {}, {}
    # The first radix histogram did not previously consume Prefix. Dynamic
    # shift adds that input, using the adjacent choose pass's identical view.
    for k in package["kernels"]:
        p = compiler.pointers(k)
        indirect = set(p) == {"Pointers"} and k["name"].startswith("flash_decode_state/")
        if indirect:
            roles, offsets = private_roles(compiler.read(k["source"]).decode(), columns, stride)
            private[k["name"]] = roles, offsets
            if qsa_kind(roles.keys()) == "choose":
                prefixes[k["name"].rsplit("/", 1)[0]] = (roles["Prefix"], offsets["Prefix"])
        elif qsa_kind(p.keys()) == "choose":
            prefixes[k["name"].rsplit("/", 1)[0]] = p["Prefix"]
    changes = []
    for i, old in enumerate(package["kernels"]):
        p = compiler.pointers(old)
        build = key = constants = None
        capacity = dynamics.get(old["name"], {}).get("capacity", 1)
        if old["symbol"] == "hc_up13_vector_project_kernel":
            geometry = hc_geometry(old, compiler)
            constants = dict(base_exp=geometry.pop("base_exp"))
            key = "hc-parameter-" + "-".join(str(v) for v in geometry.values())

            def build(geometry=geometry):
                return hc_up13_mix.__wrapped__(
                    1024, 2560, dynamic_rows=True, base_exp=None, **geometry
                )
        else:
            indirect = old["name"] in private
            roles, offsets = private.get(old["name"], ({}, {}))
            kind = qsa_kind(roles.keys() if indirect else p.keys())
            if kind is None:
                continue
            if kind == "hist" and "Prefix" not in (roles if indirect else p):
                prefix = prefixes[old["name"].rsplit("/", 1)[0]]
                if indirect:
                    roles["Prefix"], offsets["Prefix"] = prefix
                else:
                    p["Prefix"] = prefix
            old_scalars = scalars(old, compiler.host(old))
            constants = (
                dict(stripes=stripes)
                if kind == "scores"
                else dict(
                    shift=old_scalars["shift"]
                    if "shift" in old_scalars
                    else radix_shift(kind, compiler.read(old["source"]).decode())
                )
            )
            key = f"qsa-parameter-{kind}-{'private' if indirect else 'direct'}-c{context}"
            if indirect:
                # Different layers share the same certified scratch geometry.
                key += "-" + str(sorted(roles.items())) + str(sorted(offsets.items()))
                import hashlib

                key = "qsa-private-" + hashlib.sha256(key.encode()).hexdigest()[:20]

            def build(kind=kind, indirect=indirect, roles=roles, offsets=offsets):
                options = dict(dynamic_rows=not indirect)
                rows = 1 if indirect else min(4096, context)
                if kind == "scores":
                    fn = qsa.index_scores.__wrapped__(rows, context, stripes=None, **options)
                elif kind == "hist":
                    fn = qsa.radix_histogram.__wrapped__(rows, context, None, **options)
                else:
                    fn = qsa.radix_choose.__wrapped__(rows, context, None, **options)
                if indirect:
                    fn, _ = independent_rows(
                        fn,
                        128,
                        roles,
                        {name: j for j, name in enumerate(columns)},
                        row_stride=stride,
                        dynamic=True,
                        offsets=offsets,
                    )
                return fn

        item, launch = compiler.emit(
            old, old["name"], p, capacity, build=build, key=key, constants=constants
        )
        package["kernels"][i] = item
        if old["name"] in dynamics:
            dynamics[old["name"]] = launch
        changes.append(dict(before=old, after=item, constants=constants, rows=capacity))
    package["dynamic_batch_kernels"] = list(dynamics.values())
    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "model-output", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    from tools.operators.common import configure
    from tools.model.compact import compact_package

    configure()
    data, origin, package = load_model(args.model)
    with atomic_model(args.model_output) as staging:
        root = clone_model(args.model, staging, origin)
        compiler = Compiler(
            origin, root, args.output, {b["name"]: b for b in data["metadata"]["buffers"]}
        )
        report = converge(data, package, compiler, args.model, args.output)
        report["coalesced"] = coalesce_adjacent_selections(package)
        digest = compact_package(root, package, staging / "cache/packages")
        shutil.rmtree(root)
        data["execution_package"] = digest
        write_json(staging / "cache/model.json", data)
    report.update(execution_package=digest, complete=True, compiled=len(compiler.compiled))
    write_json(args.output / "results.json", report)


def converge(data, package, compiler, model, output):
    from tools.model.flash_next.checks.metadata import tune_stripes, validate

    context = data["metadata"]["max_context"]
    stripes = tune_stripes(output, context)
    changes = publish(data, package, compiler, stripes)
    return validate(model, compiler, changes, output, context)


if __name__ == "__main__":
    main()
