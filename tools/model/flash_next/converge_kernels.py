"""Converge Flash vision and MTP entries and publish measured shared-row choices."""

import argparse
import copy
import json
import re
from pathlib import Path

from tools.model.publication import atomic_model, clone_model, load_model, write_json
from tools.model.flash_next.unify_rows import Compiler
from tools.model.upgrade_dynamic_batch import contract


def divide_rows(value, divisor):
    if isinstance(value, dict):
        if value == {"op": "rows"}:
            # Valid images have multiples of four patches; ceil keeps generic
            # validation-only short rows legal without changing image launches.
            return dict(
                op="divide",
                lhs=dict(op="add", lhs=value, rhs=dict(op="constant", value=divisor - 1)),
                rhs=dict(op="constant", value=divisor),
            )
        return {k: divide_rows(v, divisor) for k, v in value.items()}
    if isinstance(value, list):
        return [divide_rows(v, divisor) for v in value]
    return value


def entries(data, package, compiler):
    """Capacity is a storage bound; runtime supplies the true image/verify length."""
    removed, added, launches = set(), [], []
    vision = data["metadata"].get("vision")
    if vision:
        capacity = vision["max_patches"]
        source = f"vision_m{capacity}/"
        for k in package["kernels"]:
            if not k["name"].startswith("vision_m"):
                continue
            removed.add(k["name"])
            if not k["name"].startswith(source):
                continue
            item = copy.deepcopy(k)
            item["name"] = item["name"].replace(source, "vision/", 1)
            merge = item["name"].endswith(("/end/k1", "/end/k2"))
            launch = contract(item, compiler.host(k), capacity // 4 if merge else capacity)
            launch["capacity"] = capacity
            if merge:
                launch["grid"] = divide_rows(launch["grid"], 4)
                launch["arguments"] = divide_rows(launch["arguments"], 4)
            added.append(item)
            launches.append(launch)
        vision["plans"] = [dict(patches=capacity, min_patches=4, program="vision")]
    spec = data["metadata"].get("mtp")
    if spec:
        for k in package["kernels"]:
            if re.match(r"verify_m[2-8]/begin/", k["name"]):
                removed.add(k["name"])
                if k["name"] == "verify_m8/begin/k0":
                    item, launch = compiler.emit(k, "mtp_verify/begin/k0", compiler.pointers(k), 8)
                    added.append(item)
                    launches.append(launch)
        spec["verification_plans"] = [
            dict(
                tokens=8,
                min_tokens=2,
                program="mtp_verify",
                restore_program="mtp_restore",
                capture_program="mtp_capture",
            )
        ]
    package["kernels"] = [k for k in package["kernels"] if k["name"] not in removed] + added
    package["dynamic_batch_kernels"] = [
        k for k in package["dynamic_batch_kernels"] if k["name"] not in removed
    ] + launches
    package["plan_kernel_contracts"] = sorted(
        set(package["plan_kernel_contracts"]) - removed | {k["name"] for k in added}
    )


def shared_choices(package, compiler, tuning):
    """Local choices replace complete old row portfolios, including unused defaults."""
    records = json.loads(tuning.read_text())
    if not records["complete"] or records["source"] != compiler.reader.base.name:
        raise ValueError("Incomplete or mismatched shared-row autotune")
    original = {k["name"]: k for k in package["kernels"]}
    removed, added, dynamic = set(), [], []
    for group in records["groups"]:
        for slot, definition in group["slots"].items():
            capacity = definition["capacity"]
            candidates = definition["candidates"]
            removed.update(n for n in original if n == slot or n.startswith(slot + "/rows/"))
            selected = group["selection"]
            # Fixed-row AOT selections are explicit compile specializations.
            fixed = definition["fixed"]
            # Pick a default actually used outside exact AOT rows, including
            # capacity-clipped cached evidence. Never publish an unused fallback.
            from collections import Counter

            usage = Counter()
            for first, last, implementation in selected["intervals"]:
                if first <= capacity:
                    usage[implementation] += sum(
                        high - low + 1
                        for low, high in split_fixed(first, min(last, capacity), fixed)
                    )
            default = max(usage, key=lambda key: (usage[key], key))
            item, launch = compiler.emit(
                original[candidates[default]],
                slot,
                compiler.pointers(original[candidates[default]]),
                capacity,
                multiplier=definition["multiplier"],
            )
            added.append(item)
            dynamic.append(launch)
            for choice in selected["intervals"]:
                first, last, name = choice
                # Real fixed specializations remain legal only at their compiled row count.
                intervals = split_fixed(first, last, fixed)
                for low, high in intervals:
                    if name == default:
                        continue
                    old = original[candidates[name]]
                    item, launch = compiler.emit(
                        old,
                        f"{slot}/rows/{low}-{high}",
                        compiler.pointers(old),
                        capacity,
                        multiplier=definition["multiplier"],
                    )
                    added.append(item)
                    dynamic.append(launch)
            for row, name in fixed.items():
                item = copy.deepcopy(original[name])
                item["name"] = f"{slot}/rows/{row}-{row}"
                added.append(item)
                old_launch = next(
                    (x for x in package["dynamic_batch_kernels"] if x["name"] == name), None
                )
                if old_launch:
                    launch = copy.deepcopy(old_launch)
                    launch["name"] = item["name"]
                    dynamic.append(launch)
    package["kernels"] = [k for k in package["kernels"] if k["name"] not in removed] + added
    package["dynamic_batch_kernels"] = [
        k for k in package["dynamic_batch_kernels"] if k["name"] not in removed
    ] + dynamic
    package["plan_kernel_contracts"] = sorted(
        set(package["plan_kernel_contracts"]) - removed | {k["name"] for k in added}
    )


def injection_choices(package, compiler, tuning):
    """Use K16 MMA for every HC injection, including the single-row reference."""
    from tools.operators.abi import parse_host, validate_parameter_count
    from tools.model.publication import link_or_copy, file_hash
    from tools.model.flash_next.optimize_decode import replace_binding

    report = json.loads((tuning / "results.json").read_text())
    if not report["complete"] or report["source"] != compiler.reader.base.name:
        raise ValueError("Incomplete or mismatched HC injection autotune")
    assets = {}
    hosts = {}
    sources = {}
    for name in report["used"]:
        folder = tuning / "aot" / name
        assets[name] = {}
        for field, file in [
            ("module", "kernel.cubin"),
            ("source", "kernel.cu"),
            ("host_abi", "host.txt"),
        ]:
            target = compiler.root / "injection" / name / file
            target.parent.mkdir(parents=True, exist_ok=True)
            link_or_copy(folder / file, target)
            assets[name][field] = dict(
                file=str(target.relative_to(compiler.root)), sha256=file_hash(target)
            )
        hosts[name] = parse_host((folder / "host.txt").read_text())[0]
        sources[name] = (folder / "kernel.cu").read_text()
        validate_parameter_count(sources[name], hosts[name])
    original = {k["name"]: k for k in package["kernels"]}
    removed = set()
    added = []
    launches = []
    slots = []
    for k in original.values():
        if "/rows/" in k["name"] or "/skip/" in k["name"]:
            continue
        p = compiler.pointers(k)
        weight = next(
            (v for v in p.values() if v.get("name", "").endswith("_inject.weight_value")), None
        )
        if weight is None or not k["name"].startswith(("flash_decode/", "draft_rows/", "decode/")):
            continue
        slots.append((k, p, weight))
    for old, p, weight in slots:
        slot = old["name"]
        output = next((p[name] for name in ("Out", "C", "Output") if name in p), None)
        if output is None:
            raise ValueError(f"Missing HC injection output binding: {slot}")
        legacy = slot.startswith("decode/")
        cap = 1 if legacy else 512 if slot.startswith("draft_rows/") else 1024
        first = report["default"]
        if legacy:
            first = next(
                (
                    v["implementation"]
                    for v in report["selections"]
                    if v["min_rows"] <= 1 <= v["max_rows"]
                ),
                first,
            )
        removed.update(n for n in original if n == slot or n.startswith(slot + "/rows/"))
        selected = [(slot, first)]
        if not legacy:
            selected.extend(
                (f"{slot}/rows/{v['min_rows']}-{min(v['max_rows'], cap)}", v["implementation"])
                for v in report["selections"]
                if v["min_rows"] <= cap
            )
        for name, implementation in selected:
            pointers = dict(A=p["A"])
            if implementation.startswith("tiny"):
                pointers.update(Weight=weight, Output=output)
            else:
                pointers.update(W=weight, C=output)
            host = hosts[implementation]
            k = replace_binding(
                dict(old, name=name),
                host,
                assets[implementation],
                pointers,
                dimensions=dict(m=cap, rows=cap),
            )
            source = sources[implementation]
            prototype = re.search(r"\b" + re.escape(k["symbol"]) + r"\s*\(([^)]*)\)", source)
            k["readonly_arguments"] = [
                "*" not in v or bool(re.search(r"\bconst\b", v)) for v in prototype[1].split(",")
            ]
            added.append(k)
            if not legacy:
                launches.append(contract(k, host, cap))
    package["kernels"] = [k for k in package["kernels"] if k["name"] not in removed] + added
    package["dynamic_batch_kernels"] = [
        k for k in package["dynamic_batch_kernels"] if k["name"] not in removed
    ] + launches
    package["plan_kernel_contracts"] = sorted(
        set(package["plan_kernel_contracts"]) - removed | {k["name"] for k in added}
    )


def coalesce_single_implementations(package, compiler):
    """One dynamic cubin needs one default; only real fixed AOT rows stay named."""
    from collections import defaultdict

    slots = defaultdict(list)
    for k in package["kernels"]:
        if "/rows/" in k["name"]:
            slots[k["name"].split("/rows/")[0]].append(k)
    removed, added, launches = set(), [], []
    for slot, ks in slots.items():
        dynamic = []
        for k in ks:
            if any(
                a["ctype"] in ("ctypes.c_int32", "c_int32") and a["value"] in ("m", "rows")
                for a in compiler.host(k)["ordered_arguments"]
            ):
                dynamic.append(k)
        if not dynamic or len({k["module"]["sha256"] for k in dynamic}) != 1:
            continue
        # Already measured tables may have distinct winners; no measurements are discarded.
        ranges = [k["name"].split("/rows/")[1] for k in dynamic]
        if all(x.split("-")[0] == x.split("-")[1] for x in ranges):
            continue
        old = dynamic[0]
        capacity = next(
            c["capacity"] for c in package["dynamic_batch_kernels"] if c["name"] == old["name"]
        )
        canonical = next((k for k in package["kernels"] if k["name"] == slot), None)
        covered = set()
        for k in ks:
            low, high = map(int, k["name"].split("/rows/")[1].split("-"))
            covered.update(range(low, high + 1))
        if covered != set(range(1, capacity + 1)) and (
            canonical is None or canonical["module"]["sha256"] != old["module"]["sha256"]
        ):
            continue
        # Only the binding identity changes. Preserve scalar multipliers and
        # grids exactly, including token -> expert-assignment expansion.
        item = copy.deepcopy(old)
        item["name"] = slot
        launch = copy.deepcopy(
            next(c for c in package["dynamic_batch_kernels"] if c["name"] == old["name"])
        )
        launch["name"] = slot
        removed.add(slot)
        removed.update(k["name"] for k in dynamic)
        added.append(item)
        launches.append(launch)
    package["kernels"] = [k for k in package["kernels"] if k["name"] not in removed] + added
    package["dynamic_batch_kernels"] = [
        k for k in package["dynamic_batch_kernels"] if k["name"] not in removed
    ] + launches
    package["plan_kernel_contracts"] = sorted(
        set(package["plan_kernel_contracts"]) - removed | {k["name"] for k in added}
    )


def coalesce_adjacent_selections(package):
    """Merge identical launch contracts even in a multi-implementation portfolio.

    Exact AOT implementations remain separate when any argument, grid or
    symbolic expression differs. Coupled expert route tiles are never inferred.
    """
    from collections import defaultdict

    launches = {k["name"]: k for k in package["dynamic_batch_kernels"]}
    slots = defaultdict(list)
    for k in package["kernels"]:
        if "/rows/" in k["name"] and k["name"] in launches:
            slot, extent = k["name"].split("/rows/")
            first, last = map(int, extent.split("-"))
            slots[slot].append((first, last, k))
    removed, added, dynamic = set(), [], []

    def identity(k):
        return [{a: b for a, b in item.items() if a != "name"} for item in (k, launches[k["name"]])]

    for slot, items in slots.items():
        spans = []
        for first, last, k in sorted(items, key=lambda x: x[0]):
            if spans and first <= spans[-1][1]:
                raise ValueError("Overlapping row selections")
            if spans and first == spans[-1][1] + 1 and identity(k) == identity(spans[-1][2][0]):
                spans[-1][1] = last
                spans[-1][2].append(k)
            else:
                spans.append([first, last, [k]])
        for first, last, ks in spans:
            if len(ks) == 1:
                continue
            name = f"{slot}/rows/{first}-{last}"
            item, launch = copy.deepcopy(ks[0]), copy.deepcopy(launches[ks[0]["name"]])
            item["name"] = launch["name"] = name
            removed.update(k["name"] for k in ks)
            added.append(item)
            dynamic.append(launch)
    package["kernels"] = [k for k in package["kernels"] if k["name"] not in removed] + added
    package["dynamic_batch_kernels"] = [
        k for k in package["dynamic_batch_kernels"] if k["name"] not in removed
    ] + dynamic
    package["plan_kernel_contracts"] = sorted(
        set(package["plan_kernel_contracts"]) - removed | {k["name"] for k in added}
    )
    return len(removed) - len(added)


def split_fixed(first, last, fixed):
    result = []
    for row in sorted(int(r) for r in fixed if first <= int(r) <= last):
        if first < row:
            result.append((first, row - 1))
        first = row + 1
    if first <= last:
        result.append((first, last))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "model-output", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--tuning", type=Path, required=True)
    p.add_argument("--injection-tuning", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    data, origin, package = load_model(a.model)
    with atomic_model(a.model_output) as staging:
        root = clone_model(a.model, staging, origin)
        compiler = Compiler(
            origin, root, a.output, {b["name"]: b for b in data["metadata"]["buffers"]}
        )
        shared_choices(package, compiler, a.tuning)
        coalesce_single_implementations(package, compiler)
        injection_choices(package, compiler, a.injection_tuning)
        coalesce_adjacent_selections(package)
        entries(data, package, compiler)
        from tools.model.flash_next.converge_metadata import converge

        converge(data, package, compiler, a.model, a.output)
        from tools.model.compact import compact_package
        import shutil

        digest = compact_package(root, package, staging / "cache/packages")
        shutil.rmtree(root)
        data["execution_package"] = digest
        write_json(staging / "cache/model.json", data)
    write_json(
        a.output / "results.json", dict(execution_package=digest, bindings=len(package["kernels"]))
    )


if __name__ == "__main__":
    main()
