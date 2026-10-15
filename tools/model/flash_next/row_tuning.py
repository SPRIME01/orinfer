"""Measure existing shared-row cubins as a portfolio, without preset boundaries.

Coupled expert route/up/down candidates are measured and selected together.
Fixed-row AOT specializations keep their exact legal extent. All other choices
are inferred from CUDA Graph timings and checked for output/tail/replay equality.
"""

import argparse
import ctypes as C
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from tools.model.publication import AssetReader, load_model, write_json
from tools.model.flash_next.unify_rows import Compiler
from tools.model.flash_next.prefill_tuning import ranges
from tools.operators.abi import evaluate


def groups(package, compiler):
    slots = defaultdict(list)
    original = {k["name"]: k for k in package["kernels"]}
    for k in package["kernels"]:
        if "/rows/" in k["name"]:
            slots[k["name"].split("/rows/")[0]].append(k)
    dynamic_contracts = {c["name"]: c for c in package["dynamic_batch_kernels"]}
    singles = []
    experts = defaultdict(list)
    for slot, ks in slots.items():
        if not any(re.search(r"/rows/(?:33|129)-", k["name"]) for k in ks):
            continue
        # AOT integer projections and SwiGLU share their already tuned exact choices.
        # The remaining dynamic kernels are legal across the whole allocation.
        candidates = {}
        fixed = {}
        canonical = [original[slot]] if slot in original else []
        if any({"Hi", "Lo", "Normed", "Out"} <= compiler.pointers(k).keys() for k in ks):
            canonical = []  # These fuse the following mix; canonical computes only its projection.
        for k in canonical + ks:
            host = compiler.host(k)
            dynamic = any(
                a["ctype"] in ("ctypes.c_int32", "c_int32") and a["value"] in ("rows", "m")
                for a in host["ordered_arguments"]
            )
            suffix = k["name"].split("/rows/")[-1]
            if "-" in suffix:
                lo, hi = map(int, suffix.split("-"))
                if lo == hi:
                    fixed[str(lo)] = k["name"]
            if not dynamic:
                continue
            identity = k["module"]["sha256"]
            if identity not in candidates:
                candidates[identity] = k["name"]
        if len(candidates) < 2:
            continue
        capacity = 128 if "/head/" in slot else 512 if slot.startswith("draft_rows/") else 1024
        p = compiler.pointers(original[next(iter(candidates.values()))])
        if any(v.get("name", "").endswith("_inject.weight_value") for v in p.values()):
            # Dedicated K16 tuning also canonicalizes the single-token reference.
            continue
        sample = original[next(iter(candidates.values()))]
        bound = dynamic_contracts[sample["name"]]["capacity"]
        multipliers = {
            sample["args"][i]["value"] // bound
            for i, a in enumerate(compiler.host(sample)["ordered_arguments"])
            if a["ctype"] in ("ctypes.c_int32", "c_int32") and a["value"] in ("m", "rows")
        }
        if len(multipliers) != 1 or any(x < 1 for x in multipliers):
            raise ValueError("Invalid row multiplier contract")
        entry = dict(
            capacity=capacity,
            candidates=candidates,
            fixed=fixed,
            multiplier=multipliers.pop(),
        )
        if {"IDs", "Counts", "RelativeSlot", "TileExpert"} <= p.keys() or {
            "Low",
            "High",
            "P",
            "Counts",
            "Out",
        } <= p.keys():
            experts[slot.rsplit("/k", 1)[0]].append((slot, entry))
        else:
            singles.append(dict(slots={slot: entry}))
    for section, items in experts.items():
        # Metadata and both expert GEMMs must use the same route tile.
        paired = {}
        for slot, entry in items:
            versions = {}
            for kernel in slots[slot]:
                host = compiler.host(kernel)
                if any(
                    a["ctype"] in ("ctypes.c_int32", "c_int32") and a["value"] in ("rows", "m")
                    for a in host["ordered_arguments"]
                ):
                    suffix = kernel["name"].split("/rows/")[1]
                    versions[suffix] = kernel["name"]
            entry["candidates"] = versions
            paired[slot] = entry
        common = set.intersection(*(set(x["candidates"]) for x in paired.values()))
        if len(paired) != 3 or len(common) < 2:
            raise ValueError("Incomplete coupled expert portfolio: " + section)
        unique = {}
        for key in sorted(common):
            identity = tuple(
                original[entry["candidates"][key]]["module"]["sha256"] for entry in paired.values()
            )
            unique.setdefault(identity, key)
        for entry in paired.values():
            entry["candidates"] = {key: entry["candidates"][key] for key in unique.values()}
        singles.append(dict(slots=paired))
    return singles


class Driver:
    def __init__(self, reader):
        self.reader = reader
        self.library = C.CDLL("libcuda.so.1")
        self.modules = {}
        self.functions = {}

    def check(self, code):
        if code:
            raise RuntimeError(f"CUDA driver {code}")

    def function(self, k):
        key = (k["module"]["sha256"], k["symbol"])
        if key not in self.functions:
            if key[0] not in self.modules:
                raw = C.create_string_buffer(self.reader.read(k["module"]))
                module = C.c_void_p()
                self.check(self.library.cuModuleLoadData(C.byref(module), C.cast(raw, C.c_void_p)))
                self.modules[key[0]] = module
            f = C.c_void_p()
            self.check(
                self.library.cuModuleGetFunction(
                    C.byref(f), self.modules[key[0]], k["symbol"].encode()
                )
            )
            if k["shared_memory_bytes"] > 49152:
                self.check(self.library.cuFuncSetAttribute(f, 8, k["shared_memory_bytes"]))
            self.functions[key] = f
        return self.functions[key]

    def call(self, k, host, pointers, rows, multiplier):
        import torch

        dims = dict(m=rows * multiplier, rows=rows * multiplier)
        values = [
            C.c_uint64(pointers[a["name"]].data_ptr() + a.get("offset", 0))
            if a["kind"].startswith("buffer")
            else C.c_int32(evaluate(h["value"], dims))
            for a, h in zip(k["args"], host["ordered_arguments"])
        ]
        args = (C.c_void_p * len(values))(*(C.cast(C.byref(v), C.c_void_p) for v in values))
        grid = [evaluate(host["launch_expressions"]["gridDim" + axis], dims) for axis in "XYZ"]
        f = self.function(k)

        def run(values=values):
            # Keep the host argument storage alive for every direct/replay capture.
            self.check(
                self.library.cuLaunchKernel(
                    f,
                    *grid,
                    *k["block"],
                    k["shared_memory_bytes"],
                    C.c_void_p(torch.cuda.current_stream().cuda_stream),
                    args,
                    None,
                )
            )

        return run


def intervals(records, capacity):
    selected = ranges(records, capacity)
    live = [selected["default"]] * capacity
    for x in selected["selections"]:
        for i in range(x["min_rows"], x["max_rows"] + 1):
            live[i - 1] = x["implementation"]
    out = []
    first = 1
    for n in range(2, capacity + 2):
        if n == capacity + 1 or live[n - 1] != live[first - 1]:
            out.append([first, n - 1, live[first - 1]])
            first = n
    return dict(default=selected["default"], used=selected["used"], intervals=out)


def tune(model, output):
    import torch
    from safetensors import safe_open
    from tools.operators.common import configure, benchmark, environment

    configure()
    torch.manual_seed(20261002)
    data, origin, package = load_model(model)
    specs = {b["name"]: b for b in data["metadata"]["buffers"]}
    compiler = Compiler(origin, origin, output, specs)
    driver = Driver(AssetReader(origin))
    original = {k["name"]: k for k in package["kernels"]}
    portfolio = groups(package, compiler)
    weight_map = json.loads((model / "cache/weights/model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    report = dict(
        complete=False, identity=environment(), source=data["execution_package"], groups=[]
    )
    report["tool_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    cache = {}
    for index, group in enumerate(portfolio):
        entries = group["slots"]
        capacity = next(iter(entries.values()))["capacity"]
        names = sorted(set(n for entry in entries.values() for n in entry["candidates"].values()))
        # Kernel and operand geometry fully identify a reusable tuning family.
        signature = []
        for slot, entry in entries.items():
            signature.append(
                [
                    (key, original[name]["module"]["sha256"])
                    for key, name in entry["candidates"].items()
                ]
            )
        key = hashlib.sha256(
            json.dumps(dict(capacity=capacity, kernels=signature), sort_keys=True).encode()
        ).hexdigest()
        if len(entries) == 1:
            entry = next(iter(entries.values()))
            entry["candidates"] = {
                str(i): name for i, name in enumerate(entry["candidates"].values())
            }
        if key in cache:
            group["selection"] = cache[key]
            report["groups"].append(group)
            continue
        pointers = {}
        outputs = set()
        roles_by_name = {}
        for name in names:
            roles = compiler.pointers(original[name])
            roles_by_name[name] = roles
            for role, binding in roles.items():
                if binding["kind"] not in ("buffer", "buffer_slice"):
                    continue
                bname = binding["name"]
                if role in ("Out", "Output", "C"):
                    outputs.add(bname)
                if bname in pointers:
                    continue
                spec = specs[bname]
                dtype = getattr(
                    torch,
                    {
                        "bf16": "bfloat16",
                        "f16": "float16",
                        "f32": "float32",
                        "i8": "int8",
                        "u8": "uint8",
                        "i32": "int32",
                        "u32": "uint32",
                        "u16": "uint16",
                        "i64": "int64",
                        "u64": "uint64",
                    }[spec["dtype"]],
                )
                if spec.get("data"):
                    with safe_open(
                        model / "cache/weights" / weight_map[bname], framework="pt", device="cpu"
                    ) as f:
                        tensor = f.get_tensor(bname).reshape(spec["shape"]).cuda()
                else:
                    tensor = torch.zeros(spec["shape"], dtype=dtype, device="cuda")
                    if tensor.is_floating_point():
                        tensor.normal_(0, 0.01)
                    elif dtype == torch.int8:
                        tensor.random_(-128, 128)
                    elif dtype == torch.uint32:
                        tensor.copy_(
                            torch.randint(
                                0, 2**32, tensor.shape, device="cuda", dtype=torch.int64
                            ).to(dtype)
                        )
                pointers[bname] = tensor
        # Name candidate IDs consistently across geometrically equal singleton slots.
        if len(entries) == 1:
            slot, entry = next(iter(entries.items()))
            entry["candidates"] = {
                str(i): name for i, name in enumerate(entry["candidates"].values())
            }
        candidates = next(iter(entries.values()))["candidates"]
        measures = []
        rejected = set()
        samples = sorted(
            set(range(1, 18))
            | {round(math.exp(math.log(18) + i * math.log(capacity / 18) / 24)) for i in range(25)}
            | {
                n + d
                for n in (32, 64, 128, 256, 512, 1024)
                for d in (-1, 0, 1)
                if 1 <= n + d <= capacity
            }
        )
        for rows in samples:
            if len(entries) == 3:
                route = next(name for name in names if "IDs" in roles_by_name[name])
                roles = roles_by_name[route]
                ids = (
                    torch.arange(rows * 10, device="cuda").reshape(rows, 10) * 37 // 10
                    + torch.arange(10, device="cuda")
                ) % 512
                pointers[roles["IDs"]["name"]].view(-1)[: rows * 10].copy_(ids.flatten())
            times = {}
            reference = None
            # Candidate order follows route metadata, gate/up and down.
            for choice in candidates:
                calls = []
                for slot, entry in entries.items():
                    name = entry["candidates"][choice]
                    k = original[name]
                    calls.append(
                        driver.call(k, compiler.host(k), pointers, rows, entry["multiplier"])
                    )

                def run():
                    for call in calls:
                        call()

                for name in outputs:
                    pointers[name].fill_(91)
                run()
                torch.cuda.synchronize()
                for n in outputs:
                    extent = pointers[n].numel() // capacity * rows
                    assert bool((pointers[n].view(-1)[extent:] == 91).all()), (
                        "tail",
                        index,
                        rows,
                        choice,
                        n,
                    )
                actual = {n: pointers[n].clone() for n in outputs}
                if reference is None:
                    reference = actual
                else:
                    for n in outputs:
                        exact = torch.equal(actual[n], reference[n])
                        # Floating projection candidates must preserve the reference
                        # bit pattern across every sampled length, just like integer paths.
                        if not exact:
                            rejected.add(choice)
                timing, graph = benchmark(run, repetitions=8)
                # Check replay after a live input change using a direct launch oracle.
                first = next(iter(entries.values()))["candidates"][choice]
                rs = roles_by_name[first]
                role = "A" if "A" in rs else "IDs" if "IDs" in rs else "X"
                if role in rs:
                    x = pointers[rs[role]["name"]]
                    backup = x.clone()
                    if x.is_floating_point():
                        x.neg_()
                    elif role == "IDs":
                        x.add_(1).remainder_(512)
                    else:
                        x.zero_()
                    run()
                    torch.cuda.synchronize()
                    updated = {n: pointers[n].clone() for n in outputs}
                    for n in outputs:
                        pointers[n].fill_(91)
                    graph.replay()
                    torch.cuda.synchronize()
                    assert all(torch.equal(pointers[n], updated[n]) for n in outputs), (
                        "replay",
                        index,
                        rows,
                        choice,
                    )
                    x.copy_(backup)
                times[choice] = timing["median_ms"]
            measures.append(dict(rows=rows, times=times))
        group["records"] = measures
        group["quality_rejected"] = sorted(rejected)
        legal = [
            dict(
                rows=r["rows"],
                times={key: time for key, time in r["times"].items() if key not in rejected},
            )
            for r in measures
        ]
        if not legal[0]["times"]:
            raise ValueError("No numerically legal row candidate")
        group["selection"] = intervals(legal, capacity)
        cache[key] = group["selection"]
        report["groups"].append(group)
        write_json(output / "results.json", report)
        print(
            "TUNED",
            index + 1,
            len(portfolio),
            list(entries),
            group["selection"]["used"],
            flush=True,
        )
        del pointers
        torch.cuda.empty_cache()
    report["complete"] = True
    write_json(output / "results.json", report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    tune(a.model, a.output)


if __name__ == "__main__":
    main()
