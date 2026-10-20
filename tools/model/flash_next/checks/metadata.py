"""Exact exported-cubin comparisons for parameterized HC and QSA operators."""

import json
import math

import torch

from tools.model.flash_next.converge_metadata import qsa_kind
from tools.model.flash_next.row_tuning import Driver
from tools.model.publication import AssetReader, write_json
from tools.operators.common import benchmark


def tune_stripes(output, capacity):
    import tilelang
    from kernels.model import qsa
    from kernels.model.rows import explicit_rows
    from tools.operators.abi import parse_host
    from tools.model.host_compiler import replace_binding
    from tools.model.publication import file_hash
    from tools.operators.common import export_kernel

    kernel = tilelang.compile(
        explicit_rows(
            qsa.index_scores.__wrapped__(min(4096, capacity), capacity, True, stripes=None)
        ),
        out_idx=[],
        execution_backend="nvrtc",
        target={"kind": "cuda", "arch": "sm_87"},
    )
    folder = output / "stripe-probe"
    export_kernel(kernel, folder)
    host = parse_host((folder / "host.txt").read_text())[0]
    driver = Driver(AssetReader(folder))
    assets = {
        field: dict(file=file, sha256=file_hash(folder / file))
        for field, file in (
            ("module", "kernel.cubin"),
            ("source", "kernel.cu"),
            ("host_abi", "host.txt"),
        )
    }
    records = []
    blocks = (capacity + 3) // 4
    for rows in sorted({min(capacity, r) for r in (1, 7, 33, 129, 512)}):
        query = torch.randn(rows, 4, 128, device="cuda", dtype=torch.float16)
        cache = torch.randn(blocks, 128, device="cuda", dtype=torch.float16)
        position = torch.zeros(1, device="cuda", dtype=torch.int32)
        scores = torch.full((blocks, rows), 91, device="cuda")
        for context in sorted({max(rows, capacity // 32), max(rows, capacity // 4), capacity}):
            position.fill_(context - rows)
            reference = None
            times = {}
            for stripes in (1, 2, 4, 8, 16, 32, 64):
                pointers = dict(Cache=cache, Position=position, Query=query, Scores=scores)
                bound = replace_binding(
                    dict(name="stripe-probe"),
                    host,
                    assets,
                    {name: dict(kind="buffer", name=name) for name in pointers},
                    dimensions=dict(m=rows, stripes=stripes),
                )
                run = driver.call(bound, host, pointers, rows, 1, dict(stripes=stripes))

                scores.fill_(91)
                run()
                torch.cuda.synchronize()
                if reference is None:
                    reference = scores.clone()
                else:
                    assert torch.equal(scores, reference), ("stripes", rows, context, stripes)
                timing, _ = benchmark(run, repetitions=5)
                times[stripes] = timing["median_ms"]
            records.append(dict(rows=rows, context=context, times=times))
            print("STRIPES CASE", rows, context, min(times, key=times.get), flush=True)
    # Normalize each workload to avoid letting the longest context dominate.
    costs = {
        s: math.exp(
            sum(math.log(r["times"][s] / min(r["times"].values())) for r in records) / len(records)
        )
        for s in (1, 2, 4, 8, 16, 32, 64)
    }
    selected = min(costs, key=costs.get)
    write_json(output / "stripes.json", dict(records=records, costs=costs, selected=selected))
    print("STRIPES", selected, costs, flush=True)
    return selected


def validate(model, compiler, changes, output, capacity):
    from safetensors import safe_open

    torch.manual_seed(20261002)
    before = Driver(compiler.reader)
    after = Driver(AssetReader(compiler.root))
    weight_map = json.loads((model / "cache/weights/model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    checked, seen = [], set()
    blocks = (capacity + 3) // 4
    segments = (blocks + 1023) // 1024
    for change in changes:
        old, new = change["before"], change["after"]
        p = compiler.pointers(old)
        hc = old["symbol"] == "hc_up13_vector_project_kernel"
        kind = qsa_kind(p.keys())
        if not hc and kind is None:
            continue  # Pointer-table wrappers are covered by full-model state tests.
        identity = (old["module"]["sha256"], p["Lo"]["name"] if hc else "")
        if identity in seen:
            continue
        seen.add(identity)
        dynamic = any(h["value"] in ("m", "rows") for h in compiler.host(old)["ordered_arguments"])
        samples = (
            sorted({1, min(3, change["rows"]), min(17, change["rows"]), min(129, change["rows"])})
            if dynamic
            else [1]
        )
        for rows in samples:
            tensors = {}
            all_p = dict(p, **compiler.pointers(new))
            for role, binding in all_p.items():
                spec = compiler.buffers[binding["name"]]
                if hc and role in ("Lo", "Hi", "Patches"):
                    with safe_open(
                        model / "cache/weights" / weight_map[binding["name"]],
                        framework="pt",
                        device="cpu",
                    ) as f:
                        tensor = f.get_tensor(binding["name"]).reshape(spec["shape"]).cuda()
                elif hc:
                    rank = compiler.buffers[p["Lo"]["name"]]["shape"][1] * 64
                    width = 2560 if role == "Out" else 4 * 2560 if role == "Normed" else rank
                    tensor = torch.randn(rows + 2, width, dtype=torch.float16, device="cuda") * 0.01
                elif role == "Query":
                    tensor = torch.randn(rows, 4, 128, dtype=torch.float16, device="cuda")
                elif role == "Cache":
                    tensor = torch.randn(blocks, 128, dtype=torch.float16, device="cuda")
                elif role == "Scores":
                    tensor = torch.rand(blocks, rows, device="cuda")
                elif role == "Hist":
                    tensor = torch.zeros(rows + 2, segments, 256, dtype=torch.int32, device="cuda")
                elif role == "Position":
                    tensor = torch.tensor(
                        [min(capacity, 65536) - rows], dtype=torch.int32, device="cuda"
                    )
                else:
                    tensor = torch.zeros(rows + 2, dtype=torch.int32, device="cuda")
                # Operands below use standalone arrays, not their original arenas.
                tensors[role] = tensor
            output_roles = (
                ["Out"]
                if hc
                else ["Scores"]
                if kind == "scores"
                else ["Hist"]
                if kind == "hist"
                else ["Prefix", "Remaining"]
            )
            # Histogram later bytes need a realistic previously selected prefix.
            if kind == "hist" and change["constants"]["shift"] != 24:
                bits = tensors["Scores"][0].contiguous().view(torch.int32)
                shift = change["constants"]["shift"]
                tensors["Prefix"][:rows].copy_((bits >> (shift + 8)) << (shift + 8))
            if kind == "choose":
                tensors["Hist"].random_(0, 4)
                tensors["Remaining"].fill_(3)

            def call(driver, k, constants=None):
                # Preserve ABI names but remove arena byte offsets for this probe.
                import copy

                k = copy.deepcopy(k)
                pointers = {}
                for role, binding in compiler.pointers(k).items():
                    name = "probe_" + role
                    for argument in k["args"]:
                        if argument == binding:
                            argument.update(name=name, offset=0)
                    pointers[name] = tensors[role]
                return driver.call(k, compiler.host(k), pointers, rows, 1, constants)

            old_call, new_call = call(before, old), call(after, new, change["constants"])
            initial = {r: tensors[r].clone() for r in output_roles}
            old_call()
            torch.cuda.synchronize()
            expected = {r: tensors[r].clone() for r in output_roles}
            for role in output_roles:
                tensors[role].copy_(initial[role])
            new_call()
            torch.cuda.synchronize()
            assert all(torch.equal(tensors[r], expected[r]) for r in output_roles), (
                old["name"],
                rows,
            )
            # Change a live input, then compare direct launch and graph replay.
            _, graph = benchmark(new_call, repetitions=2)
            role = (
                "A"
                if hc
                else "Query"
                if kind == "scores"
                else "Scores"
                if kind == "hist"
                else "Hist"
            )
            tensors[role].zero_()
            for r in output_roles:
                tensors[r].copy_(initial[r])
            new_call()
            torch.cuda.synchronize()
            updated = {r: tensors[r].clone() for r in output_roles}
            for r in output_roles:
                tensors[r].copy_(initial[r])
            graph.replay()
            torch.cuda.synchronize()
            assert all(torch.equal(tensors[r], updated[r]) for r in output_roles), (
                "replay",
                old["name"],
                rows,
            )
            checked.append(dict(name=old["name"], rows=rows, exact=True, graph_replay=True))
        print("CHECKED", old["name"], samples, flush=True)
    report = dict(cases=checked, rebound=len(changes), complete=False)
    write_json(output / "checks.json", report)
    return report
