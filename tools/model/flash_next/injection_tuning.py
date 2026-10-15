"""Tune one HC injection arithmetic contract for token and bulk execution."""

import argparse
import json
from pathlib import Path
from tools.model.flash_next.prefill_tuning import ranges, samples
from tools.model.publication import load_model, write_json


def tune(model, output):
    import torch
    import tilelang
    from safetensors import safe_open
    from kernels.model.hyperconnection import hc_injection, hc_projection
    from kernels.model.rows import explicit_rows
    from tools.operators.common import configure, benchmark, export_kernel, environment

    configure()
    torch.manual_seed(20261002)
    kernels = {}
    for name, bm in [("tiny1", 1), ("tiny2", 2), ("bulk16", 16), ("bulk32", 32), ("bulk64", 64)]:
        prim = (
            hc_injection.__wrapped__(1024, 4, 10240, dynamic_rows=True, block_m=bm)
            if name.startswith("tiny")
            else hc_projection.__wrapped__(
                1024, 4, 10240, block_m=bm, block_n=16, dtype="float16", dynamic_rows=True
            )
        )
        kernels[name] = tilelang.compile(
            explicit_rows(prim),
            out_idx=[],
            execution_backend="nvrtc",
            target={"kind": "cuda", "arch": "sm_87"},
        )
        export_kernel(kernels[name], output / "aot" / name)
    data, _, _ = load_model(model)
    names = [
        b["name"]
        for b in data["metadata"]["buffers"]
        if b.get("data") and b["name"].endswith("_inject.weight_value")
    ]
    weight_map = json.loads((model / "cache/weights/model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    cases = []
    records = []
    # Every real injection matrix, both signs/scale ranges, nonaligned tails and
    # changed-input Graph replay. Tiling cannot change a single output bit.
    for index, name in enumerate(names):
        with safe_open(
            model / "cache/weights" / weight_map[name], framework="pt", device="cpu"
        ) as f:
            w = f.get_tensor(name).cuda()
        live = samples() if index == 0 else [1, 3, 17, 33, 129, 257, 513, 1024]
        for rows in [n for n in live if n <= 1024]:
            a = torch.randn(rows, 10240, device="cuda", dtype=torch.float16) * (
                2 if index % 2 else 0.1
            )
            storage = torch.full((rows + 1, 4), 91.0, device="cuda", dtype=torch.float16)
            out = storage[:rows]

            def run(variant):
                k = kernels[variant]
                k.adapter.func(a, w, out, rows, stream=torch.cuda.current_stream().cuda_stream)

            run("bulk16")
            torch.cuda.synchronize()
            expected = out.clone()
            times = {}
            for variant in kernels:
                run(variant)
                torch.cuda.synchronize()
                assert torch.equal(out, expected), (
                    name,
                    rows,
                    variant,
                    float((out - expected).abs().max()),
                )
                assert bool((storage[-1] == 91).all())
                if index == 0:
                    timing, graph = benchmark(lambda: run(variant), repetitions=12)
                    a.neg_()
                    run("bulk16")
                    torch.cuda.synchronize()
                    changed = out.clone()
                    out.fill_(91)
                    graph.replay()
                    torch.cuda.synchronize()
                    assert torch.equal(out, changed), (name, rows, variant, "replay")
                    a.neg_()
                    times[variant] = timing["median_ms"]
            if times:
                records.append(dict(rows=rows, times=times))
            cases.append(dict(matrix=name, rows=rows, exact=True))
        print("CHECKED", index + 1, len(names), name, flush=True)
    selected = ranges(records, 1024)
    write_json(
        output / "results.json",
        dict(
            complete=True,
            identity=environment(),
            source=data["execution_package"],
            records=records,
            cases=cases,
            **selected,
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    tune(a.model, a.output)


if __name__ == "__main__":
    main()
