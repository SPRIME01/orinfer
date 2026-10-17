"""Offline measured prefill choices; no length thresholds enter the runtime."""

import argparse
import math
from pathlib import Path

from tools.model.publication import write_json
from tools.model.row_selection import ranges as ranges

CAPACITY = 4096
ROUTER_TILES = [(16, 64), (32, 64), (64, 128), (32, 128), (64, 64)]


def samples():
    geometric = {
        round(math.exp(math.log(33) + i * math.log(CAPACITY / 33) / 32)) for i in range(33)
    }
    return sorted(
        set(range(1, 18))
        | geometric
        | {n + d for n in range(128, CAPACITY + 1, 128) for d in (-1, 0, 1) if n + d <= CAPACITY}
    )


def primitive(kind, name):
    if kind == "router":
        from kernels.model.flash_next import dense_projection

        bm, bn = ROUTER_TILES[int(name)]
        return dense_projection.__wrapped__(
            CAPACITY, 512, 2560, "bfloat16", "float32", True, block_m=bm, block_n=bn
        )
    if kind == "hc":
        from kernels.model.hc_prefill import hc_up_mix_prefill

        return hc_up_mix_prefill.__wrapped__(
            CAPACITY, 2560, 320, dynamic_rows=True, aligned_rows=name == "aligned"
        )
    from kernels.model.e8p_prefill import dispatch_prefill_a8, split_prefill_a8
    from kernels.model.moe import expert_dispatch

    if name == "fused":
        return dispatch_prefill_a8.__wrapped__(CAPACITY, dynamic_rows=True)
    if name == "pack":
        return split_prefill_a8.__wrapped__(CAPACITY * 10, 2560, True)
    return expert_dispatch.__wrapped__(CAPACITY, 2560, scale_group=2560, dynamic_rows=True)


def inputs(kind, rows, kernels):
    import torch

    torch.manual_seed(20261002)

    def call(name, *args):
        kernels[name].adapter.func(*args, stream=torch.cuda.current_stream().cuda_stream)

    if kind == "router":
        a = torch.randn(rows, 2560, device="cuda", dtype=torch.float16) * 0.02
        w = torch.randn(512, 2560, device="cuda", dtype=torch.bfloat16) * 0.02
        storage = torch.full((rows + 1, 512), 91.0, device="cuda")
        out = storage[:rows]

        def run(name):
            call(name, a, w, out, rows)

        def change():
            a.neg_()
    elif kind == "hc":
        a = torch.randn(rows, 320, device="cuda", dtype=torch.bfloat16) * 0.02
        w = torch.randn(10240, 320, device="cuda", dtype=torch.bfloat16) * 0.02
        residual = torch.randn(rows, 4, 2560, device="cuda", dtype=torch.float16) * 0.1
        gamma = torch.randn(4, 2560, device="cuda") * 0.1
        inv = torch.rand(rows, 4, device="cuda") + 0.1
        storage = torch.full((rows + 1, 2560), 91.0, device="cuda", dtype=torch.float16)
        out = storage[:rows]

        def run(name):
            call(name, a, w, residual, gamma, inv, out, rows)

        def change():
            residual.neg_()
    else:
        slots = rows * 10
        ids = torch.rand(rows, 512, device="cuda").topk(10, dim=1).indices.to(torch.int32)
        flat = ids.flatten().long()
        # Sort token/expert pairs without constructing a slots x experts tensor.
        order = torch.argsort(flat, stable=True)
        counts = torch.bincount(flat, minlength=512).to(torch.int32)
        offsets = (counts.cumsum(0) - counts).to(torch.int32)
        relative = torch.empty(slots, device="cuda", dtype=torch.int32)
        relative[order] = torch.arange(
            slots, device="cuda", dtype=torch.int32
        ) - torch.repeat_interleave(offsets, counts.long())
        relative = relative.reshape(rows, 10)
        a = torch.randint(-128, 128, (rows, 2560), device="cuda", dtype=torch.int8)
        scale = torch.full((rows, 1), 0.01, device="cuda", dtype=torch.float16)
        routed = torch.empty(slots, 2560, device="cuda", dtype=torch.int8)
        outputs = [
            torch.empty(40, slots, 8, device="cuda", dtype=torch.uint32) for _ in range(2)
        ] + [
            torch.empty(slots, device="cuda", dtype=torch.int32),
            torch.empty(slots, 1, device="cuda", dtype=torch.float16),
            torch.empty_like(ids),
        ]

        def run(name):
            if name == "fused":
                call(name, a, scale, ids, relative, offsets, *outputs, rows)
            else:
                call(
                    "separate",
                    a,
                    scale,
                    ids,
                    relative,
                    offsets,
                    routed,
                    outputs[3],
                    outputs[4],
                    rows,
                )
                call("pack", routed, *outputs[:3], slots)

        def change():
            a.bitwise_not_()

        def result():
            return [v.clone() for v in outputs]

        def guard():
            return True

        return run, result, change, guard

    def result():
        return [out.clone()]

    def guard():
        return bool((storage[-1] == 91).all())

    return run, result, change, guard


def tune(output):
    import torch
    import tilelang
    from kernels.model.rows import explicit_rows
    from tools.operators.common import configure, benchmark, export_kernel, environment

    configure()
    report = dict(complete=False, identity=environment(), families={})
    for kind, names in [
        ("router", [str(i) for i in range(len(ROUTER_TILES))]),
        ("hc", ["general", "aligned"]),
        ("dispatch", ["separate", "fused"]),
    ]:
        kernels = {}
        for name in names + (["pack"] if kind == "dispatch" else []):
            kernels[name] = tilelang.compile(
                explicit_rows(primitive(kind, name)),
                out_idx=[],
                execution_backend="nvrtc",
                target={"kind": "cuda", "arch": "sm_87"},
            )
            export_kernel(kernels[name], output / "aot" / kind / name)
        records = []
        for rows in samples():
            run, result, change, guard = inputs(kind, rows, kernels)
            base = names[0]
            run(base)
            torch.cuda.synchronize()
            expected = result()
            times = {}
            for name in names:
                if name == "aligned" and rows % 128:
                    continue
                timing, graph = benchmark(lambda: run(name), repetitions=12)
                assert all(torch.equal(x, y) for x, y in zip(result(), expected)), (
                    kind,
                    rows,
                    name,
                    "numeric",
                )
                assert guard(), (kind, rows, name, "tail")
                change()
                run(base)
                torch.cuda.synchronize()
                updated = result()
                graph.replay()
                torch.cuda.synchronize()
                assert all(torch.equal(x, y) for x, y in zip(result(), updated)), (
                    kind,
                    rows,
                    name,
                    "replay",
                )
                change()
                run(base)
                torch.cuda.synchronize()
                times[name] = timing["median_ms"]
            records.append(dict(rows=rows, times=times))
            print(kind, rows, min(times, key=times.get), flush=True)
        report["families"][kind] = dict(records=records, **ranges(records))
        write_json(output / "results.json", report)
    report["complete"] = True
    write_json(output / "results.json", report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    tune(p.parse_args().output)


if __name__ == "__main__":
    main()
