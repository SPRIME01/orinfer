"""Verify row-generic expert rotation, dispatch and exact A8 plane fusion."""

import argparse
from pathlib import Path

import tilelang
import torch

from kernels.model.e8p_exact import split_a8
from kernels.model.moe import expert_dispatch
from kernels.model.rotation_a8 import rotate_activation_a8
from kernels.model.rows import explicit_rows
from tools.model.flash_next.decode_tuning import profile_stream
from tools.operators.common import configure, write_json


def compile_dynamic(fn):
    return profile_stream(
        tilelang.compile(
            explicit_rows(fn),
            out_idx=[],
            execution_backend="nvrtc",
            target={"kind": "cuda", "arch": "sm_87"},
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure()
    torch.manual_seed(20261002)
    report = dict(complete=False, cases=[])
    for down in [False, True]:
        capacity, width = (1280, 640) if down else (128, 2560)
        candidate = compile_dynamic(
            rotate_activation_a8.__wrapped__(
                capacity,
                width,
                swiglu=down,
                dynamic_rows=True,
                output_layout="e8p_rows" if down else "e8p_scatter",
            )
        )
        rotation = compile_dynamic(
            rotate_activation_a8.__wrapped__(capacity, width, swiglu=down, dynamic_rows=True)
        )
        pack = compile_dynamic(split_a8.__wrapped__(1280, width, dynamic_rows=True))
        dispatch = (
            None
            if down
            else compile_dynamic(
                expert_dispatch.__wrapped__(128, width, scale_group=width, dynamic_rows=True)
            )
        )
        x = torch.randn(capacity, width * (2 if down else 1), device="cuda", dtype=torch.float16)
        signs = torch.randint(0, 2, (width,), device="cuda", dtype=torch.int8) * 2 - 1
        q = torch.empty(capacity, width, device="cuda", dtype=torch.int8)
        scale = torch.empty(capacity, 1, device="cuda", dtype=torch.float16)
        actual_q, actual_scale = torch.empty_like(q), torch.empty_like(scale)
        expected = [
            torch.empty(1280, width // 8, device="cuda", dtype=torch.uint32),
            torch.empty(1280, width // 8, device="cuda", dtype=torch.uint32),
            torch.empty(1280, device="cuda", dtype=torch.int32),
        ]
        actual = [torch.empty_like(t) for t in expected]
        routed = torch.empty(1280, width, device="cuda", dtype=torch.int8)
        for rows in [1, 2, 3, 4, 8, 9, 16, 17, 32, 33, 127, 128]:
            live = rows * 10 if down else rows
            routing = []
            if not down:
                ids = torch.rand(rows, 512, device="cuda").topk(10, dim=1).indices.to(torch.int32)
                flat = ids.flatten().long()
                counts = torch.bincount(flat, minlength=512)
                offsets = (counts.cumsum(0) - counts).to(torch.int32)
                one_hot = torch.nn.functional.one_hot(flat, 512).to(torch.int32)
                relative = (
                    one_hot.cumsum(0, dtype=torch.int32).gather(1, flat[:, None]).flatten() - 1
                ).reshape(rows, 10)
                expected_ds = torch.empty(rows * 10, 1, device="cuda", dtype=torch.float16)
                actual_ds = torch.empty_like(expected_ds)
                expected_map, actual_map = torch.empty_like(ids), torch.empty_like(ids)
                routing = [ids, relative, offsets, actual_ds, actual_map]

            def reference():
                rotation.adapter.func(x, signs, q, scale, live)
                if down:
                    pack.adapter.func(q, *expected, live)
                else:
                    dispatch.adapter.func(
                        q, scale, ids, relative, offsets, routed, expected_ds, expected_map, rows
                    )
                    pack.adapter.func(routed, *expected, rows * 10)

            def fused():
                candidate.adapter.func(x, signs, actual_q, actual_scale, *actual, *routing, live)

            def check():
                torch.cuda.synchronize()
                assert torch.equal(q[:live], actual_q[:live])
                assert torch.equal(scale[:live], actual_scale[:live])
                for a, b in zip(actual, expected, strict=True):
                    assert torch.equal(a[: rows * 10], b[: rows * 10]), (down, rows)
                    assert bool((a[rows * 10 :] == 42).all())
                if not down:
                    assert torch.equal(actual_ds, expected_ds) and torch.equal(
                        actual_map, expected_map
                    )

            for a in actual:
                a.fill_(42)
            reference()
            fused()
            check()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fused()
            x.mul_(-0.75)
            for a in actual:
                a.fill_(42)
            reference()
            graph.replay()
            check()
            report["cases"].append(
                dict(down=down, rows=rows, exact=True, tail_intact=True, replay=True)
            )
            print("PASS EXPERT FUSION", down, rows, flush=True)
    report["complete"] = True
    write_json(args.output / "results.json", report)


if __name__ == "__main__":
    main()
