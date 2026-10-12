"""Exact integer oracle, output tails and graph replay for dense INT8."""

import argparse
from pathlib import Path
import subprocess

import torch

from kernels.model.int8_projection import int8_projection, int8_small_projection
from tools.operators.common import configure, benchmark, export_kernel, error, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    configure()
    cases = []
    for m, n, k in ((1, 48, 2560), (3, 65, 128), (17, 129, 640)):
        for dtype in ("float16", "float32"):
            x = torch.randint(-128, 128, (m, k), device="cuda", dtype=torch.int8)
            w = torch.randint(-128, 128, (n, k), device="cuda", dtype=torch.int8)
            ws = torch.full((n,), 0.015625, device="cuda", dtype=torch.float16)
            ts = torch.full((m,), 0.0078125, device="cuda", dtype=torch.float16)
            out = torch.full((m + 1, n), 99.0, device="cuda", dtype=getattr(torch, dtype))
            kernel = int8_projection(m, n, k, dtype)

            def run():
                kernel(x, w, ws, ts, out[:m])

            timing, graph = benchmark(run, repetitions=5)
            expected = (
                ((x.double() @ w.double().T).float() * ws.float()[None, :]) * ts.float()[:, None]
            ).to(out.dtype)
            assert torch.equal(out[:m], expected), error(out[:m], expected)
            assert bool((out[-1] == 99).all())
            saved = x.clone()
            x.zero_()
            out[:m].fill_(99)
            graph.replay()
            torch.cuda.synchronize()
            assert bool((out[:m] == 0).all())
            x.copy_(saved)
            ws.mul_(2)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out[:m], expected * 2)
            folder = a.output / f"M{m}-N{n}-K{k}-{dtype}"
            exported = export_kernel(kernel, folder)
            sass = subprocess.check_output(
                ["/usr/local/cuda/bin/cuobjdump", "--dump-sass", str(folder / "kernel.cubin")],
                text=True,
            )
            assert "IMMA.16832.S8.S8" in sass
            cases.append(
                {
                    "shape": [m, n, k],
                    "output_dtype": dtype,
                    "exact_integer_oracle": True,
                    "tail_guard": True,
                    "changed_input_and_scale_graph": True,
                    "timing": timing,
                    "export": exported,
                }
            )
    for m, n, k in ((1, 48, 2560), (3, 49, 2560), (5, 1, 128), (8, 127, 640), (17, 48, 2560)):
        for dtype in ("float16", "float32"):
            x = torch.randint(-128, 128, (m, k), device="cuda", dtype=torch.int8)
            w = torch.randint(-128, 128, (n, k), device="cuda", dtype=torch.int8)
            ws = torch.rand(n, device="cuda").half() * 0.017
            ts = torch.rand(m, device="cuda").half() * 0.009
            storage = torch.full((m * n + 17,), 99.0, device="cuda", dtype=getattr(torch, dtype))
            out = storage[: m * n].view(m, n)
            kernel = int8_small_projection(m, n, k, dtype)

            def run_small():
                kernel(x.view(torch.int32), w.view(torch.int32), ws, ts, out)

            def oracle():
                return (
                    ((x.double() @ w.double().T).float() * ws.float()[None, :])
                    * ts.float()[:, None]
                ).to(out.dtype)

            timing, graph = benchmark(run_small, repetitions=8)
            assert torch.equal(out, oracle()), error(out, oracle())
            for changed in ("activation", "weight", "scale", "zero"):
                if changed == "activation":
                    x.bitwise_not_()
                elif changed == "weight":
                    w.bitwise_not_()
                elif changed == "scale":
                    ws.mul_(1.5)
                    ts.mul_(0.75)
                else:
                    x.zero_()
                out.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                assert torch.equal(out, oracle()), error(out, oracle())
                assert bool((storage[m * n :] == 99).all())
            cases.append(
                {
                    "shape": [m, n, k],
                    "output_dtype": dtype,
                    "small_dp4a": True,
                    "exact_integer_oracle": True,
                    "tail_guard": True,
                    "changed_input_weight_and_scale_graph": True,
                    "timing": timing,
                    "export": export_kernel(kernel, a.output / f"small-M{m}-N{n}-K{k}-{dtype}"),
                }
            )
    write_json(a.output / "results.json", {"complete": True, "cases": cases, "scope": __doc__})


if __name__ == "__main__":
    main()
