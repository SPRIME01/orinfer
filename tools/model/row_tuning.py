"""Offline selection of shared dynamic-row kernel pipelines.

The published ranges reuse Flash Next's measured row selection machinery.
Every candidate receives the same data, tail guards and changed-input graph
check. Only selected tiles are emitted; no tuning runs in the Rust runtime.
"""

import hashlib
import json
import math
from pathlib import Path
from tools.model.compile import primitive
from tools.model.row_selection import ranges
from tools.model.publication import write_json

ROTATION_TILES = (256, 512, 1024)
ATTENTION_TILES = [(g, m, n) for g in (1, 6) for m in (16, 32, 64) for n in (32, 64)]
ATTENTION_PIPELINES = [
    dict(head_group=1, block_m=64, block_n=64, threads=256, stages=s) for s in (0, 1)
]
ATTENTION_TARGET = 16
ATTENTION_SPLITS = 8
GDN_AB_TILES = [
    dict(BM=16, BN=n, BK=k, threads=t)
    for n, k, t in ((8, 512, 32), (16, 256, 64), (16, 512, 64), (32, 256, 128), (64, 64, 128))
] + [dict(BM=m, BN=64, BK=128, threads=128) for m in (16, 64)]


def launch(kernel, *args):
    import torch

    kernel.adapter.func(*args, stream=torch.cuda.current_stream().cuda_stream)


class RowTuner:
    def __init__(self, output, identity):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        self.records = {}

    def select(self, spec, configs, build, inputs):
        from tools.operators.common import benchmark, SEED
        import torch

        torch.manual_seed(SEED)
        identity = dict(toolchain=self.identity, spec=spec, configs=configs, seed=SEED)
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        path = self.output / (key + ".json")
        if path.exists():
            cached = json.loads(path.read_text())
            if cached["identity"] != identity:
                raise ValueError("Row pipeline tuning cache identity changed")
            return cached
        kernels = [build(c) for c in configs]
        records = []
        for rows in [1, 2, 3, 4, 8, 17, 64, 129, 512, 4096]:
            times = {str(i): [] for i in range(len(configs))}
            for fixture in inputs(rows):
                for index, pair in enumerate(kernels):
                    run, check, change = fixture(pair, configs[index])
                    run()
                    check()
                    pilot, _ = benchmark(run, repetitions=1, warmup=1)
                    repetitions = max(1, min(32, math.ceil(0.25 / pilot["median_ms"])))
                    timing, graph = benchmark(run, repetitions=repetitions, warmup=0)
                    change()
                    graph.replay()
                    check()
                    change()
                    times[str(index)].append(timing["median_ms"])
            record = dict(
                rows=rows,
                times={k: math.exp(sum(map(math.log, v)) / len(v)) for k, v in times.items()},
            )
            records.append(record)
            print("ROW TUNING", spec, record, flush=True)
        chosen = ranges(records, capacity=4096)
        result = dict(identity=identity, configs=configs, measurements=records, **chosen)
        write_json(path, result)
        return result

    def rotation(self, k, swiglu, *, plane_rows=0):
        from kernels.model.rotation_a8 import (
            rotate_activation_tiles,
            quantize_rotation_tiles,
            rotate_activation_a8,
        )
        import torch

        spec = dict(kind="rotation", k=k, swiglu=swiglu, plane_rows=plane_rows)
        if str(spec) in self.records:
            return self.records[str(spec)]
        configs = [dict(block_k=b) for b in ROTATION_TILES]
        reference = primitive(rotate_activation_a8, 4096, k, swiglu=swiglu, dynamic_rows=True)

        def build(c):
            b = c["block_k"]
            return (
                primitive(
                    rotate_activation_tiles,
                    4096,
                    k,
                    b,
                    swiglu,
                    dynamic_rows=True,
                    plane_rows=plane_rows,
                ),
                primitive(quantize_rotation_tiles, 4096, k, b, dynamic_rows=True),
            )

        def inputs(rows):
            shape = (plane_rows + rows, k) if plane_rows else (rows, k * (2 if swiglu else 1))
            x = torch.randn(shape, device="cuda", dtype=torch.float16) * 0.1
            signs = torch.randint(0, 2, (k,), device="cuda", dtype=torch.int8) * 2 - 1
            eq = torch.empty((rows, k), device="cuda", dtype=torch.int8)
            es = torch.empty(rows, device="cuda", dtype=torch.float16)

            def fixture(pair, c):
                first, second = pair
                b = c["block_k"]
                rotated = torch.empty((rows, k), device="cuda", dtype=torch.float16)
                maxima = torch.empty((rows, (k + b - 1) // b), device="cuda")
                storage = torch.full((rows + 1, k), 91, device="cuda", dtype=torch.int8)
                scales = torch.full((rows + 1,), 91, device="cuda", dtype=torch.float16)
                q, s = storage[:rows], scales[:rows]

                def run():
                    launch(first, x, signs, rotated, maxima, rows)
                    launch(second, rotated, maxima, q, s, rows)

                def check():
                    original = (
                        torch.cat((x[:rows], x[plane_rows : plane_rows + rows]), dim=1)
                        if plane_rows
                        else x
                    )
                    launch(reference, original, signs, eq, es, rows)
                    assert torch.equal(q, eq) and torch.equal(s, es), (
                        "Rotation pipeline changed codes/scales"
                    )
                    assert bool(torch.all(storage[rows:] == 91)) and bool(
                        torch.all(scales[rows:] == 91)
                    )

                return run, check, lambda: x.neg_()

            yield fixture

        result = self.select(spec, configs, build, inputs)
        self.records[str(spec)] = result
        return result

    def attention(self, context):
        from kernels.model.attention_rows import causal_attention_int8, merge_attention_int8
        from kernels.model.kv_int8 import attention_prefill_int8
        import torch

        spec = dict(
            kind="attention",
            context=context,
            target_tiles=ATTENTION_TARGET,
            max_splits=ATTENTION_SPLITS,
        )
        if str(spec) in self.records:
            return self.records[str(spec)]
        configs = [
            dict(head_group=g, block_m=m, block_n=n, max_splits=ATTENTION_SPLITS)
            for g, m, n in ATTENTION_TILES
        ] + [c | dict(max_splits=ATTENTION_SPLITS) for c in ATTENTION_PIPELINES]
        # Include the unsplit reference geometry in the same computation body.
        # This removes split bookkeeping at compile time, without a row cutoff.
        configs.append(dict(head_group=1, block_m=32, block_n=32, max_splits=1))
        reference = primitive(attention_prefill_int8, 1, 4096, context, dynamic_rows=True)

        def build(c):
            return (
                primitive(
                    causal_attention_int8,
                    4096,
                    context,
                    c["block_m"],
                    c["block_n"],
                    ATTENTION_TARGET,
                    c["max_splits"],
                    head_group=c["head_group"],
                    threads=c.get("threads", 128),
                    stages=c.get("stages", 0),
                ),
                primitive(
                    merge_attention_int8,
                    4096,
                    c["block_m"],
                    ATTENTION_TARGET,
                    c["max_splits"],
                    head_group=c["head_group"],
                ),
            )

        def inputs(rows):
            # Autoregressive positions usually end inside a tile. Exact tile
            # multiples alone hide loading/masking and instruction-cache costs.
            for length in sorted({max(rows, 536), max(rows, 8216)}):
                q = torch.randn((rows, 24, 256), device="cuda", dtype=torch.float16) * 0.1
                gate = torch.randn_like(q)
                kv = [
                    torch.randint(-127, 128, (1, context, 4, 256), device="cuda", dtype=torch.int8)
                    for _ in range(2)
                ]
                scales = [
                    torch.full((1, context, 4, 4), 0.001, device="cuda", dtype=torch.float16)
                    for _ in range(2)
                ]
                pos = torch.arange(length - rows, length, device="cuda", dtype=torch.int32)
                lengths = torch.tensor([length], device="cuda", dtype=torch.int32)
                expected = torch.empty_like(q)

                def fixture(
                    pair,
                    c,
                    q=q,
                    gate=gate,
                    kv=kv,
                    scales=scales,
                    pos=pos,
                    lengths=lengths,
                    expected=expected,
                ):
                    first, second = pair
                    bm = c["block_m"]
                    heads = 24 // c["head_group"]
                    partial = torch.empty((heads, 2 * ATTENTION_TARGET, bm, 256), device="cuda")
                    maximum = torch.empty((heads, 2 * ATTENTION_TARGET, bm), device="cuda")
                    denom = torch.empty_like(maximum)
                    storage = torch.full(
                        (rows + 1, 24, 256), 91, device="cuda", dtype=torch.float16
                    )
                    out = storage[:rows]

                    def run():
                        launch(
                            first,
                            q,
                            *kv,
                            *scales,
                            gate,
                            pos,
                            lengths,
                            out,
                            partial,
                            maximum,
                            denom,
                            rows,
                        )
                        launch(second, partial, maximum, denom, gate, out, rows)

                    def check():
                        launch(reference, q, *kv, gate, pos, lengths, expected, *scales, rows)
                        torch.testing.assert_close(out, expected, atol=4e-5, rtol=0.005)
                        assert bool(torch.all(storage[rows:] == 91))

                    return run, check, lambda: q.neg_()

                yield fixture

        result = self.select(spec, configs, build, inputs)
        self.records[str(spec)] = result
        return result

    def gdn_ab(self):
        """Tune CTA coverage while preserving the FP16 MMA K accumulation order."""
        import tilelang.language as T
        import torch
        from kernels.operators.op07_gdn_ab import gdn_ab_tensorcore

        spec = dict(kind="gdn_ab", k=5120, n=96)
        if str(spec) in self.records:
            return self.records[str(spec)]
        reference = primitive(gdn_ab_tensorcore, T.dynamic("m"))

        def build(config):
            return primitive(gdn_ab_tensorcore, T.dynamic("m"), **config)

        def inputs(rows):
            x = torch.randn((rows, 5120), device="cuda", dtype=torch.float16) * 0.1
            w = torch.randn((96, 5120), device="cuda", dtype=torch.float16) * 0.1
            expected = torch.empty((rows, 96), device="cuda", dtype=torch.float16)

            def fixture(kernel, _config):
                storage = torch.full((rows + 1, 96), 91, device="cuda", dtype=torch.float16)
                out = storage[:rows]

                def run():
                    launch(kernel, x, w, out)

                def check():
                    launch(reference, x, w, expected)
                    assert torch.equal(out, expected), "GDN A/B tile changed the FP16 result"
                    assert bool(torch.all(storage[rows:] == 91))

                return run, check, lambda: x.neg_()

            yield fixture

        result = self.select(spec, GDN_AB_TILES, build, inputs)
        self.records[str(spec)] = result
        return result

    def gdn_sequence(self):
        """Select loop unrolling without changing the sequential FP32 recurrence."""
        import torch
        from kernels.model.gdn_sequence import _gdn_sequence

        spec = dict(kind="gdn_sequence", value_tile=16, threads=32, prefetch_qk=True)
        if str(spec) in self.records:
            return self.records[str(spec)]
        configs = [dict(token_unroll=u) for u in (1, 2, 4, 8)]

        def build(config):
            return primitive(
                _gdn_sequence,
                4096,
                in_place=True,
                value_tile=16,
                threads=32,
                prefetch_qk=True,
                dynamic_rows=True,
                **config,
            )

        reference = build(configs[0])

        def inputs(rows):
            q = torch.nn.functional.normalize(
                torch.randn((16, rows, 128), device="cuda"), dim=-1
            ).half()
            k = torch.nn.functional.normalize(torch.randn_like(q.float()), dim=-1).half()
            v = torch.randn((48, rows, 128), device="cuda", dtype=torch.float16) * 0.1
            g = -torch.rand((rows, 48), device="cuda") * 0.1
            beta = torch.rand_like(g)
            initial = torch.randn((48, 128, 128), device="cuda") * 0.01
            expected_state = torch.empty_like(initial)
            expected = torch.empty((rows, 6144), device="cuda", dtype=torch.float16)
            scratch = torch.empty(1, device="cuda")

            def fixture(kernel, _config):
                state = torch.empty_like(initial)
                storage = torch.full((rows + 1, 6144), 91, device="cuda", dtype=torch.float16)
                out = storage[:rows]

                def run():
                    state.copy_(initial)
                    launch(kernel, q, k, v, g, beta, state, scratch, out, rows)

                def check():
                    expected_state.copy_(initial)
                    launch(reference, q, k, v, g, beta, expected_state, scratch, expected, rows)
                    assert torch.equal(state, expected_state) and torch.equal(out, expected), (
                        "GDN unrolling changed the FP32 state or FP16 output"
                    )
                    assert bool(torch.all(storage[rows:] == 91))

                return run, check, lambda: v.neg_()

            yield fixture

        result = self.select(spec, configs, build, inputs)
        self.records[str(spec)] = result
        return result
