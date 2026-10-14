"""Offline TileLang autotuning with fixed inputs and exact execution checks.

Only integer dot-product tiles are searched. Scales, output rounding and the
model's floating-point reductions are not tuning parameters. No Torch or
autotuner dependency enters the published Rust runtime.
"""

import hashlib
import json
from pathlib import Path

from tools.model.flash_next.decode_policy import q4_configs, w8_configs, swiglu_configs
from tools.model.publication import write_json

TARGET = {"kind": "cuda", "arch": "sm_87"}
EXACT_PASSES = {
    "tl.disable_thread_storage_sync": True,
    "tl.disable_shared_memory_reuse": True,
    "tl.if_stmt_binding_inline_replayable_binds": False,
}


def make_prim(spec, config):
    from kernels.model.rows import explicit_rows

    rows, n, k = spec["rows"], spec["outputs"], spec["inputs"]
    dynamic = spec["dynamic"]
    if spec["kind"] == "q4":
        from kernels.model.q4_projection import q4_decode

        fn = q4_decode(rows, n, k, output_dtype=spec["dtype"], dynamic_rows=dynamic, **config)
    elif spec["kind"] == "w8":
        from kernels.model.int8_projection import int8_projection, int8_small_projection

        if config.get("algorithm") == "dp4a":
            fn = int8_small_projection.__wrapped__(
                rows,
                n,
                k,
                spec["dtype"],
                dynamic,
                block_m=config["block_m"],
                block_n=config["block_n"],
            )
        else:
            knobs = {key: value for key, value in config.items() if key != "algorithm"}
            fn = int8_projection.__wrapped__(
                rows, n, k, spec["dtype"], dynamic_rows=dynamic, **knobs
            )
    elif spec["kind"] == "swiglu":
        from kernels.model.int8_swiglu import int8_swiglu

        fn = int8_swiglu.__wrapped__(
            rows,
            n,
            k,
            config["block_n"],
            dynamic,
            algorithm=config["algorithm"],
            block_m=config["block_m"],
        )
    elif spec["kind"] == "expert":
        from kernels.model.e8p_exact import e8p_exact_grouped

        assignments = rows * 10
        tile = config["block_m"]
        tiles = (assignments + tile - 1) // tile + min(spec["experts"], assignments)
        fn = e8p_exact_grouped.__wrapped__(
            assignments, spec["experts"], tiles, n, k, block_m=tile, dynamic_rows=dynamic
        )
    else:
        raise ValueError("Unsupported decode tuning operator")
    return explicit_rows(fn) if dynamic else fn


def factory(spec):
    # TileLang includes primitive closure values in its compiled/tuning cache
    # key. Avoid capturing mutable tensors or factories in the kernel closure.
    encoded = json.dumps(spec, sort_keys=True)

    def build(algorithm="tensor_core", block_m=16, block_n=64, block_k=128, stages=2):
        decoded = json.loads(encoded)
        cfg = dict(block_m=block_m)
        if decoded["kind"] != "expert":
            cfg.update(algorithm=algorithm, block_n=block_n, block_k=block_k)
        if decoded["kind"] == "q4":
            cfg["stages"] = stages
        return make_prim(decoded, cfg)

    return build


def profile_stream(kernel):
    """Use the capture stream even when an NVRTC adapter defaults to stream 0."""
    import torch

    launch = kernel.adapter.func

    def run(*inputs, stream=None):
        return launch(
            *inputs, stream=torch.cuda.current_stream().cuda_stream if stream is None else stream
        )

    kernel.adapter.func = run
    return kernel


class DecodeTuner:
    def __init__(self, output: Path, cache: Path | None = None):
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        self.cache = cache or output / "tuning-cache"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.results = {}

    def compile(
        self,
        kind,
        rows,
        outputs,
        inputs,
        dtype="float16",
        *,
        dynamic=False,
        experts=512,
        block_m=None,
    ):
        import tilelang
        import torch
        from tilelang.autotuner import AutoTuner
        from tools.operators.common import SEED

        spec = dict(
            kind=kind, rows=rows, outputs=outputs, inputs=inputs, dtype=dtype, dynamic=dynamic
        )
        if kind == "expert":
            spec["experts"] = experts
        if kind == "q4":
            configs = q4_configs(rows)
        elif kind == "w8":
            configs = w8_configs(rows, outputs)
        elif kind == "swiglu":
            configs = swiglu_configs(rows)
        elif kind == "expert":
            configs = [dict(block_m=m) for m in ((8, 16) if block_m is None else (block_m,))]
        else:
            raise ValueError("Unsupported decode tuning operator")
        # Runtime-row fallback is tuned at a non-specialized size and checked
        # at tails and at the allocation bound before the cubin is published.
        sample = min(rows, 17) if dynamic else rows
        source = hashlib.sha256()
        for file in ("decode_tuning.py", "decode_policy.py"):
            source.update(Path(__file__).with_name(file).read_bytes())
        root = Path(__file__).resolve().parents[3]
        for file in (
            "q4_projection.py",
            "int8_projection.py",
            "int8_swiglu.py",
            "e8p_exact.py",
            "rows.py",
        ):
            source.update((root / "kernels/model" / file).read_bytes())
        identity = dict(
            spec=spec,
            configs=configs,
            sample=sample,
            seed=SEED,
            source=source.hexdigest(),
            tilelang=tilelang.__version__,
            cuda=torch.version.cuda,
            device=torch.cuda.get_device_name(),
        )
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        cached = self.cache / f"{key}.json"
        passes = EXACT_PASSES if kind == "expert" else {}
        if cached.exists():
            record = json.loads(cached.read_text())
            if record["identity"] != identity or record["config"] not in configs:
                raise ValueError("Invalid decode tuning cache")
            kernel = tilelang.compile(
                make_prim(spec, record["config"]),
                out_idx=[],
                execution_backend="nvrtc",
                target=TARGET,
                pass_configs=passes,
            )
        else:
            supply, reference, check = self._inputs(spec, sample)
            tuner = AutoTuner.from_kernel(factory(dict(spec, tuning_identity=key)), configs)
            tuner.set_compile_args(
                out_idx=[], target=TARGET, execution_backend="nvrtc", pass_configs=passes
            )
            tuner.set_profile_args(
                supply_prog=supply,
                ref_prog=reference,
                manual_check_prog=check,
                cache_input_tensors=False,
                backend="cudagraph",
            )

            def compile_candidate(**config):
                return profile_stream(tuner._default_compile(**config))

            tuner.jit_compile = compile_candidate
            result = tuner.run(warmup=3, rep=20, timeout=120)
            kernel = profile_stream(result.kernel)
            record = dict(identity=identity, config=result.config, latency_ms=result.latency)
            # The native tuner has validated its profiling inputs; validate
            # additional runtime tails and replay using the selected geometry.
            checked = []
            for live in sorted(set([sample, 1, 5, 7, 9, 33, 127, rows])) if dynamic else [rows]:
                if live > rows:
                    continue
                self._validate(kernel, spec, record["config"], live)
                checked.append(live)
            record["checked_rows"] = checked
            write_json(cached, record)
        self.results[key] = record
        write_json(self.output / "tuning.json", list(self.results.values()))
        return kernel, record["config"]

    @staticmethod
    def _inputs(spec, live):
        """Same mathematical inputs for all candidates; expert maps match tiles."""
        import torch
        from kernels.model.q4_projection import q4_gemv
        from kernels.model.e8p_exact import split_a8, e8p_exact_grouped
        from kernels.model.int8_projection import int8_projection
        from tools.operators.common import SEED

        torch.manual_seed(SEED)
        n, k = spec["outputs"], spec["inputs"]
        device = "cuda"
        a = torch.randint(-128, 128, (live, k), device=device, dtype=torch.int8)
        scales = torch.full((live,), 0.001, device=device, dtype=torch.float16)
        dtype = getattr(torch, spec["dtype"])
        storage = torch.full((live + 2, n), 91, device=device, dtype=dtype)
        out = storage[:live]
        expected = torch.empty_like(out)
        kind = spec["kind"]
        if kind == "q4":
            weight = torch.randint(0, 65536, (n, k // 4), device=device, dtype=torch.int32).to(
                torch.uint16
            )
            ws = torch.full((n,), 0.001, device=device, dtype=torch.float16)
            base = q4_gemv(live, n, k, output_dtype=spec["dtype"], block_m=min(live, 8))
            base.adapter.func(
                a, weight, ws, scales, expected, stream=torch.cuda.current_stream().cuda_stream
            )
            args = [a, weight, ws, scales, out]
        elif kind == "w8":
            weight = torch.randint(-128, 128, (n, k), device=device, dtype=torch.int8)
            ws = torch.full((n,), 0.001, device=device, dtype=torch.float16)
            base = int8_projection(live, n, k, output_dtype=spec["dtype"])
            base.adapter.func(
                a, weight, ws, scales, expected, stream=torch.cuda.current_stream().cuda_stream
            )
            args = [a, weight, ws, scales, out]
        elif kind == "swiglu":
            from kernels.model.flash_next import swiglu

            gate = torch.empty_like(expected)
            up = torch.empty_like(expected)
            wg = torch.randint(-128, 128, (n, k), device=device, dtype=torch.int8)
            wu = torch.randint(-128, 128, (n, k), device=device, dtype=torch.int8)
            gs = torch.full((n,), 0.001, device=device, dtype=torch.float16)
            us = torch.full((n,), 0.0007, device=device, dtype=torch.float16)
            base = int8_projection(live, n, k)
            stream = torch.cuda.current_stream().cuda_stream
            base.adapter.func(a, wg, gs, scales, gate, stream=stream)
            base.adapter.func(a, wu, us, scales, up, stream=stream)
            swiglu(live, n).adapter.func(gate, up, expected, stream=stream)
            args = [a, wg, wu, gs, us, scales, out]
        else:
            e = spec["experts"]
            assignments = live * 10
            a = torch.randint(-128, 128, (assignments, k), device=device, dtype=torch.int8)
            scales = torch.full((assignments,), 0.001, device=device, dtype=torch.float16)
            storage = torch.full((assignments + 2, n), 91, device=device, dtype=dtype)
            out, expected = storage[:assignments], torch.empty_like(storage[:assignments])
            low = torch.empty((assignments, k // 8), device=device, dtype=torch.uint32)
            high, sums = (
                torch.empty_like(low),
                torch.empty(assignments, device=device, dtype=torch.int32),
            )
            weight = torch.randint(
                0, 65536, (e, k // 64, n, 8), device=device, dtype=torch.int32
            ).to(torch.uint16)
            book = torch.randint(0, 2**32, (1, 256, 1), device=device, dtype=torch.int64).to(
                torch.uint32
            )
            ws = torch.full((e, n), 0.001, device=device, dtype=torch.float16)
            # Mix spread routing with a hot expert to exercise both tile
            # padding and repeated rows; the ordered activations stay fixed.
            counts = [0] * e
            counts[0] = live
            for j in range(assignments - counts[0]):
                counts[1 + j % (e - 1)] += 1
            offsets, first = [], 0
            for count in counts:
                offsets.append(first)
                first += count
            count_t = torch.tensor(counts, device=device, dtype=torch.int32)
            offset_t = torch.tensor(offsets, device=device, dtype=torch.int32)
            split_a8(assignments, k).adapter.func(
                a, low, high, sums, stream=torch.cuda.current_stream().cuda_stream
            )

            def maps(bm, capacity):
                te, tr = [], []
                for expert, count in enumerate(counts):
                    for row in range(0, count, bm):
                        te.append(expert)
                        tr.append(row)
                assert len(te) <= capacity
                return [
                    torch.tensor(te + [0] * (capacity - len(te)), device=device, dtype=torch.int32),
                    torch.tensor(tr + [0] * (capacity - len(tr)), device=device, dtype=torch.int32),
                    torch.tensor([len(te)], device=device, dtype=torch.int32),
                ]

            tile_cap = (assignments + 7) // 8 + min(e, assignments)
            te, tr, tc = maps(8, tile_cap)
            args = [low, high, sums, weight, book, ws, scales, count_t, offset_t, te, tr, tc, out]
            base = e8p_exact_grouped(assignments, e, tile_cap, n, k, block_m=8)
            base.adapter.func(*args[:-1], expected, stream=torch.cuda.current_stream().cuda_stream)

        torch.cuda.synchronize()
        scalar = live * 10 if kind == "expert" else live
        if spec["dynamic"]:
            args.append(scalar)

        def supply(params):
            supplied = list(args)
            if kind == "expert":
                # TE's static allocation reflects the compiled BM. The host
                # builds matching tile maps; these are never profiling inputs
                # for a differently sized expert tile.
                capacity = int(params[9].shape[0])
                assignments = spec["rows"] * 10
                bm = next(
                    m
                    for m in (8, 16)
                    if (assignments + m - 1) // m + min(spec["experts"], assignments) == capacity
                )
                supplied[9:12] = maps(bm, capacity)
            return supplied

        def reference(*_):
            out.fill_(91)
            return []

        def check(*_):
            if not torch.equal(out, expected) or not bool(torch.all(storage[scalar:] == 91)):
                raise AssertionError(
                    "Decode tile changed integer/scaling output or wrote beyond live rows"
                )

        return supply, reference, check

    @staticmethod
    def _validate(kernel, spec, config, live):
        import torch

        supply, _, check = DecodeTuner._inputs(spec, live)
        args = supply(kernel.get_profiler()._get_params(with_output=False))
        kernel.adapter.func(*args, stream=torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        check()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            kernel.adapter.func(*args, stream=torch.cuda.current_stream().cuda_stream)
        # Scratch outputs change after capture; replay must actually run.
        out = args[12 if spec["kind"] == "expert" else 6 if spec["kind"] == "swiglu" else 4]
        out.fill_(91)
        graph.replay()
        torch.cuda.synchronize()
        check()
        # Compare replay after changing the live input with a direct launch.
        # The independent arithmetic reference above has already checked the
        # candidate; this additionally proves capture observes updated storage.
        args[0].zero_()
        kernel.adapter.func(*args, stream=torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        changed = out.clone()
        out.fill_(91)
        graph.replay()
        torch.cuda.synchronize()
        if not torch.equal(out, changed):
            raise AssertionError("Decode graph replay ignored the changed live input")
