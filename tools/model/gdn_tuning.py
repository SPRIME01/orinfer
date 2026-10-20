"""Select a complete request-state strategy over measured decode cohorts."""

import inspect
import math

from tools.model.row_tuning import launch


def select(tuner, dense):
    import torch
    import tilelang
    from kernels.model.gdn_sequence import _gdn_sequence
    from kernels.model.gdn_deferred import recurrent
    from kernels.model.independent_rows import independent_rows
    from tools.model.qwen35.parallel import geometry

    hk, hv = tuner.hk, tuner.hv
    configs = [dict(deferred=False)] + [
        dict(deferred=True, capacity=4, value_tile=vt, threads=threads)
        for vt, threads in [(16, 64)]
    ]

    def compile_function(fn, prefetch=False):
        private = {"State", "LogK", "LogD", "LogG", "Count"}
        names = {b.name for b in fn.buffer_map.values()} & private
        columns = {name: i for i, name in enumerate(sorted(names))}
        roles = {
            b.name: ("private", b.name) if b.name in private else ("row", b.name)
            for b in fn.buffer_map.values()
        }
        fn, inputs = independent_rows(fn, 128, roles, columns, dynamic=True)
        passes = (
            {"tl.disable_thread_storage_sync": True, "tl.disable_shared_memory_reuse": True}
            if prefetch
            else {}
        )
        kernel = tilelang.compile(
            fn,
            out_idx=[],
            execution_backend="nvrtc",
            target={"kind": "cuda", "arch": "sm_87"},
            pass_configs=passes,
        )
        return kernel, inputs, columns

    def build(config):
        if not config["deferred"]:
            return None
        return compile_function(
            recurrent(
                config["capacity"],
                key_heads=hk,
                value_heads=hv,
                value_tile=config["value_tile"],
                threads=config["threads"],
            )
        )

    def inputs(rows):
        config = geometry(dense, rows)
        reference = compile_function(
            inspect.unwrap(_gdn_sequence)(1, in_place=True, key_heads=hk, value_heads=hv, **config),
            config["prefetch_qk"],
        )
        q = torch.nn.functional.normalize(torch.randn(rows, hk, 1, 128, device="cuda"), dim=-1)
        values = dict(
            Q=q.half(),
            K=torch.nn.functional.normalize(torch.randn_like(q), dim=-1).half(),
            V=torch.randn(rows, hv, 1, 128, device="cuda").half() * 0.1,
            G=-torch.rand(rows, 1, hv, device="cuda") * 0.1,
            Beta=torch.rand(rows, 1, hv, device="cuda"),
        )
        initial = torch.randn(rows, hv, 128, 128, device="cuda") * 0.01
        expected = torch.empty(rows, 1, hv * 128, device="cuda", dtype=torch.float16)
        expected_state = torch.empty_like(initial)

        def runner(compiled, state, out, logs):
            kernel, names, columns = compiled
            private = dict(State=state, **logs)
            table = torch.tensor(
                [[private[name][i].data_ptr() for name in columns] for i in range(rows)],
                device="cuda",
                dtype=torch.uint64,
            )
            args = [
                table
                if name is None
                else rows
                if name == "__rows__"
                else out
                if name == "Out"
                else values[name]
                for name in names
            ]

            def run():
                state[:rows].copy_(initial)
                if logs:
                    logs["Count"][:rows].zero_()
                # Equal complete cycles, including the dense merge. Reset and
                # state copies are charged identically to both strategies.
                for _ in range(4):
                    launch(kernel, *args)

            return run

        run_reference = runner(reference, expected_state, expected, {})

        def fixture(compiled, config):
            state = torch.full((rows + 1, hv, 128, 128), 91.0, device="cuda")
            out = torch.full((rows + 1, 1, hv * 128), 91, device="cuda", dtype=torch.float16)
            logs = {}
            if config["deferred"]:
                for name, shape, dtype in [
                    ("LogK", [4, hk, 128], torch.float16),
                    ("LogD", [4, hv, 128], torch.float32),
                    ("LogG", [4, hv], torch.float32),
                    ("Count", [hv, 128], torch.int32),
                ]:
                    logs[name] = torch.full((rows + 1, *shape), 91, device="cuda", dtype=dtype)
            run = runner(compiled or reference, state, out, logs)

            def check():
                run_reference()
                assert torch.equal(state[:rows], expected_state)
                assert torch.equal(out[:rows], expected)
                assert bool(torch.all(state[rows:] == 91)) and bool(torch.all(out[rows:] == 91))
                assert all(bool(torch.all(value[rows:] == 91)) for value in logs.values())

            return run, check, lambda: values["V"].neg_()

        yield fixture

    result = tuner.select(
        dict(kind="gdn_state_strategy", key_heads=hk, value_heads=hv, cycles=4, dense=dense),
        configs,
        build,
        inputs,
        measure_rows=[1, 2, 3, 4, 8, 17, 32, 64, 128],
    )
    # Logs cannot change representation when batch membership changes. Select
    # one whole strategy by relative geometric mean, retaining the measured
    # dense policy if no log strategy wins across these cohorts.
    scores = {
        name: sum(math.log(row["times"][name]) for row in result["measurements"])
        for name in result["measurements"][0]["times"]
    }
    chosen = configs[int(min(scores, key=scores.get))]
    if not chosen["deferred"]:
        return dense
    return dict(configs=[chosen], default="0", selections=[])
