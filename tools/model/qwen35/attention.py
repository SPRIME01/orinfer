"""Offline owner-attention selection using the existing row computation."""

import hashlib
from pathlib import Path

from tools.model.row_selection import aggregate_winner

COLUMNS = {
    name: i
    for i, name in enumerate(
        ("K", "V", "KS", "VS", "Positions", "Lengths", "Partial", "Maximum", "Denominator")
    )
}
OWNERS = (1, 2, 3, 4, 8, 17, 32, 64, 128)


def candidates(query_heads, kv_heads, reference):
    if min(query_heads, kv_heads) <= 0 or query_heads % kv_heads:
        raise ValueError("Invalid owner-attention heads")
    keys = ("block_m", "block_n", "head_group", "max_splits", "threads", "stages")
    original = {key: reference[key] for key in keys if key in reference}
    base = dict(block_m=16, head_group=query_heads // kv_heads, max_splits=16)
    options = [original] + [
        dict(base, block_n=bn, threads=threads, stages=1)
        for bn, threads in ((32, 128), (64, 128), (64, 256))
    ]
    return [value for i, value in enumerate(options) if value not in options[:i]]


def select(tuner, context, reference):
    import torch
    import tilelang
    from kernels.model.independent_rows import independent_rows
    from tools.model.qwen35.parallel import factory
    from tools.model.row_tuning import launch, ATTENTION_TARGET

    qh, kvh = tuner.qh, tuner.kvh
    configs = candidates(qh, kvh, reference)
    lengths = sorted({min(context, n) for n in (256, 2048, 8192)})
    text = dict(num_attention_heads=qh, num_key_value_heads=kvh)

    def build(config):
        compiled = []
        for role in ("attention", "attention_merge"):
            fn = factory(role, context, config, text)
            modes = {
                b.name: ("private" if b.name in COLUMNS else "row", b.name)
                for b in fn.buffer_map.values()
            }
            fn, names = independent_rows(
                fn,
                128,
                modes,
                COLUMNS,
                dynamic=True,
                batch_parameters=("independent_instances",),
            )
            compiled.append(
                (
                    tilelang.compile(
                        fn,
                        out_idx=[],
                        execution_backend="nvrtc",
                        target={"kind": "cuda", "arch": "sm_87"},
                    ),
                    names,
                )
            )
        return compiled

    baseline = build(configs[0])
    partials = max(qh // c["head_group"] * 2 * ATTENTION_TARGET * c["block_m"] for c in configs)

    def inputs(owners):
        values = dict(
            Q=torch.randn(owners + 1, qh, 256, device="cuda", dtype=torch.float16) * 0.1,
            Gate=torch.randn(owners + 1, qh, 256, device="cuda", dtype=torch.float16),
            K=torch.randint(
                -100, 101, (owners, max(lengths), kvh * 256), device="cuda", dtype=torch.int8
            ),
            V=torch.randint(
                -100, 101, (owners, max(lengths), kvh * 256), device="cuda", dtype=torch.int8
            ),
            KS=torch.rand(owners, max(lengths), kvh * 4, device="cuda", dtype=torch.float16) * 0.002
            + 0.001,
            VS=torch.rand(owners, max(lengths), kvh * 4, device="cuda", dtype=torch.float16) * 0.002
            + 0.001,
            Positions=torch.empty(owners, 1, device="cuda", dtype=torch.int32),
            Lengths=torch.empty(owners, 1, device="cuda", dtype=torch.int32),
            Partial=torch.full((owners + 1, partials * 256), 91, device="cuda"),
            Maximum=torch.full((owners + 1, partials), 91, device="cuda"),
            Denominator=torch.full((owners + 1, partials), 91, device="cuda"),
        )
        table = torch.tensor(
            [[values[name][i].data_ptr() for name in COLUMNS] for i in range(owners)],
            device="cuda",
            dtype=torch.uint64,
        )
        storage = torch.full((owners + 1, qh, 256), 91, device="cuda", dtype=torch.float16)
        expected = torch.empty_like(storage)

        def runner(pair, output):
            def run():
                for kernel, names in pair:
                    args = [
                        table
                        if name is None
                        else owners
                        if name == "__rows__"
                        else output
                        if name == "Y"
                        else values[name]
                        for name in names
                    ]
                    launch(kernel, *args)

            return run

        reference_run = runner(baseline, expected)
        for length in lengths:
            actual = length - torch.arange(owners, device="cuda", dtype=torch.int32).remainder(
                min(17, length)
            )
            values["Lengths"][:, 0].copy_(actual)
            values["Positions"][:, 0].copy_(actual - 1)

            def fixture(pair, config):
                run = runner(pair, storage)

                def check():
                    reference_run()
                    torch.testing.assert_close(
                        storage[:owners], expected[:owners], atol=4e-5, rtol=0.005
                    )
                    assert bool(torch.all(storage[owners:] == 91))
                    assert all(
                        bool(torch.all(values[name][owners:] == 91))
                        for name in ("Partial", "Maximum", "Denominator")
                    )

                return run, check, lambda: values["Q"].neg_()

            yield fixture

    measured = tuner.select(
        dict(
            kind="private_owner_attention",
            query_heads=qh,
            kv_heads=kvh,
            context=context,
            lengths=lengths,
            source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            output_atol=4e-5,
            output_rtol=0.005,
            target_tiles=ATTENTION_TARGET,
        ),
        configs,
        build,
        inputs,
        measure_rows=list(OWNERS),
    )
    # One measured geometry serves every owner count; no runtime batch ranges.
    return configs[int(aggregate_winner(measured["measurements"]))]
