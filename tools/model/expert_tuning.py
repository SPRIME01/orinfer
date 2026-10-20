"""Shared expert geometry, kernel construction and offline pipeline selection.

Model adapters supply routing/operand fixtures. Candidate and publication paths
use the same factory, so tile capacity, codec, tail policy and compiler passes
cannot diverge between measurement and the emitted execution package.
"""

from tools.model.row_tuning import RowTuner, launch
from tools.model.compile import primitive


EXACT_EXPERT_PASSES = {
    "tl.disable_thread_storage_sync": True,
    "tl.disable_shared_memory_reuse": True,
    "tl.if_stmt_binding_inline_replayable_binds": False,
}


def tile_capacity(rows, experts, top_k, block_m):
    if any(type(n) is not int or n <= 0 for n in (rows, experts, top_k, block_m)):
        raise ValueError("Invalid routed projection geometry")
    slots = rows * top_k
    return (slots + block_m - 1) // block_m + min(experts, slots)


def expert_configs(codec, *, block_m=None, direct=False):
    if codec == "e8p_q2":
        return [dict(block_m=m) for m in ((8, 16) if block_m is None else (block_m,))]
    if codec != "e8p_q4" or block_m is not None:
        raise ValueError("Unsupported routed projection search")
    configs = [
        dict(block_m=16, block_n=64, block_k=128, stages=2, tail_m=0, expand=False),
        dict(block_m=32, block_n=64, block_k=128, stages=1, tail_m=8, expand=False),
        dict(block_m=32, block_n=128, block_k=128, stages=1, tail_m=8, expand=False),
        dict(block_m=32, block_n=64, block_k=128, stages=1, tail_m=0, expand=True),
        dict(block_m=16, block_n=64, block_k=128, stages=2, tail_m=0, expand=True),
        dict(
            block_m=32, block_n=64, block_k=128, stages=1, tail_m=8, expand=False, reuse_shared=True
        ),
    ]
    if direct:
        configs += [
            dict(block_m=16, expand=False, folded=True, minimum_blocks=6, sparse_rows=sparse)
            for sparse in (False, True)
        ]
    return configs


def expert_passes(codec, config):
    if codec == "e8p_q2" or config.get("folded"):
        return EXACT_EXPERT_PASSES
    return {"tl.enable_aggressive_shared_memory_merge": True} if config.get("reuse_shared") else {}


def expert_projection(rows, outputs, inputs, experts, top_k, *, codec, config, dynamic_rows=False):
    import inspect

    tile = config["block_m"]
    tiles = tile_capacity(rows, experts, top_k, tile)
    if codec == "e8p_q2":
        from kernels.model.e8p_exact import e8p_exact_grouped

        return inspect.unwrap(e8p_exact_grouped)(
            rows * top_k, experts, tiles, outputs, inputs, block_m=tile, dynamic_rows=dynamic_rows
        )
    if codec not in ("e8p_q4", "w8"):
        raise ValueError("Unsupported expert codec")
    if config.get("folded"):
        from kernels.model.e8p_prefill import e8p_prefill_grouped

        return inspect.unwrap(e8p_prefill_grouped)(
            rows * top_k,
            experts,
            tiles,
            outputs,
            inputs,
            dynamic_rows=dynamic_rows,
            sparse_rows=config["sparse_rows"],
            codec=codec,
            minimum_blocks=config["minimum_blocks"],
        )
    from kernels.model.moe import integer_experts

    options = {k: v for k, v in config.items() if k not in ("expand", "reuse_shared", "folded")}
    return inspect.unwrap(integer_experts)(
        rows,
        outputs,
        inputs,
        experts,
        top_k,
        codec="w8" if config.get("expand") else codec,
        dynamic_rows=dynamic_rows,
        **options,
    )


class MoeTuner(RowTuner):
    def routed(self, hidden, intermediate, experts, top_k, weights, weight_identity):
        import torch
        from kernels.model import moe
        from kernels.model.e8p_prefill import split_prefill_a8
        from tools.quantization.vq import e8p_selector_supports_parity_xor

        spec = dict(
            kind="routed_q4",
            hidden=hidden,
            intermediate=intermediate,
            experts=experts,
            top_k=top_k,
            weights=weight_identity,
            timed_routing=["independent"],
            validation_routing=["four", "hot"],
        )
        configs = expert_configs(
            "e8p_q4", direct=e8p_selector_supports_parity_xor(weights["book"].cpu().numpy())
        )
        dimensions = [(2 * intermediate, hidden), (hidden, intermediate)]

        def build(config):
            result = dict(
                offsets=primitive(moe.expert_offsets, experts, config["block_m"]),
                tiles=primitive(
                    moe.expert_tiles, 4096, experts, top_k, config["block_m"], dynamic_rows=True
                ),
            )
            if config.get("folded"):
                slots = 4096 * top_k
                result["packs"] = [
                    primitive(split_prefill_a8, slots, k, dynamic_rows=True) for _, k in dimensions
                ]
            result["products"] = [
                primitive(
                    expert_projection,
                    4096,
                    n,
                    k,
                    experts,
                    top_k,
                    codec="e8p_q4",
                    config=config,
                    dynamic_rows=True,
                    compiler_options=expert_passes("e8p_q4", config),
                )
                for n, k in dimensions
            ]
            if config["expand"]:
                result["unpack"] = [
                    primitive(moe.expand_experts, n, k, experts) for n, k in dimensions
                ]
            return result

        reference = build(configs[0])

        def inputs(rows, patterns=("independent",)):
            for pattern in patterns:
                choices = (
                    rows if pattern == "independent" else min(rows, 4 if pattern == "four" else 1)
                )
                ids = torch.stack([torch.randperm(experts)[:top_k] for _ in range(choices)])
                ids = ids[torch.arange(rows) % choices].flatten()
                counts = torch.bincount(ids, minlength=experts).int().cuda()
                data = [
                    torch.randint(-100, 101, (rows * top_k, k), device="cuda", dtype=torch.int8)
                    for _, k in dimensions
                ]
                scale = torch.full((rows * top_k,), 0.001, device="cuda", dtype=torch.float16)
                expected = [
                    torch.empty((rows * top_k, n), device="cuda", dtype=torch.float16)
                    for n, _ in dimensions
                ]

                def runner(kernels, config, outputs):
                    tile = config["block_m"]
                    capacity = (4096 * top_k + tile - 1) // tile + experts
                    offsets = torch.empty(experts, device="cuda", dtype=torch.int32)
                    tile_offsets = torch.empty_like(offsets)
                    tile_count = torch.empty(1, device="cuda", dtype=torch.int32)
                    tile_expert = torch.empty(capacity, device="cuda", dtype=torch.int32)
                    tile_row = torch.empty_like(tile_expert)
                    expanded = torch.empty(
                        experts * max(n * k for n, k in dimensions) if config["expand"] else 1,
                        device="cuda",
                        dtype=torch.int8,
                    )
                    planes = (
                        [
                            torch.empty(
                                rows * top_k * max(hidden, intermediate) // 2,
                                device="cuda",
                                dtype=torch.int8,
                            )
                            for _ in range(2)
                        ]
                        if config.get("folded")
                        else []
                    )
                    sums = torch.empty(rows * top_k, device="cuda", dtype=torch.int32)

                    def run():
                        launch(kernels["offsets"], counts, offsets, tile_offsets, tile_count)
                        launch(kernels["tiles"], counts, tile_offsets, tile_expert, tile_row, rows)
                        for i, product in enumerate(kernels["products"]):
                            w = weights["banks"][i][0]
                            if config.get("folded"):
                                launch(kernels["packs"][i], data[i], *planes, sums, rows * top_k)
                                launch(
                                    product,
                                    *planes,
                                    sums,
                                    w,
                                    weights["book"],
                                    weights["banks"][i][1],
                                    scale,
                                    counts,
                                    offsets,
                                    tile_expert,
                                    tile_row,
                                    tile_count,
                                    outputs[i],
                                    rows * top_k,
                                )
                                continue
                            if config["expand"]:
                                launch(kernels["unpack"][i], w, weights["book"], counts, expanded)
                                w = expanded
                            launch(
                                product,
                                data[i],
                                w,
                                weights["book"],
                                weights["banks"][i][1],
                                scale,
                                counts,
                                offsets,
                                tile_expert,
                                tile_row,
                                tile_count,
                                outputs[i],
                                rows,
                            )

                    return run

                run_reference = runner(reference, configs[0], expected)

                def fixture(kernels, config):
                    storage = [
                        torch.full((rows * top_k + 1, n), 91, device="cuda", dtype=torch.float16)
                        for n, _ in dimensions
                    ]
                    actual = [value[:-1] for value in storage]
                    run = runner(kernels, config, actual)

                    def check():
                        run_reference()
                        assert all(torch.equal(a, b) for a, b in zip(actual, expected)), (
                            "Routed pipeline changed integer projection outputs"
                        )
                        assert all(bool(torch.all(value[-1] == 91)) for value in storage)

                    def change():
                        for value in data:
                            value.neg_()

                    return run, check, change

                yield fixture

        return self.select(
            spec,
            configs,
            build,
            inputs,
            measure_rows=[1, 2, 3, 4, 8, 17, 32, 64, 128, 512, 4096],
            # Independent routes determine the general serving cost, as in
            # vLLM's random gating benchmark. Duplicated-route extremes still
            # undergo exact/tail and changed-input graph checks, but are not
            # assumed to occupy two thirds of real decode iterations.
            validation_inputs=lambda rows: inputs(rows, ("four", "hot")),
        )
