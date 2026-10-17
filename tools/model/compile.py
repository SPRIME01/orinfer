"""Shared native AOT compilation with explicit runtime row scalars."""

import inspect


def primitive(function, *args, **kwargs):
    import tilelang
    from kernels.model.rows import explicit_rows

    factory = inspect.unwrap(getattr(function, "func", function))
    fn = factory(*args, **kwargs)
    if kwargs.get("dynamic_rows"):
        fn = explicit_rows(fn)
    return tilelang.compile(
        fn,
        out_idx=[],
        execution_backend="nvrtc",
        target={"kind": "cuda", "arch": "sm_87"},
        pass_configs={
            "tl.disable_thread_storage_sync": True,
            "tl.disable_shared_memory_reuse": True,
        }
        if kwargs.get("prefetch_qk")
        else {},
    )
