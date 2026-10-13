"""Bind a device-resident logical row count without changing tensor strides."""

from tvm import tirx as tir


def extent(capacity, enabled):
    if type(capacity) is not int or capacity <= 0 or type(enabled) is not bool:
        raise ValueError("Invalid logical row capacity")
    control = tir.decl_buffer((1,), "int32", name="Length")
    return (tir.BufferLoad(control, [0]) if enabled else capacity), control


def bind(function, control, enabled):
    if not enabled:
        return function
    parameter = tir.Var("Length_handle", "handle")
    return tir.PrimFunc(
        [*function.params, parameter],
        function.body,
        function.ret_type,
        {**function.buffer_map, parameter: control},
        function.attrs,
    )


def guard(function, control, axis, factor=1, divisor=1):
    """Skip complete padded CTAs with a uniform condition, keeping barriers safe."""
    grids = []

    def inspect(node):
        if isinstance(node, tir.For) and node.thread_binding is not None:
            if node.thread_binding.thread_tag == "blockIdx." + axis:
                grids.append(node.loop_var)

    tir.stmt_functor.post_order_visit(function.body, inspect)
    if len(grids) != 1:
        raise ValueError("Expected one logical row grid")

    def wrap(node):
        if isinstance(node, tir.SBlock) and node.name_hint == "tilelang_root":
            return tir.SBlock(
                node.iter_vars,
                node.reads,
                node.writes,
                node.name_hint,
                tir.IfThenElse(
                    tir.floordiv(grids[0] * factor, divisor) < tir.BufferLoad(control, [0]),
                    node.body,
                    None,
                ),
                node.init,
                node.alloc_buffers,
                node.match_buffers,
                node.annotations,
            )
        return None

    return tir.PrimFunc(
        function.params,
        tir.stmt_functor.ir_transform(function.body, None, wrap),
        function.ret_type,
        function.buffer_map,
        function.attrs,
    )
