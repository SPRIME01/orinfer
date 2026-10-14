"""Signed block128 rotation and row A8 with exact FP16 materialization boundaries."""

import tilelang
import tilelang.language as T
from tools.operators.common import orin_jit
from kernels.model.rows import row_count


@orin_jit
def rotate_activation_a8(
    M: int,
    K: int,
    swiglu=False,
    reciprocal=True,
    dynamic_rows: bool = False,
    output_layout: str = "a8",
):
    if type(M) is not int or M <= 0 or type(K) is not int or K <= 0 or K % 128:
        raise ValueError("Invalid rotation dimensions")
    if output_layout not in ("a8", "e8p", "e8p_dispatch", "e8p_rows", "e8p_scatter"):
        raise ValueError("Invalid activation output layout")
    exact = output_layout != "a8"
    scatter = output_layout == "e8p_scatter"
    dispatch = output_layout in ("e8p_dispatch", "e8p_scatter")
    if output_layout in ("e8p", "e8p_dispatch") and (
        dynamic_rows
        or (M, K, swiglu, dispatch)
        not in (
            (1, 2560, False, True),
            (10, 640, True, False),
        )
    ):
        raise ValueError("Unsupported decode activation fusion")
    if output_layout == "e8p_rows" and (M > 1280 or K != 640 or not swiglu):
        raise ValueError("Unsupported row-tiled expert down fusion")
    if scatter and (M > 128 or K != 2560 or swiglu):
        raise ValueError("Unsupported row-tiled expert input fusion")
    width = 2 * K if swiglu else K
    M = row_count(M, dynamic_rows)
    slots = M * 10 if dispatch else M

    @T.macro
    def body(X, Signs, Q, S, Low, High, Sum, IDs, RelativeSlot, RowOffsets, DS, SlotMap, row):
        current = T.alloc_shared((K,), T.float32)
        values = T.alloc_fragment((K,), T.float32)
        quant = T.alloc_fragment((K,), T.int32)
        low = T.alloc_fragment((K,), T.uint32)
        high = T.alloc_fragment((K,), T.uint32)
        total = T.alloc_fragment((1,), T.int32)
        absolute = T.alloc_fragment((K,), T.float32)
        maximum = T.alloc_fragment((1,), T.float32)
        scale = T.alloc_fragment((1,), T.float16)
        inv = T.alloc_fragment((1,), T.float32)
        T.annotate_layout(
            {
                v: tilelang.Fragment(
                    (K,),
                    forward_thread_fn=lambda k: k % 128,
                    forward_index_fn=lambda k: k // 128,
                )
                for v in (values, quant, low, high)
            }
        )
        # Every warp owns 32 consecutive lanes of each block128 transform.
        # K is a multiple of 128, so shuffle calls always have a full mask.
        for k in T.Parallel(K, coalesced_width=T.int32(1)):
            if swiglu:
                gate = T.cast(X[row, k], T.float32)
                e = T.exp(-T.abs(gate))
                sigmoid = T.if_then_else(gate >= 0, 1 / (1 + e), e / (1 + e))
                value = T.cast(
                    T.cast((gate * sigmoid) * T.cast(X[row, k + K], T.float32), T.float16),
                    T.float32,
                )
                values[k] = value * T.cast(Signs[k], T.float32)
            else:
                values[k] = T.cast(X[row, k], T.float32) * T.cast(Signs[k], T.float32)
        for stage in T.unroll(5):
            for k in T.Parallel(K):
                other = T.call_pure_extern(
                    "float32",
                    "__shfl_xor_sync",
                    T.uint32(0xFFFFFFFF),
                    values[k],
                    1 << stage,
                    32,
                )
                values[k] = other + values[k] * (1 - 2 * ((k >> stage) & 1))
        for stage in T.unroll(5, 7):
            # Only the last two butterflies cross warp boundaries. Retain
            # the original FP32 add order and subsequent FP16 rounding.
            T.copy(values, current)
            T.sync_threads()
            for k in T.Parallel(K):
                values[k] = current[k ^ (1 << stage)] + values[k] * (1 - 2 * ((k >> stage) & 1))
            T.sync_threads()
        for k in T.Parallel(K):
            values[k] = T.cast(T.cast(values[k] * 0.08838834764831845, T.float16), T.float32)
            absolute[k] = T.abs(values[k])
        T.reduce_max(absolute, maximum, dim=0)
        scale[0] = T.if_then_else(
            maximum[0] > 0,
            T.max(T.call_extern("float32", "__fdiv_rn", maximum[0], 127.0), 2**-24),
            1.0,
        )
        S[row, 0] = scale[0]
        if reciprocal:
            inv[0] = T.call_extern("float32", "__fdiv_rn", 1.0, T.cast(scale[0], T.float32))
        for k in T.Parallel(K):
            if reciprocal:
                ratio = values[k] * inv[0]
                ratio = T.if_then_else(
                    T.abs(ratio - T.round(ratio)) >= 0.5 - 2**-14,
                    T.call_extern("float32", "__fdiv_rn", values[k], T.cast(scale[0], T.float32)),
                    ratio,
                )
            else:
                ratio = T.call_extern(
                    "float32", "__fdiv_rn", values[k], T.cast(scale[0], T.float32)
                )
            quant[k] = T.cast(T.max(-127.0, T.min(127.0, T.round(ratio))), T.int32)
            Q[row, k] = quant[k]
            if exact:
                position = ((k & 1) << 1) | ((k & 2) >> 1) | (k & 4)
                low[k] = (T.cast(quant[k], T.uint32) & 15) << (position * 4)
                high[k] = ((T.cast(quant[k], T.uint32) >> 4) & 15) << (position * 4)
        if exact:
            for stage in T.unroll(3):
                for k in T.Parallel(K):
                    low[k] |= T.call_pure_extern(
                        "uint32", "__shfl_xor_sync", T.uint32(0xFFFFFFFF), low[k], 1 << stage, 32
                    )
                    high[k] |= T.call_pure_extern(
                        "uint32", "__shfl_xor_sync", T.uint32(0xFFFFFFFF), high[k], 1 << stage, 32
                    )
            T.reduce_sum(quant, total, dim=0)
            if scatter:
                T.sync_threads()
                for choice in T.serial(10):
                    slot = RowOffsets[IDs[row, choice]] + RelativeSlot[row, choice]
                    for k in T.Parallel(K):
                        if k % 8 == 0:
                            Low[slot, k // 8] = low[k]
                            High[slot, k // 8] = high[k]
                    if T.get_thread_binding() == 0:
                        Sum[slot] = total[0]
                        SlotMap[row, choice] = slot
                        DS[slot, 0] = S[row, 0]
            else:
                for k in T.Parallel(K):
                    if k % 8 == 0:
                        if dispatch:
                            Low[0, k // 8] = low[k]
                            High[0, k // 8] = high[k]
                        else:
                            Low[row, k // 8] = low[k]
                            High[row, k // 8] = high[k]
                if dispatch:
                    Sum[0] = total[0]
                else:
                    Sum[row] = total[0]

    if not exact:

        @T.prim_func
        def main(
            X: T.Tensor((M, width), T.float16),
            Signs: T.Tensor((K,), T.int8),
            Q: T.Tensor((M, K), T.int8),
            S: T.Tensor((M, 1), T.float16),
        ):
            with T.Kernel(M, threads=128) as row:
                body(X, Signs, Q, S, Q, Q, S, Q, Q, Q, S, Q, row)

        return main

    if not dispatch:

        @T.prim_func
        def rotate_split_a8(
            X: T.Tensor((M, width), T.float16),
            Signs: T.Tensor((K,), T.int8),
            Q: T.Tensor((M, K), T.int8),
            S: T.Tensor((M, 1), T.float16),
            Low: T.Tensor((slots, K // 8), T.uint32),
            High: T.Tensor((slots, K // 8), T.uint32),
            Sum: T.Tensor((slots,), T.int32),
        ):
            with T.Kernel(M, threads=128) as row:
                body(X, Signs, Q, S, Low, High, Sum, Q, Q, Q, S, Q, row)

        return (
            rotate_split_a8.with_attr("global_symbol", "rows_rotate_split")
            if output_layout == "e8p_rows"
            else rotate_split_a8
        )

    @T.prim_func
    def rotate_dispatch_a8(
        X: T.Tensor((M, width), T.float16),
        Signs: T.Tensor((K,), T.int8),
        Q: T.Tensor((M, K), T.int8),
        S: T.Tensor((M, 1), T.float16),
        Low: T.Tensor((slots, K // 8), T.uint32),
        High: T.Tensor((slots, K // 8), T.uint32),
        Sum: T.Tensor((slots,), T.int32),
        IDs: T.Tensor((M, 10), T.int32),
        RelativeSlot: T.Tensor((M, 10), T.int32),
        RowOffsets: T.Tensor((512,), T.int32),
        DS: T.Tensor((slots, 1), T.float16),
        SlotMap: T.Tensor((M, 10), T.int32),
    ):
        with T.Kernel(M, threads=128) as row:
            body(X, Signs, Q, S, Low, High, Sum, IDs, RelativeSlot, RowOffsets, DS, SlotMap, row)
            if not scatter:
                for rank in T.Parallel(10):
                    slot = RowOffsets[IDs[0, rank]] + RelativeSlot[0, rank]
                    SlotMap[0, rank] = slot
                    DS[slot, 0] = S[0, 0]

    if scatter:
        return rotate_dispatch_a8.with_attr("global_symbol", "rows_rotate_scatter")
    return rotate_dispatch_a8
