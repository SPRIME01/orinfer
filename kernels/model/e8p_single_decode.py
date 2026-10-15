"""Exact single-token E8P decode using row-cached A8 and direct MMA operands.

Each selected expert has one row, guaranteed by the registered decode router.
Only one compressed expert bank is read; four private stages prefetch weights.
Low/High uint32 views alias the unchanged packed A8 workspace.
"""

import tilelang.language as T
from tools.operators.common import orin_jit
from kernels.model.e8p_core import SOURCE, mma_pair, store_rows


@orin_jit(
    pass_configs={
        "tl.disable_thread_storage_sync": True,
        "tl.disable_shared_memory_reuse": True,
        "tl.if_stmt_binding_inline_replayable_binds": False,
    }
)
def e8p_single_decode_grouped(rows, E, tiles, N, K, *, shared_input=False):
    if (
        any(type(x) is not int or x <= 0 for x in (rows, E, tiles, N, K))
        or K % 128
        or K * 128 * 128 > 2**31 - 1
        or rows > E
        or tiles < rows
    ):
        raise ValueError("Invalid exact E8P geometry")
    if type(shared_input) is not bool:
        raise ValueError("Invalid unique-route shared-input contract")
    input_rows = 1 if shared_input else rows
    BM, BN, BK = 8, 128, 64
    # NCU identifies excess shared wavefronts for strided stores. Swizzle
    # sixteen-byte row halves so ldmatrix and writes both use distinct banks.
    # The small decode tile benefits from weight swizzling only.
    input_stride, input_mask = (32, 16) if BM == 16 else (48, 0)

    multiply, store = mma_pair(BM), store_rows(BM, BN, N)

    @T.prim_func
    def exact_e8p_int4(
        Low: T.Tensor((input_rows, K // 8), T.uint32),
        High: T.Tensor((input_rows, K // 8), T.uint32),
        Sum: T.Tensor((input_rows,), T.int32),
        P: T.Tensor((E, K // 64, N, 8), T.uint16),
        Book: T.Tensor((1, 256, 1), T.uint32),
        WS: T.Tensor((E, N), T.float16),
        AS: T.Tensor((input_rows,), T.float16),
        Counts: T.Tensor((E,), T.int32),
        Offsets: T.Tensor((E,), T.int32),
        TE: T.Tensor((tiles,), T.int32),
        TR: T.Tensor((tiles,), T.int32),
        TC: T.Tensor((1,), T.int32),
        Out: T.Tensor((rows, N), T.float16),
    ):
        with T.Kernel(T.ceildiv(N, BN), tiles, threads=128) as (bx, tile):
            T.import_source(SOURCE)
            tid = T.get_thread_binding()
            lane = tid % 32
            warp = tid // 32
            packed = T.alloc_shared((4, BN, 8), T.uint16)
            input_lo = T.alloc_shared((K // 8,), T.uint32)
            input_hi = T.alloc_shared((K // 8,), T.uint32)
            book = T.alloc_shared((256,), T.uint32)
            lo = T.alloc_local((BM,), T.int32)
            hi = T.alloc_local((BM,), T.int32)
            v = T.alloc_local((4,), T.uint32)
            a = T.alloc_local((BM // 2,), T.uint32)
            if tile < TC[0]:
                expert, first = TE[tile], TR[tile]
                offset = Offsets[expert] + first
                input_offset = 0 if shared_input else offset
                count = T.min(1, Counts[expert] - first)
                for i in T.unroll(2):
                    book[tid + i * 128] = Book[0, tid + i * 128, 0]
                for i in T.unroll(BM):
                    lo[i] = 0
                    hi[i] = 0
                T.copy(Low[input_offset, 0], input_lo)
                T.copy(High[input_offset, 0], input_hi)
                T.sync_threads()
                for initial in T.unroll(min(3, K // BK)):
                    T.ptx_cp_async(
                        T.access_ptr(packed[initial, tid, 0], "w", 8),
                        T.access_ptr(P[expert, initial, T.min(bx * BN + tid, N - 1), 0], "r", 8),
                        8,
                        predicate=bx * BN + tid < N,
                    )
                    T.ptx_commit_group()
                for step in T.serial(K // BK):
                    slot = step % 4
                    if step + 2 < K // BK:
                        T.ptx_wait_group(2)
                    elif step + 1 < K // BK:
                        T.ptx_wait_group(1)
                    else:
                        T.ptx_wait_group(0)
                    T.sync_threads()
                    if step + 3 < K // BK:
                        nxt = (step + 3) % 4
                        T.ptx_cp_async(
                            T.access_ptr(packed[nxt, tid, 0], "w", 8),
                            T.access_ptr(
                                P[expert, step + 3, T.min(bx * BN + tid, N - 1), 0], "r", 8
                            ),
                            8,
                            predicate=bx * BN + tid < N,
                        )
                        T.ptx_commit_group()
                    col = lane % 4
                    a[0] = input_lo[step * 8 + col]
                    a[1] = input_lo[step * 8 + col + 4]
                    a[2] = input_hi[step * 8 + col]
                    a[3] = input_hi[step * 8 + col + 4]
                    for nt in T.unroll(2):
                        for rr in T.unroll(4):
                            n = warp * 32 + nt * 16 + lane // 4 + (rr % 2) * 8
                            col = lane % 4 + (rr // 2) * 4
                            code = T.cast(packed[slot, n, col], T.uint32)
                            v[rr] = T.call_pure_extern("uint32", "nib_fast", code, book[code >> 8])
                        for stripe in T.unroll(BM // 8):
                            multiply(lo, hi, v, a, nt, stripe)
                    T.sync_threads()
                store(lo, hi, Sum, WS, AS, Out, bx, warp, lane, count, expert, offset, input_offset)

    return (
        exact_e8p_int4.with_attr("global_symbol", "exact_e8p_shared_input")
        if shared_input
        else exact_e8p_int4
    )
