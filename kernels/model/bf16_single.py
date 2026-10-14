"""Exact single-token BF16 projection with cached inputs and direct MMA operands.

Grouped columns remain in the original row-major weights; no read cache or
expanded representation is created. FP32 accumulates in the original K16 order.
"""

import tilelang.language as T
from tools.operators.common import orin_jit
from kernels.model.bf16_mma import SOURCE
from kernels.model.rows import row_count


@orin_jit(
    pass_configs={"tl.disable_thread_storage_sync": True, "tl.disable_shared_memory_reuse": True}
)
def bf16_single(
    N,
    K,
    groups=1,
    block_n=32,
    block_k=256,
    stages=3,
    *,
    rows=1,
    dynamic_rows=False,
    row_major_grid=False,
):
    if (
        any(type(x) is not int or x <= 0 for x in (N, K, groups, block_n, block_k, stages))
        or groups not in (1, 4)
        or block_n not in (16, 32)
        or block_k not in (128, 256)
        or stages not in (2, 3, 4)
        or N % block_n
        or K % block_k
    ):
        raise ValueError("Unsupported single-token BF16 projection geometry")
    bn, bk = block_n, block_k
    threads = bn * 2
    if type(rows) is not int or not 1 <= rows <= 128:
        raise ValueError("Unsupported BF16 decode rows")
    if type(row_major_grid) is not bool:
        raise ValueError("Invalid request tile ordering")
    rows = row_count(rows, dynamic_rows)

    @T.prim_func
    def main(
        A: T.Tensor((rows, groups, K), T.float16),
        Weight: T.Tensor((N, groups * K // 2), T.uint32),
        Output: T.Tensor((groups, rows, N), T.float32),
    ):
        with T.Kernel(
            (N // bn) * rows if row_major_grid else N // bn,
            groups,
            1 if row_major_grid else rows,
            threads=threads,
        ) as (block, group, lane):
            bx = block // rows if row_major_grid else block
            row = block % rows if row_major_grid else lane
            T.import_source(SOURCE)
            tid = T.get_thread_binding()
            lane = tid % 32
            warp = tid // 32
            live = T.alloc_shared((K,), T.bfloat16)
            w = T.alloc_shared((stages, bn, bk // 2), T.uint32)
            acc = T.alloc_local((4,), T.float32)
            v = T.alloc_local((4,), T.uint32)
            T.clear(acc)
            for initial in T.unroll(min(stages - 1, K // bk)):
                for it in T.unroll(bn * bk // (threads * 8)):
                    index = (tid + it * threads) * 4
                    n = index // (bk // 2)
                    col = index % (bk // 2)
                    T.ptx_cp_async(
                        T.access_ptr(w[initial, n, col ^ ((n % 8) * 4)], "w", 4),
                        T.access_ptr(
                            Weight[bx * bn + n, group * (K // 2) + initial * (bk // 2) + col],
                            "r",
                            4,
                        ),
                        4,
                    )
                T.ptx_commit_group()
            for j in T.Parallel(K):
                live[j] = A[row, group, j]
            for step in T.serial(K // bk):
                slot = step % stages
                if stages == 4:
                    if step + 2 < K // bk:
                        T.ptx_wait_group(2)
                    elif step + 1 < K // bk:
                        T.ptx_wait_group(1)
                    else:
                        T.ptx_wait_group(0)
                elif stages == 3:
                    if step + 1 < K // bk:
                        T.ptx_wait_group(1)
                    else:
                        T.ptx_wait_group(0)
                else:
                    T.ptx_wait_group(0)
                T.sync_threads()
                if step + stages - 1 < K // bk:
                    nxt = (step + stages - 1) % stages
                    for it in T.unroll(bn * bk // (threads * 8)):
                        index = (tid + it * threads) * 4
                        n = index // (bk // 2)
                        col = index % (bk // 2)
                        T.ptx_cp_async(
                            T.access_ptr(w[nxt, n, col ^ ((n % 8) * 4)], "w", 4),
                            T.access_ptr(
                                Weight[
                                    bx * bn + n,
                                    group * (K // 2) + (step + stages - 1) * (bk // 2) + col,
                                ],
                                "r",
                                4,
                            ),
                            4,
                        )
                    T.ptx_commit_group()
                for kk in T.unroll(bk // 16):
                    b0 = T.call_pure_extern(
                        "uint32",
                        "hc_raw_pair",
                        T.access_ptr(live[step * bk + kk * 16 + (lane % 4) * 2], "r", 2),
                    )
                    b1 = T.call_pure_extern(
                        "uint32",
                        "hc_raw_pair",
                        T.access_ptr(live[step * bk + kk * 16 + (lane % 4) * 2 + 8], "r", 2),
                    )
                    for rr in T.unroll(4):
                        n = warp * 16 + lane // 4 + (rr % 2) * 8
                        col = kk * 8 + lane % 4 + (rr // 2) * 4
                        v[rr] = w[slot, n, col ^ ((n % 8) * 4)]
                    T.evaluate(
                        T.call_extern(
                            "void",
                            "hc_mma_bf16",
                            T.access_ptr(acc[0], "rw", 4),
                            v[0],
                            v[1],
                            v[2],
                            v[3],
                            b0,
                            b1,
                        )
                    )
                T.sync_threads()
            if lane % 4 == 0:
                for rr in T.unroll(2):
                    Output[group, row, bx * bn + warp * 16 + lane // 4 + rr * 8] = acc[rr * 2]

    return main.with_attr("global_symbol", "bf16_single")
