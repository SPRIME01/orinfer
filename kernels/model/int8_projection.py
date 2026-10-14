"""Row-scaled dense INT8 Tensor Core projections for Flash Next.

Only the final INT32 accumulator is rescaled. Weights remain stored INT8,
including the output head; no persistent FP16 or expanded expert copy exists.
"""

import tilelang
import tilelang.language as T

from tools.operators.common import orin_jit
from kernels.model.rows import row_count


_DP4A_SOURCE = """
__device__ __forceinline__ int orinfer_dp4a(int a, int b, int c) {
    return __dp4a(a, b, c);
}
"""


@orin_jit
def int8_projection(
    M: int,
    N: int,
    K: int,
    output_dtype: str = "float16",
    block_m: int = 16,
    block_n: int = 64,
    block_k: int = 64,
    group_m: int = 0,
    dynamic_rows: bool = False,
):
    if any(type(v) is not int or v <= 0 for v in (M, N, K)) or K % 64 or K * 128 * 128 > 2**31 - 1:
        raise ValueError("Invalid INT8 projection geometry")
    if output_dtype not in ("float16", "float32"):
        raise ValueError("Invalid INT8 projection output dtype")
    if block_m not in (16, 32, 64, 128):
        raise ValueError("Invalid INT8 row tile")
    if block_n not in (64, 128) or block_k not in (64, 128) or K % block_k:
        raise ValueError("Invalid INT8 column/reduction tile")
    if type(group_m) is not int or group_m not in (0, 1, 2, 4, 8):
        raise ValueError("Invalid INT8 row grouping")
    M = row_count(M, dynamic_rows)
    mt, nt = (M + block_m - 1) // block_m, (N + block_n - 1) // block_n

    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.int8),
        Weight: T.Tensor((N, K), T.int8),
        WeightScale: T.Tensor((N,), T.float16),
        TokenScale: T.Tensor((M,), T.float16),
        Output: T.Tensor((M, N), output_dtype),
    ):
        with T.Kernel(mt * nt, threads=128) as pid:
            by = T.alloc_var(T.int32)
            bx = T.alloc_var(T.int32)
            if group_m:
                first = (pid // (group_m * nt)) * group_m
                actual = T.min(mt - first, group_m)
                by = first + (pid % (group_m * nt)) % actual
                bx = (pid % (group_m * nt)) // actual
            else:
                by = pid % mt
                bx = pid // mt
            a = T.alloc_shared((block_m, block_k), T.int8)
            w = T.alloc_shared((block_n, block_k), T.int8)
            acc = T.alloc_fragment((block_m, block_n), T.int32)
            T.clear(acc)
            for kg in T.Pipelined(K // block_k, num_stages=2):
                T.copy(A[by * block_m, kg * block_k], a)
                T.copy(Weight[bx * block_n, kg * block_k], w)
                T.gemm(a, w, acc, transpose_B=True)
            for i, j in T.Parallel(block_m, block_n):
                if by * block_m + i < M and bx * block_n + j < N:
                    Output[by * block_m + i, bx * block_n + j] = (
                        T.cast(acc[i, j], T.float32)
                        * T.cast(WeightScale[bx * block_n + j], T.float32)
                    ) * T.cast(TokenScale[by * block_m + i], T.float32)

    return main


@orin_jit
def int8_small_projection(
    M: int,
    N: int,
    K: int,
    output_dtype: str = "float16",
    dynamic_rows: bool = False,
    block_m: int | None = None,
    block_n: int = 4,
):
    """Row-tiled DP4A projections; packed views alias the original INT8.

    Each warp reduces one output, reusing its weight words across four token
    rows. INT32 accumulation is exact; rescaling keeps the Tensor Core path's
    FP32 multiply order and final output rounding.
    """
    if any(type(v) is not int or v <= 0 for v in (M, N, K)) or K % 128 or K * 128 * 128 > 2**31 - 1:
        raise ValueError("Invalid small INT8 projection geometry")
    if output_dtype not in ("float16", "float32"):
        raise ValueError("Invalid small INT8 projection output dtype")
    bm, bn = min(M, 4) if block_m is None else block_m, block_n
    if not 1 <= bm <= 8 or bn not in (4, 8, 16):
        raise ValueError("Invalid INT8 DP4A tile")
    M = row_count(M, dynamic_rows)

    @T.prim_func
    def main(
        A: T.Tensor((M, K // 4), T.int32),
        Weight: T.Tensor((N, K // 4), T.int32),
        WeightScale: T.Tensor((N,), T.float16),
        TokenScale: T.Tensor((M,), T.float16),
        Output: T.Tensor((M, N), output_dtype),
    ):
        with T.Kernel(T.ceildiv(M, bm), T.ceildiv(N, bn), threads=128) as (by, bx):
            T.import_source(_DP4A_SOURCE)
            acc = T.alloc_fragment((bm, bn, 32), T.int32)
            total = T.alloc_fragment((bm, bn), T.int32)
            w = T.alloc_fragment((bn, 32), T.int32)
            T.annotate_layout(
                {
                    acc: tilelang.Fragment(
                        (bm, bn, 32),
                        forward_thread_fn=lambda i, n, k: (n % 4) * 32 + k,
                        forward_index_fn=lambda i, n, k: i * (bn // 4) + n // 4,
                    ),
                    w: tilelang.Fragment(
                        (bn, 32),
                        forward_thread_fn=lambda n, k: (n % 4) * 32 + k,
                        forward_index_fn=lambda n, k: n // 4,
                    ),
                }
            )
            T.clear(acc)
            for kg in T.serial(K // 128):
                for n, k in T.Parallel(bn, 32):
                    w[n, k] = 0
                    if bx * bn + n < N:
                        w[n, k] = Weight[bx * bn + n, kg * 32 + k]
                for i, n, k in T.Parallel(bm, bn, 32):
                    if by * bm + i < M:
                        acc[i, n, k] = T.call_pure_extern(
                            "int32",
                            "orinfer_dp4a",
                            A[by * bm + i, kg * 32 + k],
                            w[n, k],
                            acc[i, n, k],
                        )
            T.reduce_sum(acc, total, dim=2)
            for i, n in T.Parallel(bm, bn):
                if by * bm + i < M and bx * bn + n < N:
                    Output[by * bm + i, bx * bn + n] = (
                        T.cast(total[i, n], T.float32) * T.cast(WeightScale[bx * bn + n], T.float32)
                    ) * T.cast(TokenScale[by * bm + i], T.float32)

    return main


@orin_jit
def int8_gemv(N: int, K: int, output_dtype: str = "float16", block_n: int = 16):
    """Single-token signed DP4A; packed views alias the original INT8 buffers."""
    if any(type(v) is not int or v <= 0 for v in (N, K)) or K % 128 or K * 128 * 128 > 2**31 - 1:
        raise ValueError("Invalid INT8 GEMV geometry")
    if output_dtype not in ("float16", "float32") or block_n not in (4, 8, 16, 32):
        raise ValueError("Invalid INT8 GEMV output/tile")

    @T.prim_func
    def main(
        A: T.Tensor((1, K // 4), T.int32),
        Weight: T.Tensor((N, K // 4), T.int32),
        WeightScale: T.Tensor((N,), T.float16),
        TokenScale: T.Tensor((1,), T.float16),
        Output: T.Tensor((1, N), output_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_n), threads=128) as bx:
            T.import_source(_DP4A_SOURCE)
            acc = T.alloc_fragment((block_n, 32), T.int32)
            total = T.alloc_fragment((block_n,), T.int32)
            T.annotate_layout(
                {
                    acc: tilelang.Fragment(
                        (block_n, 32),
                        forward_thread_fn=lambda n, k: (n % 4) * 32 + k,
                        forward_index_fn=lambda n, k: n // 4,
                    )
                }
            )
            T.clear(acc)
            for kg in T.serial(K // 128):
                for n, k in T.Parallel(block_n, 32):
                    if bx * block_n + n < N:
                        acc[n, k] = T.call_pure_extern(
                            "int32",
                            "orinfer_dp4a",
                            A[0, kg * 32 + k],
                            Weight[bx * block_n + n, kg * 32 + k],
                            acc[n, k],
                        )
            T.reduce_sum(acc, total, dim=1)
            for n in T.Parallel(block_n):
                if bx * block_n + n < N:
                    Output[0, bx * block_n + n] = (
                        T.cast(total[n], T.float32)
                        * T.cast(WeightScale[bx * block_n + n], T.float32)
                    ) * T.cast(TokenScale[0], T.float32)

    return main
