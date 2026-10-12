"""HC prefill uses reusable BF16 operands with explicit FP16 rounding.

Norm retains its FP32 inverse RMS. The up/mix epilogue reconstructs the exact
FP16 normalized operand from the rounded residual and stored inverse RMS;
this avoids materializing both FP16 and BF16 copies of the full activation.
BF16 projection inputs preserve the same Tensor Core accumulation traversal.
"""

import tilelang
import tilelang.language as T
from tools.operators.common import orin_jit
from kernels.model.hyperconnection import _dtype
from kernels.model.rows import row_count
from kernels.model.half_math import SOURCE as HALF_MATH, stream_sigmoid


@orin_jit
def hc_input_bf16(M: int, K: int, dynamic_rows: bool = False):
    """Materialize the original FP16-to-BF16 MMA operand once per matrix."""
    if any(type(n) is not int or n <= 0 for n in (M, K)):
        raise ValueError("Invalid HC operand geometry")
    M = row_count(M, dynamic_rows)

    @T.prim_func
    def hc_input_cast(
        Input: T.Tensor((M, K), T.float16),
        Output: T.Tensor((M, K), T.bfloat16),
    ):
        with T.Kernel(T.ceildiv(M, 4), T.ceildiv(K, 256), threads=128) as (bm, bk):
            for i, j in T.Parallel(4, 256):
                if bm * 4 + i < M and bk * 256 + j < K:
                    Output[bm * 4 + i, bk * 256 + j] = Input[bm * 4 + i, bk * 256 + j]

    return hc_input_cast


@orin_jit
def hc_norm_prefill(
    M: int,
    H: int,
    streams: int = 4,
    eps: float = 1e-6,
    dtype: str = "float16",
    dynamic_rows: bool = False,
):
    """Build (Residual[M,streams,H], Weight[streams,H], Normed), dtype/F32/dtype."""
    _dtype(dtype)
    if dtype != "float16":
        raise ValueError("HC prefill retains the original FP16 normalization boundary")
    if any(type(x) is not int or x <= 0 for x in (M, H, streams)) or not 0 < eps < 1:
        raise ValueError("Invalid HC normalization dimensions/epsilon")
    width = 1 << (H - 1).bit_length()
    M = row_count(M, dynamic_rows)

    @T.prim_func
    def hc_prefill_norm(
        Residual: T.Tensor((M, streams, H), dtype),
        Weight: T.Tensor((streams, H), T.float32),
        NormedBF: T.Tensor((M, streams, H), T.bfloat16),
        InvRms: T.Tensor((M, streams), T.float32),
    ):
        with T.Kernel(M, streams, threads=min(256, width)) as (row, branch):
            values = T.alloc_fragment((width,), T.float32)
            square = T.alloc_fragment((width,), T.float32)
            total = T.alloc_fragment((1,), T.float32)
            for j in T.Parallel(width, coalesced_width=T.int32(1)):
                values[j] = T.if_then_else(j < H, T.cast(Residual[row, branch, j], T.float32), 0.0)
                square[j] = values[j] * values[j]
            T.reduce_sum(square, total, dim=0)
            inverse = T.rsqrt(total[0] / H + eps)
            InvRms[row, branch] = inverse
            for j in T.Parallel(H, coalesced_width=T.int32(1)):
                rounded_norm = T.cast((values[j] * inverse * (1.0 + Weight[branch, j])), dtype)
                NormedBF[row, branch, j] = T.cast(rounded_norm, T.bfloat16)

    return hc_prefill_norm


@orin_jit
def hc_combine_norm_prefill(
    M: int, H: int, streams: int = 4, eps: float = 1e-6, dtype: str = "float16"
):
    """Combine residuals and normalize, retaining the rounded residual output.

    Output may alias Residual. Normed must remain disjoint from all inputs and
    Output. Each CTA owns one whole row/stream, including the FP32 reduction.
    The intervening stream-dtype rounding matches hc_combine then hc_norm.
    """
    _dtype(dtype)
    if dtype != "float16":
        raise ValueError("HC prefill retains the original FP16 normalization boundary")
    if any(type(x) is not int or x <= 0 for x in (M, H, streams)) or not 0 < eps < 1:
        raise ValueError("Invalid HC combination/normalization dimensions")
    width = 1 << (H - 1).bit_length()

    @T.prim_func
    def hc_prefill_combine_norm(
        Block: T.Tensor((M, H), dtype),
        Residual: T.Tensor((M, streams, H), dtype),
        Inject: T.Tensor((M, streams), dtype),
        Weight: T.Tensor((streams, H), T.float32),
        Output: T.Tensor((M, streams, H), dtype),
        NormedBF: T.Tensor((M, streams, H), T.bfloat16),
        InvRms: T.Tensor((M, streams), T.float32),
    ):
        with T.Kernel(M, streams, threads=min(256, width)) as (row, branch):
            T.import_source(HALF_MATH)
            values = T.alloc_fragment((width,), T.float32)
            square = T.alloc_fragment((width,), T.float32)
            total = T.alloc_fragment((1,), T.float32)
            raw = T.cast(T.cast(T.cast(Inject[row, branch], T.float32) / streams, dtype), T.float32)
            gate = T.cast(2.0 * T.cast(stream_sigmoid(raw, dtype), T.float32), dtype)
            for j in T.Parallel(width, coalesced_width=T.int32(1)):
                values[j] = 0.0
                if j < H:
                    injection = T.cast(
                        T.cast(Block[row, j], T.float32) * T.cast(gate, T.float32), dtype
                    )
                    rounded = T.cast(
                        T.cast(Residual[row, branch, j], T.float32) + T.cast(injection, T.float32),
                        dtype,
                    )
                    values[j] = T.cast(rounded, T.float32)
                    Output[row, branch, j] = rounded
                square[j] = values[j] * values[j]
            T.reduce_sum(square, total, dim=0)
            inverse = T.rsqrt(total[0] / H + eps)
            InvRms[row, branch] = inverse
            for j in T.Parallel(H, coalesced_width=T.int32(1)):
                rounded_norm = T.cast((values[j] * inverse * (1.0 + Weight[branch, j])), dtype)
                NormedBF[row, branch, j] = T.cast(rounded_norm, T.bfloat16)

    return hc_prefill_combine_norm


@orin_jit
def hc_projection_prefill(
    M: int,
    N: int,
    K: int,
    block_m: int = 16,
    dtype: str = "float16",
    block_n: int = 64,
    silu: bool = False,
    streams: int = 4,
    group_m: int = 0,
    dynamic_rows: bool = False,
    stages: int = 2,
    output_dtype: str = "float16",
):
    """Build (A[M,K], W[N,K], C[M,N]), stream dtype/BF16/stream dtype.

    FP32 accumulation, BF16 Tensor Core arithmetic, explicit output rounding.
    This preserves the original BF16 HC weights. N tails are supported for
    the four-stream injection projection; K must be a multiple of 64.
    """
    _dtype(dtype)
    _dtype(output_dtype)
    if dtype != "float16" or (output_dtype == "bfloat16" and not silu):
        raise ValueError("HC prefill keeps FP16 rounding before BF16 projection operands")
    if (
        any(type(x) is not int or x <= 0 for x in (M, N, K))
        or K % 64
        or block_m not in (16, 32, 64, 128)
    ):
        raise ValueError("Invalid HC projection dimensions")
    if block_n not in (16, 32, 64, 128):
        raise ValueError("Invalid HC output tile")
    if type(group_m) is not int or group_m not in (0, 1, 2, 4, 8):
        raise ValueError("Invalid HC row grouping")
    threads = 64 if block_n == 16 else 128
    M = row_count(M, dynamic_rows)

    @T.macro
    def project(A, W, C, by, bx):
        a = T.alloc_shared((block_m, 64), T.bfloat16)
        w = T.alloc_shared((block_n, 64), T.bfloat16)
        acc = T.alloc_fragment((block_m, block_n), T.float32)
        T.clear(acc)
        for kg in T.Pipelined(K // 64, num_stages=stages):
            T.copy(A[by * block_m, kg * 64], a)
            T.copy(W[bx * block_n, kg * 64], w)
            T.gemm(a, w, acc, transpose_B=True)
        if silu:
            for i, j in T.Parallel(block_m, block_n, coalesced_width=T.int32(1)):
                if by * block_m + i < M and bx * block_n + j < N:
                    rounded = T.cast(acc[i, j], dtype)
                    v = T.cast(T.cast(T.cast(rounded, T.float32) / streams, dtype), T.float32)
                    C[by * block_m + i, bx * block_n + j] = T.cast(v / (1 + T.exp(-v)), dtype)
        else:
            T.copy(acc, C[by * block_m, bx * block_n])

    if group_m:
        mt, nt = (M + block_m - 1) // block_m, (N + block_n - 1) // block_n

        @T.prim_func
        def hc_prefill_projection_grouped(
            A: T.Tensor((M, K), T.bfloat16),
            W: T.Tensor((N, K), T.bfloat16),
            C: T.Tensor((M, N), output_dtype),
        ):
            with T.Kernel(mt * nt, threads=threads) as pid:
                first = (pid // (group_m * nt)) * group_m
                actual = T.min(mt - first, group_m)
                by = first + (pid % (group_m * nt)) % actual
                bx = (pid % (group_m * nt)) // actual
                project(A, W, C, by, bx)

        return hc_prefill_projection_grouped
    else:

        @T.prim_func
        def hc_prefill_projection(
            A: T.Tensor((M, K), T.bfloat16),
            W: T.Tensor((N, K), T.bfloat16),
            C: T.Tensor((M, N), output_dtype),
        ):
            with T.Kernel(T.ceildiv(M, block_m), T.ceildiv(N, block_n), threads=threads) as (
                by,
                bx,
            ):
                project(A, W, C, by, bx)

        return hc_prefill_projection


@orin_jit
def hc_up_mix_prefill(M, H, rank, block_m=128, block_n=32, stages=2):
    if (
        any(type(x) is not int or x <= 0 for x in (M, H, rank, block_m, block_n, stages))
        or block_m not in (16, 32, 64, 128)
        or block_n != 32
        or H % block_n
        or rank % 64
        or stages > 8
    ):
        raise ValueError("Invalid HC prefill up/mix dimensions or pipeline")
    BM, BN = block_m, block_n

    @T.prim_func
    def hc_prefill_up_mix(
        A: T.Tensor((M, rank), T.bfloat16),
        W: T.Tensor((4 * H, rank), T.bfloat16),
        Residual: T.Tensor((M, 4, H), T.float16),
        Weight: T.Tensor((4, H), T.float32),
        InvRms: T.Tensor((M, 4), T.float32),
        Out: T.Tensor((M, H), T.float16),
    ):
        with T.Kernel(T.ceildiv(M, BM), H // BN, threads=128) as (by, bx):
            T.import_source(HALF_MATH)
            a = T.alloc_shared((BM, 64), T.bfloat16)
            w = T.alloc_shared((4 * BN, 64), T.bfloat16)
            acc = T.alloc_fragment((BM, 4 * BN), T.float32)
            rounded = T.alloc_shared((BM, 4 * BN), T.float16)
            inv = T.alloc_shared((BM, 4), T.float32)
            total = T.alloc_fragment((BM, BN), T.float32)
            T.annotate_layout(
                {
                    total: tilelang.Fragment(
                        (BM, BN),
                        forward_thread_fn=lambda i, j: (i % 4) * 32 + j,
                        forward_index_fn=lambda i, j: i // 4,
                    )
                }
            )
            T.clear(acc)
            for kg in T.Pipelined(rank // 64, num_stages=stages):
                T.copy(A[by * BM, kg * 64], a)
                for n, j in T.Parallel(4 * BN, 64):
                    w[n, j] = W[(n // BN) * H + bx * BN + n % BN, kg * 64 + j]
                T.gemm(a, w, acc, transpose_B=True)
            T.copy(acc, rounded)
            T.copy(InvRms[by * BM, 0], inv)
            T.clear(total)
            for branch in T.serial(4):
                for i, j in T.Parallel(BM, BN, coalesced_width=T.int32(1)):
                    if by * BM + i < M:
                        normed = T.cast(
                            T.cast(Residual[by * BM + i, branch, bx * BN + j], T.float32)
                            * inv[i, branch]
                            * (1.0 + Weight[branch, bx * BN + j]),
                            T.float16,
                        )
                        gate = stream_sigmoid(
                            T.cast(rounded[i, branch * BN + j], T.float32), "float16"
                        )
                        product = T.cast(
                            T.cast(gate, T.float32) * T.cast(normed, T.float32), T.float16
                        )
                        total[i, j] += T.cast(product, T.float32)
            for i, j in T.Parallel(BM, BN):
                if by * BM + i < M:
                    Out[by * BM + i, bx * BN + j] = total[i, j] / 4

    return hc_prefill_up_mix
