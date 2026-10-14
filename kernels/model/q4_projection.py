"""Row-scaled scalar Q4 indices decoded to INT8 for full-K accumulation.

Each nibble stores three magnitude bits and one sign bit. A fixed symmetric
16-point integer codebook allows two byte-permute instructions to unpack four
weights without group scale operations inside the dot-product loop.
"""

import tilelang
import tilelang.language as T
from tools.operators.common import orin_jit
from kernels.model.rows import row_count


MAGNITUDES = (3, 10, 19, 31, 46, 67, 93, 127)


def decode_source(magnitudes=MAGNITUDES):
    if len(magnitudes) != 8 or any(type(v) is not int or not 0 < v <= 127 for v in magnitudes):
        raise ValueError("Q4 needs eight positive INT8 magnitudes")
    if tuple(sorted(set(magnitudes))) != tuple(magnitudes):
        raise ValueError("Q4 magnitudes must be strictly increasing")
    lo = sum(v << (8 * i) for i, v in enumerate(magnitudes[:4]))
    hi = sum(v << (8 * i) for i, v in enumerate(magnitudes[4:]))
    return f"""
__device__ __forceinline__ unsigned q4_decode(unsigned p) {{
    unsigned v = __byte_perm({lo}u, {hi}u, p & 0x7777u);
    unsigned signs = __byte_perm(0u, 0xffffffffu, (p >> 1) & 0x4444u);
    return (v ^ signs) + (signs & 0x01010101u);
}}
__device__ __forceinline__ int q4_mad(int a, unsigned w, int c) {{
    return __dp4a(a, (int)w, c);
}}
"""


def validate_shape(M, N, K):
    if min(M, N, K) <= 0 or K % 128 or K * 128 * 127 > 2**31 - 1:
        raise ValueError("Q4 projection requires block128 geometry and bounded INT32 accumulation")


@orin_jit
def q4_gemv(M, N, K, BN=16, output_dtype="float16", magnitudes=MAGNITUDES, *, dynamic_rows=False):
    validate_shape(M, N, K)
    if not 1 <= M <= 8 or BN not in (4, 8, 16):
        raise ValueError("Q4 GEMV supports one to eight rows and BN=4/8/16")
    BM = M
    M = row_count(M, dynamic_rows)
    source = decode_source(magnitudes)

    @T.prim_func
    def main(
        A: T.Tensor((M, K // 4), T.int32),
        W: T.Tensor((N, K // 4), T.uint16),
        S: T.Tensor((N,), T.float16),
        AS: T.Tensor((M,), T.float16),
        Out: T.Tensor((M, N), output_dtype),
    ):
        with T.Kernel(T.ceildiv(N, BN), threads=128) as bx:
            T.import_source(source)
            acc = T.alloc_fragment((BM, BN, 32), T.int32)
            total = T.alloc_fragment((BM, BN), T.int32)
            weight = T.alloc_fragment((BN, 32), T.uint32)
            T.annotate_layout(
                {
                    acc: tilelang.Fragment(
                        (BM, BN, 32),
                        forward_thread_fn=lambda i, n, l: (n % 4) * 32 + l,
                        forward_index_fn=lambda i, n, l: i * (BN // 4) + n // 4,
                    ),
                    weight: tilelang.Fragment(
                        (BN, 32),
                        forward_thread_fn=lambda n, l: (n % 4) * 32 + l,
                        forward_index_fn=lambda n, l: n // 4,
                    ),
                }
            )
            T.clear(acc)
            for group in T.serial(K // 128):
                for n, l in T.Parallel(BN, 32):
                    weight[n, l] = 0
                    if bx * BN + n < N:
                        weight[n, l] = T.call_pure_extern(
                            "uint32", "q4_decode", T.cast(W[bx * BN + n, group * 32 + l], T.uint32)
                        )
                for i, n, l in T.Parallel(BM, BN, 32):
                    if i < M:
                        acc[i, n, l] = T.call_pure_extern(
                            "int32", "q4_mad", A[i, group * 32 + l], weight[n, l], acc[i, n, l]
                        )
            T.reduce_sum(acc, total, dim=2)
            for i, n in T.Parallel(BM, BN):
                if i < M and bx * BN + n < N:
                    Out[i, bx * BN + n] = (
                        T.cast(total[i, n], T.float32) * T.cast(S[bx * BN + n], T.float32)
                    ) * T.cast(AS[i], T.float32)

    return main


@orin_jit
def q4_projection(
    M,
    N,
    K,
    BN=64,
    BK=128,
    output_dtype="float16",
    magnitudes=MAGNITUDES,
    *,
    block_m=8,
    dynamic_rows=False,
    stages=2,
):
    validate_shape(M, N, K)
    if BN not in (32, 64, 128) or BK % 128 or K % BK:
        raise ValueError("Invalid Q4 Tensor Core tile")
    if block_m not in (8, 16, 32, 64, 128):
        raise ValueError("Invalid Q4 row tile")
    if type(stages) is not int or stages not in (1, 2):
        raise ValueError("Invalid Q4 pipeline depth")
    BM = block_m
    M = row_count(M, dynamic_rows)
    threads = 64 if BN == 32 else 128
    source = decode_source(magnitudes)

    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.int8),
        W: T.Tensor((N, K // 4), T.uint16),
        S: T.Tensor((N,), T.float16),
        AS: T.Tensor((M,), T.float16),
        Out: T.Tensor((M, N), output_dtype),
    ):
        with T.Kernel(T.ceildiv(M, BM), T.ceildiv(N, BN), threads=threads) as (by, bx):
            T.import_source(source)
            activation = T.alloc_shared((BM, BK), T.int8)
            weight = T.alloc_shared((BN, BK), T.int8)
            decoded = T.alloc_fragment((BN, BK // 4), T.uint32)
            acc = T.alloc_fragment((BN, BM), T.int32)
            T.clear(acc)
            for group in T.Pipelined(K // BK, num_stages=stages):
                T.copy(A[by * BM, group * BK], activation)
                for n, j in T.Parallel(BN, BK // 4):
                    decoded[n, j] = 0
                    if bx * BN + n < N:
                        decoded[n, j] = T.call_pure_extern(
                            "uint32",
                            "q4_decode",
                            T.cast(W[bx * BN + n, group * (BK // 4) + j], T.uint32),
                        )
                for n, j in T.Parallel(BN, BK):
                    weight[n, j] = T.cast((decoded[n, j // 4] >> ((j % 4) * 8)) & 255, T.int8)
                T.gemm(weight, activation, acc, transpose_B=True)
            for i, n in T.Parallel(BM, BN):
                if by * BM + i < M and bx * BN + n < N:
                    Out[by * BM + i, bx * BN + n] = (
                        T.cast(acc[n, i], T.float32) * T.cast(S[bx * BN + n], T.float32)
                    ) * T.cast(AS[by * BM + i], T.float32)

    return main


@orin_jit
def unpack_q4(N: int, K: int, magnitudes=MAGNITUDES):
    """Expand one projection into caller-owned, temporary INT8 storage.

    Packed INT32 stores preserve the exact signed integer codebook. Scales and
    rotations stay with the consuming projection; no new quantization occurs.
    """
    validate_shape(1, N, K)
    source = decode_source(magnitudes)
    words = N * (K // 4)

    @T.prim_func
    def main(
        W: T.Tensor((N, K // 4), T.uint16),
        Output: T.Tensor((N, K // 4), T.int32),
    ):
        with T.Kernel(T.ceildiv(words, 256), threads=128) as block:
            T.import_source(source)
            for offset in T.Parallel(256):
                word = block * 256 + offset
                if word < words:
                    n, k = word // (K // 4), word % (K // 4)
                    Output[n, k] = T.cast(
                        T.call_pure_extern("uint32", "q4_decode", T.cast(W[n, k], T.uint32)),
                        T.int32,
                    )

    return main
