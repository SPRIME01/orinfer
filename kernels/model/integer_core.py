"""Compile-time integer projection components shared across codecs and fusions.

Only weight loading depends on the codec. Scheduling owns accumulation and
tails; the epilogue owns scale order and the FP16 gate/up rounding boundary.
Macros inline into the caller, preserving its ABI and runtime row parameter.
"""

import tilelang
import tilelang.language as T


DP4A_SOURCE = """
__device__ __forceinline__ int orinfer_dp4a(int a, int b, int c) {
    return __dp4a(a, b, c);
}
"""


def rescale(value, weight_scale, token_scale):
    return (T.cast(value, T.float32) * T.cast(weight_scale, T.float32)) * T.cast(
        token_scale, T.float32
    )


def swiglu(gate, up):
    g = T.cast(T.cast(gate, T.float16), T.float32)
    u = T.cast(T.cast(up, T.float16), T.float32)
    return g / (1.0 + T.exp(-g)) * u


def weight_loader(codec, n, bn, bk, source):
    if codec not in ("w8", "q4"):
        raise ValueError("Unsupported integer weight codec")

    @T.macro
    def load(W, tile, bx, kg):
        if codec == "w8":
            T.copy(W[bx * bn, kg * bk], tile)
        else:
            T.import_source(source)
            decoded = T.alloc_fragment((bn, bk // 4), T.uint32)
            for j, k in T.Parallel(bn, bk // 4):
                decoded[j, k] = 0
                if bx * bn + j < n:
                    decoded[j, k] = T.call_pure_extern(
                        "uint32", "q4_decode", T.cast(W[bx * bn + j, kg * (bk // 4) + k], T.uint32)
                    )
            for j, k in T.Parallel(bn, bk):
                tile[j, k] = T.cast((decoded[j, k // 4] >> ((k % 4) * 8)) & 255, T.int8)

    return load


def tensor_core(m, n, k, bm, bn, bk, *, stages=2, codec="w8", source="", dual=False):
    load = weight_loader(codec, n, bn, bk, source)
    # Keep Q4's weight-left MMA orientation and W8's activation-left layout.
    left = codec == "q4"

    @T.macro
    def project(A, W, U, WS, US, AS, Out, by, bx):
        a = T.alloc_shared((bm, bk), T.int8)
        w = T.alloc_shared((bn, bk), T.int8)
        acc = T.alloc_fragment((bn, bm) if left else (bm, bn), T.int32)
        if dual:
            u = T.alloc_shared((bn, bk), T.int8)
            up = T.alloc_fragment((bn, bm) if left else (bm, bn), T.int32)
            T.clear(up)
        T.clear(acc)
        for kg in T.Pipelined(k // bk, num_stages=stages):
            T.copy(A[by * bm, kg * bk], a)
            load(W, w, bx, kg)
            if dual:
                load(U, u, bx, kg)
            if left:
                T.gemm(w, a, acc, transpose_B=True)
            else:
                T.gemm(a, w, acc, transpose_B=True)
            if dual:
                if left:
                    T.gemm(u, a, up, transpose_B=True)
                else:
                    T.gemm(a, u, up, transpose_B=True)
        for i, j in T.Parallel(bm, bn):
            if by * bm + i < m and bx * bn + j < n:
                value = rescale(acc[j, i] if left else acc[i, j], WS[bx * bn + j], AS[by * bm + i])
                if dual:
                    other = rescale(
                        up[j, i] if left else up[i, j], US[bx * bn + j], AS[by * bm + i]
                    )
                    Out[by * bm + i, bx * bn + j] = swiglu(value, other)
                else:
                    Out[by * bm + i, bx * bn + j] = value

    return project


def dp4a(m, n, k, bm, bn, *, codec="w8", source="", dual=False):
    if codec not in ("w8", "q4"):
        raise ValueError("Unsupported integer weight codec")
    if dual and codec != "w8":
        raise ValueError("The dual DP4A ABI requires two W8 operands")

    @T.macro
    def project(A, W, U, WS, US, AS, Out, by, bx):
        T.import_source(DP4A_SOURCE + source)
        acc = T.alloc_fragment((bm, bn, 32), T.int32)
        total = T.alloc_fragment((bm, bn), T.int32)
        if dual:
            up = T.alloc_fragment((bm, bn, 32), T.int32)
            up_total = T.alloc_fragment((bm, bn), T.int32)
            T.annotate_layout(
                {
                    up: tilelang.Fragment(
                        (bm, bn, 32),
                        forward_thread_fn=lambda i, j, l: (j % 4) * 32 + l,
                        forward_index_fn=lambda i, j, l: i * (bn // 4) + j // 4,
                    )
                }
            )
            T.clear(up)
        else:
            weight = T.alloc_fragment((bn, 32), T.int32 if codec == "w8" else T.uint32)
            T.annotate_layout(
                {
                    weight: tilelang.Fragment(
                        (bn, 32),
                        forward_thread_fn=lambda j, l: (j % 4) * 32 + l,
                        forward_index_fn=lambda j, l: j // 4,
                    )
                }
            )
        T.annotate_layout(
            {
                acc: tilelang.Fragment(
                    (bm, bn, 32),
                    forward_thread_fn=lambda i, j, l: (j % 4) * 32 + l,
                    forward_index_fn=lambda i, j, l: i * (bn // 4) + j // 4,
                )
            }
        )
        T.clear(acc)
        for kg in T.serial(k // 128):
            if not dual:
                for j, l in T.Parallel(bn, 32):
                    weight[j, l] = 0
                    if bx * bn + j < n:
                        if codec == "w8":
                            weight[j, l] = W[bx * bn + j, kg * 32 + l]
                        else:
                            weight[j, l] = T.call_pure_extern(
                                "uint32", "q4_decode", T.cast(W[bx * bn + j, kg * 32 + l], T.uint32)
                            )
            for i, j, l in T.Parallel(bm, bn, 32):
                if by * bm + i < m:
                    if dual:
                        if bx * bn + j < n:
                            acc[i, j, l] = T.call_pure_extern(
                                "int32",
                                "orinfer_dp4a",
                                A[by * bm + i, kg * 32 + l],
                                W[bx * bn + j, kg * 32 + l],
                                acc[i, j, l],
                            )
                            up[i, j, l] = T.call_pure_extern(
                                "int32",
                                "orinfer_dp4a",
                                A[by * bm + i, kg * 32 + l],
                                U[bx * bn + j, kg * 32 + l],
                                up[i, j, l],
                            )
                    else:
                        acc[i, j, l] = T.call_pure_extern(
                            "int32",
                            "orinfer_dp4a",
                            A[by * bm + i, kg * 32 + l],
                            T.cast(weight[j, l], T.int32),
                            acc[i, j, l],
                        )
        T.reduce_sum(acc, total, dim=2)
        if dual:
            T.reduce_sum(up, up_total, dim=2)
        for i, j in T.Parallel(bm, bn):
            if by * bm + i < m and bx * bn + j < n:
                value = rescale(total[i, j], WS[bx * bn + j], AS[by * bm + i])
                if dual:
                    other = rescale(up_total[i, j], US[bx * bn + j], AS[by * bm + i])
                    Out[by * bm + i, bx * bn + j] = swiglu(value, other)
                else:
                    Out[by * bm + i, bx * bn + j] = value

    return project
