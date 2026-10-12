"""Short HC up/mix uses a lossless BF16 read cache with explicit synchronization."""

import tilelang
import tilelang.language as T
from tools.operators.common import orin_jit
from kernels.model.rows import row_count
from kernels.model.bf16_mma import SOURCE as BF16_MMA_SOURCE

from kernels.model.half_math import SOURCE as HALF_MATH, stream_sigmoid

source = r"""
#include <tl_templates/cuda/instruction/mma.h>
__device__ __forceinline__ bfloat16_t hc_bf_bits(unsigned bits) {
 unsigned short value=(unsigned short)bits;
 return *reinterpret_cast<bfloat16_t*>(&value);
}
__device__ __forceinline__ unsigned long long hc_expand13_four(unsigned m,unsigned q,unsigned e) {
 unsigned s0=(m&255u)|((m&0xff00u)<<8);
 unsigned s1=((m>>16)&255u)|((m&0xff000000u)>>8);
 unsigned q0=(q&31u)|((q&0x3e0u)<<11);
 unsigned q1=((q>>10)&31u)|((q&0xf8000u)<<1);
 unsigned base=e|(e<<16);
 unsigned v0=(s0&0x007f007fu)|((s0&0x00800080u)<<8)|((q0+base)<<7);
 unsigned v1=(s1&0x007f007fu)|((s1&0x00800080u)<<8)|((q1+base)<<7);
 return v0|(static_cast<unsigned long long>(v1)<<32);
}
__device__ __forceinline__ void hc_store13_four(void* p,unsigned long long v) {
 unsigned s=(unsigned)__cvta_generic_to_shared(p);
 unsigned lo=(unsigned)v,hi=(unsigned)(v>>32);
 asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s),"r"(lo),"r"(hi):"memory");
}
"""


@orin_jit
def hc_up13_mix(
    M: int,
    H: int,
    rank: int,
    streams: int = 4,
    dynamic_rows: bool = False,
    stage_count: int = 1,
    base_exp: int = 99,
    patch_count: int = 0,
):
    """BF16 up projection and branch mix, preserving every FP16 boundary."""
    if (
        not 1 <= M <= 8
        or H != 2560
        or rank not in (256, 320)
        or streams != 4
        or stage_count != 1
        or not 0 <= patch_count <= 3
    ):
        raise ValueError("Unsupported exact BF16 HC read cache geometry")
    small = M <= 8
    bm, bn = (8 if small else 16), 32
    M = row_count(M, dynamic_rows)

    @T.prim_func
    def hc_up13_vector_project(
        A: T.Tensor((M, rank), T.float16),
        Lo: T.Tensor((H // bn, rank // 64, streams * bn, 16), T.uint32),
        Hi: T.Tensor((H // bn, rank // 64, streams * bn, 10), T.uint32),
        Patches: T.Tensor((max(1, patch_count),), T.uint64),
        Normed: T.Tensor((M, streams, H), T.float16),
        Out: T.Tensor((M, H), T.float16),
    ):
        with T.Kernel(T.ceildiv(M, bm), T.ceildiv(H, bn), threads=128) as (by, bx):
            T.import_source(HALF_MATH + source)
            a = T.alloc_shared((bm, 64), T.bfloat16)
            w = T.alloc_shared((streams * bn, 64), T.bfloat16)
            lo = T.alloc_shared((streams * bn, 16), T.uint32)
            hi = T.alloc_shared((streams * bn, 10), T.uint32)
            shape = (streams * bn, bm) if small else (bm, streams * bn)
            acc = T.alloc_fragment(shape, T.float32)
            rounded_up = T.alloc_shared(shape, T.float16)
            total = T.alloc_fragment((bm, bn), T.float32)
            T.annotate_layout(
                {
                    total: tilelang.Fragment(
                        (bm, bn),
                        forward_thread_fn=lambda i, j: (i % 4) * 32 + j,
                        forward_index_fn=lambda i, j: i // 4,
                    )
                }
            )
            T.clear(acc)
            for kg in T.Pipelined(rank // 64, num_stages=stage_count):
                for i, j in T.Parallel(bm, 64):
                    a[i, j] = 0
                    if by * bm + i < M:
                        a[i, j] = A[by * bm + i, kg * 64 + j]
                T.copy(Lo[bx, kg, 0, 0], lo)
                T.copy(Hi[bx, kg, 0, 0], hi)
                for n, j in T.Parallel(streams * bn, 16):
                    pos = j * 20
                    q = (
                        T.call_pure_extern(
                            "uint32",
                            "__funnelshift_r",
                            hi[n, pos // 32],
                            hi[n, T.min(pos // 32 + 1, 9)],
                            pos % 32,
                        )
                        & 1048575
                    )
                    v = T.call_pure_extern(
                        "uint64", "hc_expand13_four", lo[n, j], q, T.uint32(base_exp)
                    )
                    T.call_extern("void", "hc_store13_four", T.access_ptr(w[n, j * 4], "w", 4), v)
                if patch_count:
                    for patch_id in T.unroll(patch_count):
                        patch = Patches[patch_id]
                        index = T.cast(patch >> 16, T.int32)
                        if index // (streams * bn * 64) == bx * (rank // 64) + kg:
                            T.sync_threads()
                            if T.get_thread_binding() == 0:
                                w[(index // 64) % (streams * bn), index % 64] = T.call_pure_extern(
                                    "bfloat16", "hc_bf_bits", T.cast(patch & 65535, T.uint32)
                                )
                            T.sync_threads()
                T.sync_threads()
                # N x 8 MMA avoids the sixteen-row activation tile for small
                # decode/verification batches. Preserve the same K traversal.
                if small:
                    T.gemm(w, a, acc, transpose_B=True)
                else:
                    T.gemm(a, w, acc, transpose_B=True)
            T.copy(acc, rounded_up)
            T.clear(total)
            for branch in T.serial(streams):
                for i, j in T.Parallel(bm, bn):
                    if by * bm + i < M:
                        up = (
                            rounded_up[branch * bn + j, i]
                            if small
                            else rounded_up[i, branch * bn + j]
                        )
                        gate = stream_sigmoid(T.cast(up, T.float32), "float16")
                        product = T.cast(
                            T.cast(gate, T.float32)
                            * T.cast(Normed[by * bm + i, branch, bx * bn + j], T.float32),
                            T.float16,
                        )
                        total[i, j] += T.cast(product, T.float32)
            for i, j in T.Parallel(bm, bn):
                if by * bm + i < M:
                    Out[by * bm + i, bx * bn + j] = total[i, j] / streams

    return hc_up13_vector_project


DIRECT_SOURCE = (
    BF16_MMA_SOURCE
    + r"""
__device__ __forceinline__ unsigned hc_pair13(unsigned mant,unsigned q,unsigned shift,unsigned e){
 unsigned p=(mant>>shift)&65535u;
 unsigned s=(p&255u)|((p&0xff00u)<<8);
 unsigned v=(q&31u)|((q&0x3e0u)<<11);
 return (s&0x007f007fu)|((s&0x00800080u)<<8)|((v+(e|(e<<16)))<<7);
}
"""
)


@orin_jit(
    pass_configs={"tl.disable_thread_storage_sync": True, "tl.disable_shared_memory_reuse": True}
)
def hc_up13_single(H=2560, rank=320, base_exp=99, patch_count=0, from_partials=False):
    if (
        any(type(v) is not int for v in (H, rank, base_exp, patch_count))
        or H != 2560
        or rank not in (256, 320)
        or not 0 <= base_exp <= 225
        or not 0 <= patch_count <= 3
        or type(from_partials) is not bool
    ):
        raise ValueError("Unsupported HC single-token cache geometry")
    bn = 32
    channels = 4 * bn
    input_shape = (4, rank) if from_partials else (1, rank)
    input_dtype = T.float32 if from_partials else T.float16

    @T.prim_func
    def hc_up13_direct(
        A: T.Tensor(input_shape, input_dtype),
        Lo: T.Tensor((80, rank // 64, 128, 16), T.uint32),
        Hi: T.Tensor((80, rank // 64, 128, 10), T.uint32),
        Patches: T.Tensor((max(1, patch_count),), T.uint64),
        Normed: T.Tensor((1, 4, H), T.float16),
        Out: T.Tensor((1, H), T.float16),
    ):
        with T.Kernel(1, H // bn, threads=128) as (by, bx):
            T.import_source(HALF_MATH + DIRECT_SOURCE)
            tid = T.get_thread_binding()
            lane = tid % 32
            warp = tid // 32
            lo = T.alloc_shared((2, channels, 16), T.uint32)
            hi = T.alloc_shared((2, channels * 10), T.uint32)
            live = T.alloc_shared((rank,), T.bfloat16)
            product = T.alloc_shared((4, bn), T.float16)
            acc = T.alloc_local((bn // 4,), T.float32)
            v = T.alloc_local((4,), T.uint32)
            total = T.alloc_local((1,), T.float32)
            if not from_partials:
                for j in T.Parallel(rank):
                    live[j] = A[0, j]
                T.sync_threads()
            for t in T.unroll(bn // 8):
                n = (tid + t * 128) // 4
                col = ((tid + t * 128) % 4) * 4
                T.ptx_cp_async(
                    T.access_ptr(lo[0, n, col ^ (((n // 2) % 4) * 4)], "w", 4),
                    T.access_ptr(
                        Lo[
                            bx // (32 // bn),
                            0,
                            (n // bn) * 32 + (bx % (32 // bn)) * bn + n % bn,
                            col,
                        ],
                        "r",
                        4,
                    ),
                    4,
                )
            for t in T.unroll((channels * 10 + 511) // 512):
                index = (tid + t * 128) * 4
                if index < channels * 10:
                    n = index // 10
                    T.ptx_cp_async(
                        T.access_ptr(hi[0, index], "w", 4),
                        T.access_ptr(
                            Hi[
                                bx // (32 // bn),
                                0,
                                (n // bn) * 32 + (bx % (32 // bn)) * bn + n % bn,
                                index % 10,
                            ],
                            "r",
                            4,
                        ),
                        4,
                    )
            T.ptx_commit_group()
            if from_partials:
                padded = T.ceildiv(rank, 128) * 128
                partial = T.alloc_fragment((padded,), T.float32)
                T.annotate_layout(
                    {
                        partial: tilelang.Fragment(
                            (padded,),
                            forward_thread_fn=lambda j: j % 128,
                            forward_index_fn=lambda j: j // 128,
                        )
                    }
                )
                T.clear(partial)
                for branch in T.unroll(4):
                    for j in T.Parallel(rank):
                        partial[j] += A[branch, j]
                for j in T.Parallel(rank):
                    # Match HC down finish, including both FP16 roundings.
                    rounded = T.cast(partial[j], T.float16)
                    value = T.cast(T.cast(T.cast(rounded, T.float32) / 4, T.float16), T.float32)
                    live[j] = T.cast(value / (1 + T.exp(-value)), T.float16)
            for j in T.unroll(bn // 4):
                acc[j] = 0
            for kg in T.serial(rank // 64):
                slot = kg % 2
                T.ptx_wait_group(0)
                T.sync_threads()
                if kg + 1 < rank // 64:
                    nxt = (kg + 1) % 2
                    for t in T.unroll(bn // 8):
                        n = (tid + t * 128) // 4
                        col = ((tid + t * 128) % 4) * 4
                        T.ptx_cp_async(
                            T.access_ptr(lo[nxt, n, col ^ (((n // 2) % 4) * 4)], "w", 4),
                            T.access_ptr(
                                Lo[
                                    bx // (32 // bn),
                                    kg + 1,
                                    (n // bn) * 32 + (bx % (32 // bn)) * bn + n % bn,
                                    col,
                                ],
                                "r",
                                4,
                            ),
                            4,
                        )
                    for t in T.unroll((channels * 10 + 511) // 512):
                        index = (tid + t * 128) * 4
                        if index < channels * 10:
                            n = index // 10
                            T.ptx_cp_async(
                                T.access_ptr(hi[nxt, index], "w", 4),
                                T.access_ptr(
                                    Hi[
                                        bx // (32 // bn),
                                        kg + 1,
                                        (n // bn) * 32 + (bx % (32 // bn)) * bn + n % bn,
                                        index % 10,
                                    ],
                                    "r",
                                    4,
                                ),
                                4,
                            )
                    T.ptx_commit_group()
                for kk in T.unroll(4):
                    b0 = T.call_pure_extern(
                        "uint32",
                        "hc_raw_pair",
                        T.access_ptr(live[kg * 64 + kk * 16 + (lane % 4) * 2], "r", 2),
                    )
                    b1 = T.call_pure_extern(
                        "uint32",
                        "hc_raw_pair",
                        T.access_ptr(live[kg * 64 + kk * 16 + (lane % 4) * 2 + 8], "r", 2),
                    )
                    for nt in T.unroll(bn // 16):
                        for rr in T.unroll(4):
                            n = warp * bn + nt * 16 + lane // 4 + (rr % 2) * 8
                            col = kk * 16 + (lane % 4) * 2 + (rr // 2) * 8
                            pos = col * 5
                            q = (
                                T.call_pure_extern(
                                    "uint32",
                                    "__funnelshift_r",
                                    hi[slot, n * 10 + pos // 32],
                                    hi[slot, n * 10 + T.min(pos // 32 + 1, 9)],
                                    pos % 32,
                                )
                                & 1023
                            )
                            m = lo[slot, n, (col // 4) ^ (((n // 2) % 4) * 4)]
                            v[rr] = T.call_pure_extern(
                                "uint32", "hc_pair13", m, q, (col % 4) * 8, T.uint32(base_exp)
                            )
                            if patch_count:
                                for p in T.unroll(patch_count):
                                    patch = Patches[p]
                                    index = T.cast(patch >> 16, T.int32)
                                    ng = (n // bn) * 32 + (bx % (32 // bn)) * bn + n % bn
                                    current = (
                                        ((bx // (32 // bn)) * (rank // 64) + kg) * 128 + ng
                                    ) * 64 + col
                                    if index == current:
                                        v[rr] = (v[rr] & T.uint32(0xFFFF0000)) | T.cast(
                                            patch & 65535, T.uint32
                                        )
                                    if index == current + 1:
                                        v[rr] = (v[rr] & T.uint32(65535)) | (
                                            T.cast(patch & 65535, T.uint32) << 16
                                        )
                        T.evaluate(
                            T.call_extern(
                                "void",
                                "hc_mma_bf16",
                                T.access_ptr(acc[nt * 4], "rw", 4),
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
                for nt in T.unroll(bn // 16):
                    for rr in T.unroll(2):
                        col = nt * 16 + lane // 4 + rr * 8
                        up = T.cast(acc[nt * 4 + rr * 2], T.float16)
                        gate = stream_sigmoid(T.cast(up, T.float32), "float16")
                        product[warp, col] = T.cast(
                            T.cast(gate, T.float32)
                            * T.cast(Normed[0, warp, bx * bn + col], T.float32),
                            T.float16,
                        )
            T.sync_threads()
            if tid < bn:
                total[0] = 0
                for branch in T.unroll(4):
                    total[0] += T.cast(product[branch, tid], T.float32)
                Out[0, bx * bn + tid] = total[0] / 4

    if from_partials:
        return hc_up13_direct.with_attr("global_symbol", "hc_up13_finish")
    return hc_up13_direct
