"""Causal GQA rows with shared INT8 KV tiles and bounded split workspace.

A row is a query token, independent of request/batch/MTP meaning. The caller
binds one owner's KV and causal positions. Query heads sharing KV are packed
into the MMA row axis. Split count follows the live grid geometry; its target
and tile sizes are selected offline, never from a context-length enumeration.
"""

import tilelang.language as T
from tools.operators.common import orin_jit
from kernels.model.rows import row_count, explicit_rows
from kernels.model.kv_int8 import HALF2_SOURCE, int8_kv_tile


SCALAR_SOURCE = r"""
__device__ __forceinline__ int attn_split_count(int rows,int groups,int bm,int target,int maximum) {
    int tiles=(rows*groups+bm-1)/bm;
    return min(maximum,(target+tiles-1)/tiles);
}
"""


def geometry(rows, block_m, target_tiles, max_splits, groups):
    tiles = T.ceildiv(rows * groups, block_m)
    splits = T.min(max_splits, T.ceildiv(target_tiles, tiles))
    return tiles, splits


def gate_result(value, raw):
    return T.cast(T.cast(value, T.float16), T.float32) * T.cast(
        T.cast(1 / (1 + T.exp(-T.cast(raw, T.float32))), T.float16), T.float32
    )


@orin_jit
def causal_attention_int8(
    capacity,
    context,
    block_m=32,
    block_n=64,
    target_tiles=16,
    max_splits=8,
    query_heads=24,
    kv_heads=4,
    dimension=256,
    head_group=None,
):
    groups = query_heads // kv_heads if head_group is None else head_group
    if (
        min(capacity, context, target_tiles) <= 0
        or query_heads % kv_heads
        or dimension % 64
        or block_m not in (16, 32, 64)
        or block_n not in (32, 64)
        or max_splits not in (1, 2, 4, 8, 16)
        or groups <= 0
        or (query_heads // kv_heads) % groups
    ):
        raise ValueError("Invalid dynamic GQA geometry")
    m = row_count(capacity, True)
    work_heads = query_heads // groups
    heads_per_kv = query_heads // kv_heads // groups
    tiles, splits = geometry(m, block_m, target_tiles, max_splits, groups)
    # splits > 1 implies tiles < target_tiles, and tiles*splits < 2*target_tiles.
    workspace_tiles = 2 * target_tiles
    load_kv = int8_kv_tile(block_n, dimension)

    @T.prim_func
    def main(
        Q: T.Tensor((m, query_heads, dimension), T.float16),
        K: T.Tensor((1, context, kv_heads * dimension // 4), T.uint32),
        V: T.Tensor((1, context, kv_heads * dimension // 4), T.uint32),
        KS: T.Tensor((1, context, kv_heads * dimension // 64), T.float16),
        VS: T.Tensor((1, context, kv_heads * dimension // 64), T.float16),
        Gate: T.Tensor((m, query_heads, dimension), T.float16),
        Positions: T.Tensor((m,), T.int32),
        Lengths: T.Tensor((1,), T.int32),
        Y: T.Tensor((m, query_heads, dimension), T.float16),
        Partial: T.Tensor((work_heads, workspace_tiles, block_m, dimension), T.float32),
        Maximum: T.Tensor((work_heads, workspace_tiles, block_m), T.float32),
        Denominator: T.Tensor((work_heads, workspace_tiles, block_m), T.float32),
    ):
        with T.Kernel(tiles, work_heads, splits, threads=128) as (qb, kh, split):
            T.import_source(HALF2_SOURCE + SCALAR_SOURCE)
            split_count = T.alloc_local((1,), T.int32)
            split_count[0] = T.call_pure_extern(
                "int32", "attn_split_count", m, groups, block_m, target_tiles, max_splits
            )
            q = T.alloc_shared((block_m, dimension), T.float16)
            k = T.alloc_shared((block_n, dimension), T.float16)
            v = T.alloc_shared((block_n, dimension), T.float16)
            p = T.alloc_shared((block_m, block_n), T.float16)
            score = T.alloc_fragment((block_m, block_n), T.float32)
            out = T.alloc_fragment((block_m, dimension), T.float32)
            maximum = T.alloc_fragment((block_m,), T.float32)
            previous = T.alloc_fragment((block_m,), T.float32)
            denom = T.alloc_fragment((block_m,), T.float32)
            rowsum = T.alloc_fragment((block_m,), T.float32)
            correction = T.alloc_fragment((block_m,), T.float32)
            pos = T.alloc_fragment((block_m,), T.int32)
            maxpos = T.alloc_fragment((1,), T.int32)
            T.clear(out)
            T.clear(denom)
            T.fill(maximum, -1e30)
            for i in T.Parallel(block_m):
                token = (qb * block_m + i) // groups
                pos[i] = T.if_then_else(token < m, Positions[token], -1)
            T.reduce_max(pos, maxpos, dim=0)
            valid = T.max(0, T.min(T.min(Lengths[0], context), maxpos[0] + 1))
            width = T.alloc_local((1,), T.int32)
            width[0] = T.truncdiv(valid + split_count[0] - 1, split_count[0])
            start = split * width[0]
            end = T.min(start + width[0], valid)
            for i, d in T.Parallel(block_m, dimension):
                token = (qb * block_m + i) // groups
                q[i, d] = 0.0
                if token < m:
                    q[i, d] = Q[token, kh * groups + (qb * block_m + i) % groups, d]
            for tile in T.serial(T.ceildiv(T.max(0, end - start), block_n)):
                base = start + tile * block_n
                load_kv(K, V, KS, VS, k, v, 0, kh // heads_per_kv, base, end)
                T.sync_threads()
                T.clear(score)
                T.gemm(q, k, score, transpose_B=True)
                for i in T.Parallel(block_m):
                    previous[i] = maximum[i]
                for i, j in T.Parallel(block_m, block_n):
                    score[i, j] = T.if_then_else(
                        base + j < end and base + j <= pos[i],
                        score[i, j] * (dimension**-0.5),
                        -1e30,
                    )
                T.reduce_max(score, maximum, dim=1, clear=False)
                for i, j in T.Parallel(block_m, block_n):
                    score[i, j] = T.if_then_else(
                        base + j < end and base + j <= pos[i],
                        T.exp(score[i, j] - maximum[i]),
                        0,
                    )
                    p[i, j] = score[i, j]
                T.reduce_sum(score, rowsum, dim=1)
                for i in T.Parallel(block_m):
                    correction[i] = T.exp(previous[i] - maximum[i])
                    denom[i] = denom[i] * correction[i] + rowsum[i]
                for i, d in T.Parallel(block_m, dimension):
                    out[i, d] *= correction[i]
                T.gemm(p, v, out)
            if split_count[0] > 1:
                for i, d in T.Parallel(block_m, dimension):
                    Partial[kh, qb * split_count[0] + split, i, d] = out[i, d]
                for i in T.Parallel(block_m):
                    Maximum[kh, qb * split_count[0] + split, i] = maximum[i]
                    Denominator[kh, qb * split_count[0] + split, i] = denom[i]
            else:
                for i, d in T.Parallel(block_m, dimension):
                    token = (qb * block_m + i) // groups
                    head = kh * groups + (qb * block_m + i) % groups
                    if token < m:
                        value = T.if_then_else(denom[i] > 0, out[i, d] / T.max(denom[i], 1e-30), 0)
                        Y[token, head, d] = gate_result(value, Gate[token, head, d])

    return explicit_rows(main)


@orin_jit
def merge_attention_int8(
    capacity,
    block_m=32,
    target_tiles=16,
    max_splits=8,
    query_heads=24,
    kv_heads=4,
    dimension=256,
    head_group=None,
):
    m = row_count(capacity, True)
    groups = query_heads // kv_heads if head_group is None else head_group
    if groups <= 0 or (query_heads // kv_heads) % groups or max_splits not in (1, 2, 4, 8, 16):
        raise ValueError("Invalid query-head group")
    reduction_levels = max_splits.bit_length() - 1
    work_heads = query_heads // groups
    _, splits = geometry(m, block_m, target_tiles, max_splits, groups)
    workspace_tiles = 2 * target_tiles
    grid = T.min(m * query_heads, 1 + (splits - 1) * target_tiles * block_m * query_heads)

    @T.prim_func
    def main(
        Partial: T.Tensor((work_heads, workspace_tiles, block_m, dimension), T.float32),
        Maximum: T.Tensor((work_heads, workspace_tiles, block_m), T.float32),
        Denominator: T.Tensor((work_heads, workspace_tiles, block_m), T.float32),
        Gate: T.Tensor((m, query_heads, dimension), T.float16),
        Y: T.Tensor((m, query_heads, dimension), T.float16),
    ):
        with T.Kernel(grid, threads=128) as work:
            T.import_source(SCALAR_SOURCE)
            split_count = T.alloc_local((1,), T.int32)
            split_count[0] = T.call_pure_extern(
                "int32", "attn_split_count", m, groups, block_m, target_tiles, max_splits
            )
            if split_count[0] > 1:
                token = work // query_heads
                head = work % query_heads
                kh = head // groups
                flat = token * groups + head % groups
                qb = flat // block_m
                qi = flat % block_m
                # Each thread owns output columns. Keep only a short split
                # vector live, rather than a split-by-head-dimension fragment.
                # Both sums use a balanced FP32 tree before FP16 gate rounding.
                maxima = T.alloc_local((max_splits,), T.float32)
                factors = T.alloc_local((max_splits,), T.float32)
                weights = T.alloc_local((max_splits,), T.float32)
                values = T.alloc_local((max_splits,), T.float32)
                maximum = T.alloc_local((1,), T.float32)
                maximum[0] = -1e30
                for s in T.unroll(max_splits):
                    maxima[s] = -1e30
                    if s < split_count[0] and Denominator[kh, qb * split_count[0] + s, qi] > 0:
                        maxima[s] = Maximum[kh, qb * split_count[0] + s, qi]
                    maximum[0] = T.max(maximum[0], maxima[s])
                for s in T.unroll(max_splits):
                    factors[s] = T.exp(maxima[s] - maximum[0])
                    weights[s] = 0.0
                    if s < split_count[0]:
                        weights[s] = factors[s] * Denominator[kh, qb * split_count[0] + s, qi]
                for level in T.unroll(reduction_levels):
                    for s in T.unroll(max_splits >> (level + 1)):
                        weights[s] = weights[2 * s] + weights[2 * s + 1]
                for column in T.unroll((dimension + 127) // 128):
                    d = column * 128 + T.get_thread_binding()
                    if d < dimension:
                        for s in T.unroll(max_splits):
                            values[s] = 0.0
                            if s < split_count[0]:
                                values[s] = factors[s] * Partial[kh, qb * split_count[0] + s, qi, d]
                        for level in T.unroll(reduction_levels):
                            for s in T.unroll(max_splits >> (level + 1)):
                                values[s] = values[2 * s] + values[2 * s + 1]
                        value = T.if_then_else(
                            weights[0] > 0, values[0] / T.max(weights[0], 1e-30), 0
                        )
                        Y[token, head, d] = gate_result(value, Gate[token, head, d])
            else:
                # Keep a real buffer argument when an unsplit specialization
                # folds the merge away; NVRTC's adapter rejects zero arguments.
                if T.get_thread_binding() == 0:
                    Denominator[0, 0, 0] = 0.0

    return explicit_rows(main)
