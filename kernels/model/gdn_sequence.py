"""Causal small-sequence GDN with exact prefix snapshots for speculation.

Each CTA keeps its initial FP32 state in registers while processing a chunk.
Verification saves either full per-position states or compact rank-one updates;
the caller commits only the accepted prefix, including the target token.
In-place prefill writes only the final state and supports chunks through 4096.
Convolution history, position, and attention KV require separate handling.
"""

import tilelang
import tilelang.language as T
from tools.operators.common import orin_jit

HK, HV, DK, DV = 16, 48, 128, 128


def _gdn_sequence(
    tokens: int,
    q_scale: float = 128**-0.5,
    value_tile: int = 32,
    threads: int = 128,
    in_place: bool = False,
    compact: bool = False,
    prefetch_qk: bool = False,
):
    assert 1 <= tokens <= (4096 if in_place else 16) and q_scale > 0
    assert value_tile in (16, 32, 64, 128) and threads in (32, 64, 128, 256)
    assert not (compact and in_place)
    if (
        type(prefetch_qk) is not bool
        or prefetch_qk
        and (not in_place or compact or threads != 32 or value_tile != 16)
    ):
        raise ValueError("Q/K prefetch requires a warp-owned in-place GDN tile")

    @T.prim_func
    def main(
        Q: T.Tensor((HK, tokens, DK), T.float16),
        K: T.Tensor((HK, tokens, DK), T.float16),
        V: T.Tensor((HV, tokens, DV), T.float16),
        G: T.Tensor((tokens, HV), T.float32),
        Beta: T.Tensor((tokens, HV), T.float32),
        State: T.Tensor((HV, DK, DV), T.float32),
        Prefix: T.Tensor(
            (tokens, HV, DV) if compact else (1 if in_place else tokens, HV, DK, DV), T.float32
        ),
        Out: T.Tensor((tokens, HV * DV), T.float16),
    ):
        with T.Kernel(DV // value_tile, HV, threads=threads) as (bv, h):
            kh = h // 3
            if prefetch_qk:
                tid = T.get_thread_binding()
                qs = T.alloc_shared((2, DK), T.float16)
                ks = T.alloc_shared((2, DK), T.float16)
                vs = T.alloc_shared((2, value_tile), T.float16)
                gs = T.alloc_shared((2,), T.float32)
                bs = T.alloc_shared((2,), T.float32)
            state = T.alloc_fragment((DK, value_tile), T.float32)
            product = T.alloc_fragment((DK, value_tile), T.float32)
            if tokens <= 8 and threads == 128 and value_tile == 32:
                # Permute thread ownership without changing each lane's eight
                # K partials or the sixteen-lane reduction tree. Both dot
                # products now reduce within a warp rather than through a
                # block-wide shared-memory barrier.
                T.annotate_layout(
                    {
                        state: tilelang.Fragment(
                            (DK, value_tile),
                            forward_thread_fn=lambda i, j: (
                                (j // 8) * 32 + (i % 16) * 2 + (j // 4) % 2
                            ),
                            forward_index_fn=lambda i, j: (i // 16) * 4 + j % 4,
                        ),
                        product: tilelang.Fragment(
                            (DK, value_tile),
                            forward_thread_fn=lambda i, j: (
                                (j // 8) * 32 + (i % 16) * 2 + (j // 4) % 2
                            ),
                            forward_index_fn=lambda i, j: (i // 16) * 4 + j % 4,
                        ),
                    }
                )
            predicted = T.alloc_fragment((value_tile,), T.float32)
            result = T.alloc_fragment((value_tile,), T.float32)
            update = T.alloc_fragment((value_tile,), T.float32)
            for i, j in T.Parallel(DK, value_tile):
                state[i, j] = State[h, i, bv * value_tile + j]
            if prefetch_qk:
                T.ptx_cp_async(
                    T.access_ptr(qs[0, tid * 4], "w", 4),
                    T.access_ptr(Q[kh, 0, tid * 4], "r", 4),
                    4,
                )
                T.ptx_cp_async(
                    T.access_ptr(ks[0, tid * 4], "w", 4),
                    T.access_ptr(K[kh, 0, tid * 4], "r", 4),
                    4,
                )
                if tid < value_tile // 4:
                    T.ptx_cp_async(
                        T.access_ptr(vs[0, tid * 4], "w", 4),
                        T.access_ptr(V[h, 0, bv * value_tile + tid * 4], "r", 4),
                        4,
                    )
                if tid == 0:
                    T.ptx_cp_async(T.access_ptr(gs[0], "w", 1), T.access_ptr(G[0, h], "r", 1), 1)
                    T.ptx_cp_async(T.access_ptr(bs[0], "w", 1), T.access_ptr(Beta[0, h], "r", 1), 1)
                T.ptx_commit_group()
            for t in T.serial(tokens):
                if prefetch_qk:
                    T.ptx_wait_group(0)
                    T.sync_warp()
                    if t + 1 < tokens:
                        T.ptx_cp_async(
                            T.access_ptr(qs[(t + 1) % 2, tid * 4], "w", 4),
                            T.access_ptr(Q[kh, t + 1, tid * 4], "r", 4),
                            4,
                        )
                        T.ptx_cp_async(
                            T.access_ptr(ks[(t + 1) % 2, tid * 4], "w", 4),
                            T.access_ptr(K[kh, t + 1, tid * 4], "r", 4),
                            4,
                        )
                        if tid < value_tile // 4:
                            T.ptx_cp_async(
                                T.access_ptr(vs[(t + 1) % 2, tid * 4], "w", 4),
                                T.access_ptr(V[h, t + 1, bv * value_tile + tid * 4], "r", 4),
                                4,
                            )
                        if tid == 0:
                            T.ptx_cp_async(
                                T.access_ptr(gs[(t + 1) % 2], "w", 1),
                                T.access_ptr(G[t + 1, h], "r", 1),
                                1,
                            )
                            T.ptx_cp_async(
                                T.access_ptr(bs[(t + 1) % 2], "w", 1),
                                T.access_ptr(Beta[t + 1, h], "r", 1),
                                1,
                            )
                        T.ptx_commit_group()
                for i, j in T.Parallel(DK, value_tile):
                    state[i, j] = state[i, j] * T.exp(gs[t % 2] if prefetch_qk else G[t, h])
                    product[i, j] = state[i, j] * T.cast(
                        (ks[t % 2, i] if prefetch_qk else K[kh, t, i]), T.float32
                    )
                T.reduce_sum(product, predicted, dim=0)
                if compact:
                    for j in T.Parallel(value_tile):
                        update[j] = Beta[t, h] * (
                            T.cast(V[h, t, bv * value_tile + j], T.float32) - predicted[j]
                        )
                        Prefix[t, h, bv * value_tile + j] = update[j]
                for i, j in T.Parallel(DK, value_tile):
                    delta = (bs[t % 2] if prefetch_qk else Beta[t, h]) * (
                        T.cast(
                            vs[t % 2, j] if prefetch_qk else V[h, t, bv * value_tile + j],
                            T.float32,
                        )
                        - predicted[j]
                    )
                    if compact:
                        delta = update[j]
                    state[i, j] = (
                        state[i, j]
                        + T.cast((ks[t % 2, i] if prefetch_qk else K[kh, t, i]), T.float32) * delta
                    )
                    product[i, j] = state[i, j] * (
                        T.cast((qs[t % 2, i] if prefetch_qk else Q[kh, t, i]), T.float32) * q_scale
                    )
                T.reduce_sum(product, result, dim=0)
                for i, j in T.Parallel(DK, value_tile):
                    if in_place:
                        if t == tokens - 1:
                            State[h, i, bv * value_tile + j] = state[i, j]
                    elif not compact:
                        Prefix[t, h, i, bv * value_tile + j] = state[i, j]
                for j in T.Parallel(value_tile):
                    Out[t, h * DV + bv * value_tile + j] = result[j]

    return main


_compile_sequence = orin_jit(_gdn_sequence)
_compile_prefetch = orin_jit(
    _gdn_sequence,
    pass_configs={
        "tl.disable_thread_storage_sync": True,
        "tl.disable_shared_memory_reuse": True,
    },
)


def gdn_sequence(
    tokens: int,
    q_scale: float = 128**-0.5,
    value_tile: int = 32,
    threads: int = 128,
    in_place: bool = False,
    compact: bool = False,
    prefetch_qk: bool = False,
):
    """Preserve the recurrence and reductions while overlapping input reads.

    The optional prefill path has one warp per CTA and two private shared
    slots. Its explicit wait and warp barrier protect every slot reuse.
    """
    compiler = _compile_prefetch if prefetch_qk else _compile_sequence
    return compiler(tokens, q_scale, value_tile, threads, in_place, compact, prefetch_qk)


@orin_jit
def gdn_commit(tokens: int, value_tile: int = 32):
    """Replay saved FP32 rank-one updates in their original rounding order."""
    assert 1 <= tokens <= 8 and value_tile in (16, 32, 64, 128)

    @T.prim_func
    def main(
        K: T.Tensor((HK, tokens, DK), T.float16),
        G: T.Tensor((tokens, HV), T.float32),
        Update: T.Tensor((tokens, HV, DV), T.float32),
        Accepted: T.Tensor((1,), T.int32),
        State: T.Tensor((HV, DK, DV), T.float32),
    ):
        with T.Kernel(DV // value_tile, HV, threads=128) as (bv, h):
            state = T.alloc_fragment((DK, value_tile), T.float32)
            for i, j in T.Parallel(DK, value_tile):
                state[i, j] = State[h, i, bv * value_tile + j]
            for t in T.serial(tokens):
                if t < Accepted[0]:
                    for i, j in T.Parallel(DK, value_tile):
                        state[i, j] = state[i, j] * T.exp(G[t, h])
                        state[i, j] = (
                            state[i, j]
                            + T.cast(K[h // 3, t, i], T.float32) * Update[t, h, bv * value_tile + j]
                        )
            for i, j in T.Parallel(DK, value_tile):
                State[h, i, bv * value_tile + j] = state[i, j]

    return main
