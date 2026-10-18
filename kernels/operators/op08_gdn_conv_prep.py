"""Width-four causal depthwise conv, SiLU, L2 and private history (SM87).

Q scaling is an explicit build parameter; default Q is unscaled native prefill.
All buffers are caller owned, contiguous and mutually non-aliasing. No Torch
mathematics is used in this production kernel.
"""

import tilelang
import tilelang.language as T
from tools.operators.common import orin_jit

CHANNELS, KEY_HEADS, VALUE_HEADS, DIM = 10240, 16, 48, 128
Q_SCALE = 1.0


@orin_jit
def gdn_conv_prep(
    B: int | None = None,
    tokens: int | None = None,
    tile_tokens: int = 16,
    normalize_round_fp16: bool = True,
    q_scale: float = 1.0,
    qk_output_dtype: str = "float16",
    conv_product_round_fp16: bool = True,
    weight_dtype: str = "float16",
    compact_decode: bool = False,
    inplace_history: bool = False,
    dynamic_tokens: bool = False,
    key_heads: int = 16,
    value_heads: int = 48,
):
    """Build (X,W,HI,lengths,positions,Q,K,V,HO,positions_out).

    X[B,T,10240] and W[10240,4] FP16; chronological raw HI/HO[B,3,10240]
    FP16. lengths[B]/positions[B] int32; positions is the logical first token
    offset of each independent request. 0 <= lengths <= T. Q/K[B,16,T,128],
    Q/K dtype is qk_output_dtype, V[B,48,T,128] FP16; padded output rows are zero. Empty chunks copy HI and
    positions exactly. Nonempty start position 0 ignores prior history.

    Native FP16 tap products are rounded before FP32 accumulation; False
    selects an explicit FP32-product candidate. FP32 conv+SiLU is rounded
    to FP16 before normalization. Optional half
    rounding of normalized Q matches staged native prefill; False matches
    the native fused-decode normalization. compact_decode removes padding for
    a fixed one-token native tile while retaining its FP32 reduction order. Q's scale is then applied in FP32
    with q_scale and the exported Q cast to qk_output_dtype. Default unscaled
    FP16 output follows native staged prefill; fused decode uses False,
    qk_output_dtype='float32', q_scale=128**-.5. Downstream must apply scaling
    only for q_scale=1.0 and must preserve this mode's rounding contract.
    inplace_history replaces HI/HO with one History argument for fixed B=1
    compact decode. Each head caches its raw tail before updating its own slice.
    """
    KEY_HEADS, VALUE_HEADS, DIM = key_heads, value_heads, 128
    CHANNELS = (2 * KEY_HEADS + VALUE_HEADS) * DIM
    if min(KEY_HEADS, VALUE_HEADS) <= 0 or VALUE_HEADS % KEY_HEADS:
        raise ValueError("Invalid GDN head geometry")
    assert tile_tokens in (1, 2, 4, 8, 16)
    assert qk_output_dtype in ("float16", "float32") and q_scale > 0
    assert weight_dtype in ("float16", "float32")
    if type(compact_decode) is not bool or compact_decode and (tokens != 1 or tile_tokens != 16):
        raise ValueError(
            "Compact decode requires one token and the native 16-token reduction layout"
        )
    if type(inplace_history) is not bool or inplace_history and (not compact_decode or B != 1):
        raise ValueError("In-place history requires a fixed single-request compact decode")
    if compact_decode:
        tile_tokens = 1
    batch = T.dynamic("batch") if B is None else B
    if type(dynamic_tokens) is not bool or dynamic_tokens and (B != 1 or compact_decode):
        raise ValueError("Dynamic token strides require single-request prefill")
    time = T.dynamic("m") if dynamic_tokens else T.dynamic("tokens") if tokens is None else tokens
    threads = min(128, tile_tokens * 16)

    @T.macro
    def body(X, W, HI, lengths, positions, Q, K, V, HO, positions_out):
        with T.Kernel(
            T.ceildiv(time, tile_tokens),
            2 * KEY_HEADS + VALUE_HEADS,
            batch,
            threads=threads,
        ) as (bt, head, b):
            acc = T.alloc_fragment((tile_tokens, DIM), T.float32)
            raw = T.alloc_fragment((tile_tokens, DIM), T.float32)
            activated = T.alloc_fragment((tile_tokens, DIM), T.float16)
            value = T.alloc_fragment((tile_tokens, DIM), T.float32)
            square = T.alloc_fragment((tile_tokens, DIM), T.float32)
            total = T.alloc_fragment((tile_tokens,), T.float32)
            normalized = T.alloc_fragment((tile_tokens, DIM), T.float16)
            product = T.alloc_fragment((tile_tokens, DIM), T.float16)
            # Every time tile retains the native eight adjacent K partials
            # followed by a sixteen-lane reduction. Tile choice changes work
            # distribution, never the normalization's FP32 reduction tree.
            layout = tilelang.Fragment(
                (tile_tokens, DIM),
                forward_thread_fn=lambda ti, d: (ti % (threads // 16)) * 16 + d // 8,
                forward_index_fn=lambda ti, d: (ti // (threads // 16)) * 8 + d % 8,
            )
            T.annotate_layout(
                {
                    acc: layout,
                    raw: layout,
                    activated: layout,
                    value: layout,
                    square: layout,
                    normalized: layout,
                    product: layout,
                }
            )
            if inplace_history:
                tail = T.alloc_fragment((2, DIM), T.float16)
                T.annotate_layout(
                    {
                        tail: tilelang.Fragment(
                            (2, DIM),
                            forward_thread_fn=lambda i, d: d // 8,
                            forward_index_fn=lambda i, d: i * 8 + d % 8,
                        )
                    }
                )
                for i, d in T.Parallel(2, DIM):
                    tail[i, d] = HI[b, i + 1, head * DIM + d]
            T.clear(acc)
            for tap in T.unroll(4):
                for ti, d in T.Parallel(tile_tokens, DIM):
                    raw[ti, d] = 0.0
                    if bt * tile_tokens + ti < lengths[b]:
                        if bt * tile_tokens + ti + tap >= 3:
                            raw[ti, d] = T.cast(
                                X[b, bt * tile_tokens + ti + tap - 3, head * DIM + d], T.float32
                            )
                        elif positions[b] + bt * tile_tokens + ti + tap >= 3:
                            raw[ti, d] = T.cast(
                                HI[b, bt * tile_tokens + ti + tap, head * DIM + d], T.float32
                            )
                    if conv_product_round_fp16:
                        product[ti, d] = raw[ti, d] * T.cast(W[head * DIM + d, tap], T.float32)
                        acc[ti, d] += T.cast(product[ti, d], T.float32)
                    else:
                        acc[ti, d] += raw[ti, d] * T.cast(W[head * DIM + d, tap], T.float32)
            for ti, d in T.Parallel(tile_tokens, DIM):
                activated[ti, d] = acc[ti, d] / (1.0 + T.exp(-acc[ti, d]))
                value[ti, d] = T.cast(activated[ti, d], T.float32)
                square[ti, d] = value[ti, d] * value[ti, d]
            if head < 2 * KEY_HEADS:
                T.reduce_sum(square, total, dim=1)
                for ti, d in T.Parallel(tile_tokens, DIM):
                    if normalize_round_fp16:
                        normalized[ti, d] = value[ti, d] * T.rsqrt(total[ti] + 1e-6)
                        value[ti, d] = T.cast(normalized[ti, d], T.float32)
                    else:
                        value[ti, d] *= T.rsqrt(total[ti] + 1e-6)
                    if bt * tile_tokens + ti < time:
                        if head < KEY_HEADS:
                            Q[b, head, bt * tile_tokens + ti, d] = value[ti, d] * q_scale
                        else:
                            K[b, head - KEY_HEADS, bt * tile_tokens + ti, d] = value[ti, d]
            else:
                for ti, d in T.Parallel(tile_tokens, DIM):
                    if bt * tile_tokens + ti < time:
                        V[b, head - 2 * KEY_HEADS, bt * tile_tokens + ti, d] = activated[ti, d]
            # Only the first time-block writes each channel's final raw history.
            # Q/K are represented once, so shared value heads never race here.
            if bt == 0:
                for i, d in T.Parallel(3, DIM):
                    if lengths[b] == 0:
                        HO[b, i, head * DIM + d] = HI[b, i, head * DIM + d]
                    elif lengths[b] + i >= 3:
                        HO[b, i, head * DIM + d] = X[b, lengths[b] + i - 3, head * DIM + d]
                    elif positions[b] + lengths[b] + i >= 3:
                        if inplace_history:
                            HO[b, i, head * DIM + d] = tail[T.min(i, 1), d]
                        else:
                            HO[b, i, head * DIM + d] = HI[b, lengths[b] + i, head * DIM + d]
                    else:
                        HO[b, i, head * DIM + d] = 0.0
                if head == 0:
                    positions_out[b] = positions[b] + lengths[b]

    if inplace_history:

        @T.prim_func
        def inplace(
            X: T.Tensor((batch, time, CHANNELS), T.float16),
            W: T.Tensor((CHANNELS, 4), weight_dtype),
            History: T.Tensor((batch, 3, CHANNELS), T.float16),
            lengths: T.Tensor((batch,), T.int32),
            positions: T.Tensor((batch,), T.int32),
            Q: T.Tensor((batch, KEY_HEADS, time, DIM), qk_output_dtype),
            K: T.Tensor((batch, KEY_HEADS, time, DIM), qk_output_dtype),
            V: T.Tensor((batch, VALUE_HEADS, time, DIM), T.float16),
            positions_out: T.Tensor((batch,), T.int32),
        ):
            body(X, W, History, lengths, positions, Q, K, V, History, positions_out)

        return inplace.with_attr("global_symbol", "gdn_conv_inplace")

    @T.prim_func
    def kernel(
        X: T.Tensor((batch, time, CHANNELS), T.float16),
        W: T.Tensor((CHANNELS, 4), weight_dtype),
        HI: T.Tensor((batch, 3, CHANNELS), T.float16),
        lengths: T.Tensor((batch,), T.int32),
        positions: T.Tensor((batch,), T.int32),
        Q: T.Tensor((batch, KEY_HEADS, time, DIM), qk_output_dtype),
        K: T.Tensor((batch, KEY_HEADS, time, DIM), qk_output_dtype),
        V: T.Tensor((batch, VALUE_HEADS, time, DIM), T.float16),
        HO: T.Tensor((batch, 3, CHANNELS), T.float16),
        positions_out: T.Tensor((batch,), T.int32),
    ):
        body(X, W, HI, lengths, positions, Q, K, V, HO, positions_out)

    return kernel


def launch(kernel, X, W, HI, lengths, positions, Q, K, V, HO, positions_out, stream=None):
    """Explicit-buffer launch. Resolve the current capture stream at every call."""
    return kernel(X, W, HI, lengths, positions, Q, K, V, HO, positions_out, stream=stream)


def gdn_conv_decode(B: int | None = None, **rounding_and_scale):
    """One token per request, one-time-row tile, same explicit tensor API."""
    return gdn_conv_prep(B=B, tile_tokens=1, **rounding_and_scale)
