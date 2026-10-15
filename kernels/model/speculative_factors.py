"""Preserve the exact compact GDN factors until each request commits."""

import tilelang.language as T
from tools.operators.common import orin_jit
from kernels.model.rows import row_count


@orin_jit
def save_gdn_factors(tokens: int, dynamic_rows: bool = False):
    if type(tokens) is not int or not 2 <= tokens <= 8:
        raise ValueError("Unsupported verification factor width")

    tokens = row_count(tokens, dynamic_rows)

    @T.prim_func
    def save(
        K: T.Tensor((16, tokens, 128), T.float16),
        G: T.Tensor((tokens, 48), T.float32),
        SavedK: T.Tensor((16, tokens, 128), T.float16),
        SavedG: T.Tensor((tokens, 48), T.float32),
    ):
        with T.Kernel(16 * tokens, threads=128) as b:
            for d in T.Parallel(128):
                SavedK[b // tokens, b % tokens, d] = K[b // tokens, b % tokens, d]
            if b < tokens:
                for h in T.Parallel(48):
                    SavedG[b, h] = G[b, h]

    return save
