"""Single-owner packet boundaries, lifted by the shared dynamic-row wrapper."""

import tilelang.language as T
from tools.operators.common import orin_jit


@orin_jit
def token_input():
    @T.prim_func
    def kernel(Token: T.Tensor((1,), T.int32), Input: T.Tensor((1,), T.int32)):
        with T.Kernel(1, threads=32):
            if T.get_thread_binding() == 0:
                Input[0] = Token[0]

    return kernel


@orin_jit
def outputs(vocab):
    @T.prim_func
    def kernel(
        Source: T.Tensor((vocab,), T.float32),
        Pair: T.Tensor((2,), T.int32),
        Logits: T.Tensor((vocab,), T.float32),
        Token: T.Tensor((1,), T.int32),
        Status: T.Tensor((1,), T.int32),
    ):
        with T.Kernel(T.ceildiv(vocab, 512), threads=128) as block:
            for column in T.Parallel(512):
                index = block * 512 + column
                if index < vocab:
                    Logits[index] = Source[index]
            if block == 0 and T.get_thread_binding() == 0:
                Token[0] = Pair[0]
                Status[0] = Pair[1]

    return kernel
