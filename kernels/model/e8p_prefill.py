"""Exact short-prefill E8P products with K64-major activation planes.

All warps decode directly into MMA registers. Four private compressed stages
keep loads three K tiles ahead of consumers, hiding cp.async completion waits
without a dedicated producer group. Integer products and rounding match the
row-major exact expert recipe; only transient activation storage changes.
"""

import tilelang
import tilelang.language as T
from tools.operators.common import orin_jit

from kernels.model.e8p_exact import SOURCE as EXACT_SOURCE
from kernels.model.rows import row_count

SOURCE = (
    EXACT_SOURCE
    + r"""
// Move parity out of the selector: flipping its first bit changes the low
// nibble by 0xe, while parity changes every decoded nibble by 0x1.
__device__ __forceinline__ unsigned nib_prefill(unsigned c,unsigned b){unsigned pp=__popc(c&255u)&1u;unsigned sel=(c&0x33u)|((c&0xccu)<<6);return b^__byte_perm(0xeee00e00u,0u,sel)^(pp*0x1111111fu);}
"""
)


@orin_jit
def split_prefill_a8(rows, K, dynamic_rows=False):
    if any(type(x) is not int or x <= 0 for x in (rows, K)) or K % 128:
        raise ValueError("Invalid K64-major A8 geometry")
    W = max(32, 1 << ((K // 8 - 1).bit_length()))
    BM = 4
    rows = row_count(rows, dynamic_rows)

    @T.prim_func
    def pack_exact_a8(
        A: T.Tensor((rows, K), T.int8),
        Low: T.Tensor((K // 64, rows, 8), T.uint32),
        High: T.Tensor((K // 64, rows, 8), T.uint32),
        Sum: T.Tensor((rows,), T.int32),
    ):
        with T.Kernel(T.ceildiv(rows, BM), threads=128) as by:
            T.import_source(SOURCE)
            sums = T.alloc_fragment((BM, W), T.int32)
            total = T.alloc_fragment((BM,), T.int32)
            T.annotate_layout(
                {
                    sums: tilelang.Fragment(
                        (BM, W),
                        forward_thread_fn=lambda i, j: i * 32 + j % 32,
                        forward_index_fn=lambda i, j: j // 32,
                    )
                }
            )
            for i, j in T.Parallel(BM, W):
                sums[i, j] = 0
                if by * BM + i < rows and j < K // 8:
                    x = T.ldg32(A[by * BM + i, j * 8])
                    y = T.ldg32(A[by * BM + i, j * 8 + 4])
                    Low[j // 8, by * BM + i, j % 8] = (
                        (x & T.uint32(0x00000F0F))
                        | ((x >> 12) & T.uint32(0x0000F0F0))
                        | ((y << 16) & T.uint32(0x0F0F0000))
                        | ((y << 4) & T.uint32(0xF0F00000))
                    )
                    High[j // 8, by * BM + i, j % 8] = (
                        ((x >> 4) & T.uint32(0x00000F0F))
                        | ((x >> 16) & T.uint32(0x0000F0F0))
                        | ((y << 12) & T.uint32(0x0F0F0000))
                        | (y & T.uint32(0xF0F00000))
                    )
                    sums[i, j] = T.call_pure_extern("int32", "nib_sum", x, y)
            T.reduce_sum(sums, total, dim=1)
            for i in T.Parallel(BM):
                if by * BM + i < rows:
                    Sum[by * BM + i] = total[i]

    return pack_exact_a8


@orin_jit(
    pass_configs={
        "tl.disable_thread_storage_sync": True,
        "tl.disable_shared_memory_reuse": True,
        "tl.if_stmt_binding_inline_replayable_binds": False,
    }
)
def e8p_prefill_grouped(
    rows, E, tiles, N, K, dynamic_rows=False, sparse_rows=False, row_major=False
):
    """Keep sixteen-row routing tiles, optionally skip their unused upper half.

    The eight-row path has the same K traversal and integer/FP16 boundaries.
    Its uniform device branch depends on the actual expert tail, so graph
    replay can change routing without selecting another host launch.
    """
    if (
        any(type(x) is not int or x <= 0 for x in (rows, E, tiles, N, K))
        or type(sparse_rows) is not bool
        or type(row_major) is not bool
        or K % 128
        or K * 128 * 128 > 2**31 - 1
    ):
        raise ValueError("Invalid exact E8P geometry")
    BM, BN, BK = 16, 128, 64
    # Matching swizzle for sixteen-byte activation halves and ldmatrix reads.
    input_stride, input_mask = 32, 16
    rows = row_count(rows, dynamic_rows)
    # Group-major strides depend on live rows, rather than allocation capacity.
    group_stride, row_stride = (32, K // 2) if row_major else (rows * 32, 32)
    launch_tiles = T.min(tiles, T.ceildiv(rows, BM) + T.min(E, rows)) if dynamic_rows else tiles

    def product(tile_rows):
        @T.macro
        def project(
            Low, High, Sum, P, WS, AS, Out, packed, inputs, book, bx, expert, offset, count
        ):
            tid = T.get_thread_binding()
            lane, warp = tid % 32, tid // 32
            lo = T.alloc_local((tile_rows,), T.int32)
            hi = T.alloc_local((tile_rows,), T.int32)
            v = T.alloc_local((4,), T.uint32)
            a = T.alloc_local((tile_rows // 2,), T.uint32)
            for i in T.unroll(tile_rows):
                lo[i] = 0
                hi[i] = 0
            T.sync_threads()
            for initial in T.unroll(min(3, K // BK)):
                T.ptx_cp_async(
                    T.access_ptr(packed[initial, tid, 0], "w", 8),
                    T.access_ptr(P[expert, initial, T.min(bx * BN + tid, N - 1), 0], "r", 8),
                    8,
                    predicate=bx * BN + tid < N,
                )
                if tid < tile_rows * 2:
                    T.ptx_cp_async(
                        T.access_ptr(
                            inputs[
                                initial,
                                0,
                                tid // 2,
                                ((tid % 2) * 16) ^ (((tid // 8) % 2) * input_mask),
                            ],
                            "w",
                            16,
                        ),
                        T.access_ptr(
                            Low[
                                (initial) * group_stride
                                + T.min(offset + tid // 2, rows - 1) * row_stride
                                + (tid % 2) * 16
                            ],
                            "r",
                            16,
                        ),
                        16,
                        predicate=offset + tid // 2 < rows,
                    )
                    T.ptx_cp_async(
                        T.access_ptr(
                            inputs[
                                initial,
                                1,
                                tid // 2,
                                ((tid % 2) * 16) ^ (((tid // 8) % 2) * input_mask),
                            ],
                            "w",
                            16,
                        ),
                        T.access_ptr(
                            High[
                                (initial) * group_stride
                                + T.min(offset + tid // 2, rows - 1) * row_stride
                                + (tid % 2) * 16
                            ],
                            "r",
                            16,
                        ),
                        16,
                        predicate=offset + tid // 2 < rows,
                    )
                T.ptx_commit_group()
            # Four static stage positions avoid repeated modulo/address
            # calculations. The leading barrier also protects previous
            # readers before their shared slot is prefetched again.
            for group in T.serial(T.ceildiv(K // BK, 4)):
                for sub in T.unroll(4):
                    step = group * 4 + sub
                    if step < K // BK:
                        slot = step % 4
                        if step + 2 < K // BK:
                            T.ptx_wait_group(2)
                        elif step + 1 < K // BK:
                            T.ptx_wait_group(1)
                        else:
                            T.ptx_wait_group(0)
                        T.sync_threads()
                        if step + 3 < K // BK:
                            nxt = (step + 3) % 4
                            T.ptx_cp_async(
                                T.access_ptr(packed[nxt, tid, 0], "w", 8),
                                T.access_ptr(
                                    P[expert, step + 3, T.min(bx * BN + tid, N - 1), 0], "r", 8
                                ),
                                8,
                                predicate=bx * BN + tid < N,
                            )
                            if tid < tile_rows * 2:
                                T.ptx_cp_async(
                                    T.access_ptr(
                                        inputs[
                                            nxt,
                                            0,
                                            tid // 2,
                                            ((tid % 2) * 16) ^ (((tid // 8) % 2) * input_mask),
                                        ],
                                        "w",
                                        16,
                                    ),
                                    T.access_ptr(
                                        Low[
                                            (step + 3) * group_stride
                                            + T.min(offset + tid // 2, rows - 1) * row_stride
                                            + (tid % 2) * 16
                                        ],
                                        "r",
                                        16,
                                    ),
                                    16,
                                    predicate=offset + tid // 2 < rows,
                                )
                                T.ptx_cp_async(
                                    T.access_ptr(
                                        inputs[
                                            nxt,
                                            1,
                                            tid // 2,
                                            ((tid % 2) * 16) ^ (((tid // 8) % 2) * input_mask),
                                        ],
                                        "w",
                                        16,
                                    ),
                                    T.access_ptr(
                                        High[
                                            (step + 3) * group_stride
                                            + T.min(offset + tid // 2, rows - 1) * row_stride
                                            + (tid % 2) * 16
                                        ],
                                        "r",
                                        16,
                                    ),
                                    16,
                                    predicate=offset + tid // 2 < rows,
                                )
                            T.ptx_commit_group()
                        for stripe in T.unroll(tile_rows // 8):
                            r_index = lane % 8 + stripe * 8
                            col = (lane // 8 % 2) * 16
                            T.evaluate(
                                T.call_extern(
                                    "void",
                                    "nib_ldm2",
                                    T.access_ptr(
                                        inputs[
                                            slot,
                                            0,
                                            r_index,
                                            col ^ (((r_index // 4) % 2) * input_mask),
                                        ],
                                        "r",
                                        16,
                                    ),
                                    T.access_ptr(a[stripe * 2], "w", 2),
                                )
                            )
                            T.evaluate(
                                T.call_extern(
                                    "void",
                                    "nib_ldm2",
                                    T.access_ptr(
                                        inputs[
                                            slot,
                                            1,
                                            r_index,
                                            col ^ (((r_index // 4) % 2) * input_mask),
                                        ],
                                        "r",
                                        16,
                                    ),
                                    T.access_ptr(a[tile_rows // 4 + stripe * 2], "w", 2),
                                )
                            )
                        for nt in T.unroll(2):
                            for rr in T.unroll(4):
                                n = warp * 32 + nt * 16 + lane // 4 + (rr % 2) * 8
                                col = lane % 4 + (rr // 2) * 4
                                code = T.cast(packed[slot, n, col], T.uint32)
                                v[rr] = T.call_pure_extern(
                                    "uint32", "nib_prefill", code, book[code >> 8]
                                )
                            for stripe in T.unroll(tile_rows // 8):
                                T.evaluate(
                                    T.call_extern(
                                        "void",
                                        "nib_mma_low",
                                        T.access_ptr(
                                            lo[nt * (tile_rows // 2) + stripe * 4], "rw", 4
                                        ),
                                        v[0],
                                        v[1],
                                        v[2],
                                        v[3],
                                        a[stripe * 2],
                                        a[stripe * 2 + 1],
                                    )
                                )
                                T.evaluate(
                                    T.call_extern(
                                        "void",
                                        "nib_mma_high",
                                        T.access_ptr(
                                            hi[nt * (tile_rows // 2) + stripe * 4], "rw", 4
                                        ),
                                        v[0],
                                        v[1],
                                        v[2],
                                        v[3],
                                        a[tile_rows // 4 + stripe * 2],
                                        a[tile_rows // 4 + 1 + stripe * 2],
                                    )
                                )
            for nt in T.unroll(2):
                for stripe in T.unroll(tile_rows // 8):
                    for r in T.unroll(2):
                        n = bx * BN + warp * 32 + nt * 16 + lane // 4 + r * 8
                        for t in T.unroll(2):
                            r_index = stripe * 8 + (lane % 4) * 2 + t
                            if r_index < count and n < N:
                                j = nt * (tile_rows // 2) + stripe * 4 + r * 2 + t
                                value = 2 * (lo[j] + 16 * hi[j]) + Sum[offset + r_index]
                                Out[offset + r_index, n] = (
                                    T.cast(value, T.float32) * T.cast(WS[expert, n], T.float32)
                                ) * T.cast(AS[offset + r_index], T.float32)

        return project

    full = product(BM)
    sparse = product(8)

    @T.prim_func
    def exact_e8p_int4(
        Low: T.Tensor((rows * K // 2,), T.int8),
        High: T.Tensor((rows * K // 2,), T.int8),
        Sum: T.Tensor((rows,), T.int32),
        P: T.Tensor((E, K // 64, N, 8), T.uint16),
        Book: T.Tensor((1, 256, 1), T.uint32),
        WS: T.Tensor((E, N), T.float16),
        AS: T.Tensor((rows,), T.float16),
        Counts: T.Tensor((E,), T.int32),
        Offsets: T.Tensor((E,), T.int32),
        TE: T.Tensor((tiles,), T.int32),
        TR: T.Tensor((tiles,), T.int32),
        TC: T.Tensor((1,), T.int32),
        Out: T.Tensor((rows, N), T.float16),
    ):
        with T.Kernel(T.ceildiv(N, BN), launch_tiles, threads=128) as (bx, tile):
            T.import_source(SOURCE)
            tid = T.get_thread_binding()
            packed = T.alloc_shared((4, BN, 8), T.uint16)
            inputs = T.alloc_shared((4, 2, BM, input_stride), T.int8)
            book = T.alloc_shared((256,), T.uint32)
            if tile < TC[0]:
                expert, first = TE[tile], TR[tile]
                offset = Offsets[expert] + first
                count = T.min(BM, Counts[expert] - first)
                for i in T.unroll(2):
                    book[tid + i * 128] = Book[0, tid + i * 128, 0]
                # This is uniform across the block. All routing tiles still
                # cover sixteen rows; only a short expert tail uses eight.
                if sparse_rows:
                    if count <= 8:
                        sparse(
                            Low,
                            High,
                            Sum,
                            P,
                            WS,
                            AS,
                            Out,
                            packed,
                            inputs,
                            book,
                            bx,
                            expert,
                            offset,
                            count,
                        )
                    else:
                        full(
                            Low,
                            High,
                            Sum,
                            P,
                            WS,
                            AS,
                            Out,
                            packed,
                            inputs,
                            book,
                            bx,
                            expert,
                            offset,
                            count,
                        )
                else:
                    full(
                        Low,
                        High,
                        Sum,
                        P,
                        WS,
                        AS,
                        Out,
                        packed,
                        inputs,
                        book,
                        bx,
                        expert,
                        offset,
                        count,
                    )

    return exact_e8p_int4


@orin_jit
def dispatch_prefill_a8(M, K=2560, E=512, top_k=10, dynamic_rows=False):
    if any(type(x) is not int or x <= 0 for x in (M, K, E, top_k)) or K % 128 or top_k > E:
        raise ValueError("Invalid direct E8P dispatch geometry")
    BM = 4
    W = max(32, 1 << ((K // 8 - 1).bit_length()))
    M = row_count(M, dynamic_rows)
    slots = M * top_k

    @T.prim_func
    def dispatch_planes(
        A: T.Tensor((M, K), T.int8),
        Scale: T.Tensor((M, 1), T.float16),
        IDs: T.Tensor((M, top_k), T.int32),
        RelativeSlot: T.Tensor((M, top_k), T.int32),
        RowOffsets: T.Tensor((E,), T.int32),
        Low: T.Tensor((K // 64, slots, 8), T.uint32),
        High: T.Tensor((K // 64, slots, 8), T.uint32),
        Sum: T.Tensor((slots,), T.int32),
        DS: T.Tensor((slots, 1), T.float16),
        SlotMap: T.Tensor((M, top_k), T.int32),
    ):
        with T.Kernel(T.ceildiv(slots, BM), threads=BM * 32) as by:
            T.import_source(SOURCE)
            sums = T.alloc_fragment((BM, W), T.int32)
            total = T.alloc_fragment((BM,), T.int32)
            T.annotate_layout(
                {
                    sums: tilelang.Fragment(
                        (BM, W),
                        forward_thread_fn=lambda i, j: i * 32 + j % 32,
                        forward_index_fn=lambda i, j: j // 32,
                    )
                }
            )
            for i, j in T.Parallel(BM, W):
                assignment = by * BM + i
                row = assignment // top_k
                rank = assignment % top_k
                sums[i, j] = 0
                if assignment < slots and j < K // 8:
                    slot = RowOffsets[IDs[row, rank]] + RelativeSlot[row, rank]
                    x = T.ldg32(A[row, j * 8])
                    y = T.ldg32(A[row, j * 8 + 4])
                    Low[j // 8, slot, j % 8] = (
                        (x & T.uint32(0x00000F0F))
                        | ((x >> 12) & T.uint32(0x0000F0F0))
                        | ((y << 16) & T.uint32(0x0F0F0000))
                        | ((y << 4) & T.uint32(0xF0F00000))
                    )
                    High[j // 8, slot, j % 8] = (
                        ((x >> 4) & T.uint32(0x00000F0F))
                        | ((x >> 16) & T.uint32(0x0000F0F0))
                        | ((y << 12) & T.uint32(0x0F0F0000))
                        | (y & T.uint32(0xF0F00000))
                    )
                    sums[i, j] = T.call_pure_extern("int32", "nib_sum", x, y)
            T.reduce_sum(sums, total, dim=1)
            for i in T.Parallel(BM):
                assignment = by * BM + i
                row = assignment // top_k
                rank = assignment % top_k
                if assignment < slots:
                    slot = RowOffsets[IDs[row, rank]] + RelativeSlot[row, rank]
                    Sum[slot] = total[i]
                    DS[slot, 0] = Scale[row, 0]
                    SlotMap[row, rank] = slot

    return dispatch_planes
