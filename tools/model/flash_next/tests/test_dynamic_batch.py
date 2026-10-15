"""CPU compiler checks for the affine-shape host ABI regression."""

import unittest
from tvm.arith import Analyzer
from tvm import tirx as tir

from kernels.model.rows import explicit_rows
from tools.model.flash_next.dynamic_batch import factory
from tools.model.flash_next.decode_tuning import make_prim


class DynamicRowsTests(unittest.TestCase):
    def test_metadata_scalars_survive_independent_state_wrapping(self):
        from kernels.model import qsa
        from kernels.model.hc_bf16_cache import hc_up13_mix
        from kernels.model.independent_rows import independent_rows

        hc = explicit_rows(
            hc_up13_mix.__wrapped__(1024, 2560, 320, base_exp=None, dynamic_rows=True)
        )
        self.assertEqual({p.name for p in hc.params if p.dtype == "int32"}, {"m", "base_exp"})
        for primitive in (qsa.radix_histogram, qsa.radix_choose):
            fn = primitive.__wrapped__(1, 262144, None)
            roles = {b.name: ("private", "Scratch") for b in fn.buffer_map.values()}
            fn, identities = independent_rows(fn, 128, roles, dict(Scratch=0), dynamic=True)
            self.assertIn("shift", identities)
            self.assertEqual({p.name for p in fn.params if p.dtype == "int32"}, {"m", "shift"})
            self.assertFalse(
                tir.analysis.undefined_vars(
                    fn.body, list(fn.params) + [b.data for b in fn.buffer_map.values()]
                )
            )

    def test_moe_combine_bound_tracks_live_assignments_beyond_initial_capacity(self):
        for capacity in [8, 16, 128]:
            fn = explicit_rows(factory(f"combine-{capacity}", capacity))
            (rows,) = [p for p in fn.params if str(p.dtype) == "int32"]
            buffers = {b.name: b for b in fn.buffer_map.values()}
            bound = buffers["Expert"].shape[0]
            for live in [1, 9, 17, 33, 128]:
                actual = Analyzer().simplify(tir.stmt_functor.substitute(bound, {rows: live}))
                self.assertEqual(int(actual), live * 10)

    def test_fused_shared_ffn_launch_covers_multiple_row_tiles(self):
        fn = explicit_rows(factory("shared-w8-swiglu-8", 128))
        (rows,) = [p for p in fn.params if str(p.dtype) == "int32"]
        buffers = {b.name: b for b in fn.buffer_map.values()}
        self.assertTrue(buffers["A"].shape[0].same_as(rows))
        self.assertTrue(buffers["Output"].shape[0].same_as(rows))
        extents = []

        def visit(node):
            if isinstance(node, tir.For) and node.thread_binding is not None:
                if node.thread_binding.thread_tag == "blockIdx.y":
                    extents.append(node.extent)

        tir.stmt_functor.post_order_visit(fn.body, visit)
        self.assertEqual(len(extents), 1)
        for live in [1, 8, 9, 16, 17, 33, 127, 128]:
            extent = Analyzer().simplify(tir.stmt_functor.substitute(extents[0], {rows: live}))
            self.assertEqual(int(extent), (live + 15) // 16)

    def test_decode_factory_uses_one_row_contract_for_both_integer_algorithms(self):
        for dynamic in [False, True]:
            for rows in [1, 3, 16, 32, 128]:
                for algorithm, tile in [("dp4a", min(rows, 8)), ("tensor_core", 16)]:
                    fn = make_prim(
                        dict(
                            kind="q4",
                            rows=rows,
                            outputs=128,
                            inputs=256,
                            dtype="float32",
                            dynamic=dynamic,
                        ),
                        dict(
                            algorithm=algorithm,
                            block_m=tile,
                            block_n=16 if algorithm == "dp4a" else 64,
                        ),
                    )
                    buffers = {b.name: b for b in fn.buffer_map.values()}
                    scalars = [p for p in fn.params if str(p.dtype) == "int32"]
                    self.assertEqual(len(scalars), int(dynamic))
                    for name in ["A", "AS", "Out"]:
                        if dynamic:
                            self.assertTrue(buffers[name].shape[0].same_as(scalars[0]))
                        else:
                            self.assertEqual(int(buffers[name].shape[0]), rows)

    def test_large_w8_dp4a_uses_configured_tiles_and_live_rows(self):
        for rows, dynamic in [(1, False), (8, False), (128, True)]:
            for columns in [4, 8, 16]:
                fn = make_prim(
                    dict(
                        kind="w8",
                        rows=rows,
                        outputs=640,
                        inputs=2560,
                        dtype="float16",
                        dynamic=dynamic,
                    ),
                    dict(algorithm="dp4a", block_m=min(rows, 4), block_n=columns, block_k=128),
                )
                buffers = {b.name: b for b in fn.buffer_map.values()}
                self.assertEqual(int(buffers["Weight"].shape[0]), 640)
                scalars = [p for p in fn.params if str(p.dtype) == "int32"]
                self.assertEqual(len(scalars), int(dynamic))
                if dynamic:
                    self.assertTrue(buffers["Output"].shape[0].same_as(scalars[0]))

    def test_fused_ffn_algorithms_share_live_rows_and_exact_packed_bytes(self):
        for dynamic, rows in [(False, 1), (False, 4), (True, 128)]:
            for algorithm, bm, bn in [("dp4a", min(rows, 4), 8), ("tensor_core", 16, 64)]:
                fn = make_prim(
                    dict(
                        kind="swiglu",
                        rows=rows,
                        outputs=640,
                        inputs=2560,
                        dtype="float16",
                        dynamic=dynamic,
                    ),
                    dict(algorithm=algorithm, block_m=bm, block_n=bn, block_k=128),
                )
                buffers = {b.name: b for b in fn.buffer_map.values()}
                scalars = [p for p in fn.params if str(p.dtype) == "int32"]
                self.assertEqual(len(scalars), int(dynamic))
                if dynamic:
                    self.assertTrue(buffers["Output"].shape[0].same_as(scalars[0]))
                words = int(buffers["Gate"].shape[1])
                self.assertEqual(words * (4 if algorithm == "dp4a" else 1), 2560)
                self.assertEqual(str(buffers["Output"].dtype), "float16")

    def test_fused_route_scans_live_assignments_and_bounds_all_tiles(self):
        from kernels.model.moe import route_metadata

        for bm in (8, 16):
            fn = explicit_rows(route_metadata.__wrapped__(128, block_m=bm, dynamic_rows=True))
            (rows,) = [p for p in fn.params if str(p.dtype) == "int32"]
            buffers = {b.name: b for b in fn.buffer_map.values()}
            self.assertTrue(buffers["RelativeSlot"].shape[0].same_as(rows))
            self.assertEqual(int(buffers["TileExpert"].shape[0]), (1280 + bm - 1) // bm + 512)
            loops = []

            def visit(node):
                if isinstance(node, tir.For) and Analyzer().can_prove_equal(node.extent, rows * 10):
                    loops.append(node)

            tir.stmt_functor.post_order_visit(fn.body, visit)
            self.assertEqual(len(loops), 1)

    def test_lowered_q4_retains_explicit_live_rows(self):
        fn = explicit_rows(factory("q4-gemv-r4-n6144-k2560-d0", 4))
        (rows,) = [p for p in fn.params if str(p.dtype) == "int32"]
        buffers = {b.name: b for b in fn.buffer_map.values()}
        for name in ("A", "AS", "Out"):
            self.assertTrue(buffers[name].shape[0].same_as(rows))
        self.assertEqual(int(buffers["W"].shape[1]), 640)

    def test_lowered_head_keeps_fp32_logits(self):
        fn = explicit_rows(factory("q4-gemv-r4-n248320-k2560-float32-d0", 4))
        buffers = {b.name: b for b in fn.buffer_map.values()}
        self.assertEqual(str(buffers["Out"].dtype), "float32")

    def test_direct_bf16_rows_do_not_change_private_split_layout(self):
        fn = explicit_rows(factory("bf16-r4-n320-g4-d0", 4))
        (rows,) = [p for p in fn.params if str(p.dtype) == "int32"]
        buffers = {b.name: b for b in fn.buffer_map.values()}
        self.assertTrue(buffers["A"].shape[0].same_as(rows))
        self.assertTrue(buffers["Output"].shape[1].same_as(rows))
        self.assertEqual(int(buffers["Output"].shape[0]), 4)

    def test_large_route_histograms_keep_live_segment_strides_and_assignment_rows(self):
        for label in ("histogram-local-1024", "histogram-finish-1024"):
            fn = explicit_rows(factory(label, 1024))
            (rows,) = [p for p in fn.params if str(p.dtype) == "int32"]
            buffers = {b.name: b for b in fn.buffer_map.values()}
            self.assertTrue(buffers["IDs"].shape[0].same_as(rows))
            self.assertTrue(buffers["Slots"].shape[0].same_as(rows))
            partial = buffers.get("Partial", buffers.get("Offsets"))
            self.assertEqual(int(partial.shape[0]), 512)
            for live in [1, 512, 1024]:
                stride = tir.stmt_functor.substitute(
                    partial.shape[1], {rows: tir.IntImm("int32", live)}
                )
                self.assertEqual(int(Analyzer().simplify(stride)), (live * 10 + 1023) // 1024)
        self.assertIsNone(factory("histogram-offsets-1024", 1024))

    def test_affine_greedy_output_has_an_explicit_scalar_parameter(self):
        fn = explicit_rows(factory("greedy-merge-4-248320", 4))
        (scalar,) = [p for p in fn.params if str(p.dtype) == "int32"]
        self.assertEqual(scalar.name, "m")
        self.assertTrue(explicit_rows(fn).same_as(fn))

    def test_expert_rotation_uses_assignment_rows_without_implicit_affine_inference(self):
        fn = explicit_rows(factory("rotate-a8-ffn-8", 8))
        (scalar,) = [p for p in fn.params if str(p.dtype) == "int32"]
        for buffer in fn.buffer_map.values():
            if buffer.name in ("X", "Q", "S"):
                self.assertTrue(buffer.shape[0].same_as(scalar))

    def test_unknown_shared_geometry_is_not_silently_reused(self):
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            factory("unknown-4", 4)

    def test_single_output_gate_keeps_the_dynamic_row_argument(self):
        for capacity in (8, 128):
            with self.subTest(capacity=capacity):
                fn = explicit_rows(factory("bf16-gate-8-2560", capacity))
                (rows,) = [p for p in fn.params if str(p.dtype) == "int32"]
                buffers = {b.name: b for b in fn.buffer_map.values()}
                self.assertTrue(buffers["A"].shape[0].same_as(rows))
                self.assertTrue(buffers["Output"].shape[0].same_as(rows))
                self.assertEqual(int(buffers["Weight"].shape[0]), 1)
                self.assertEqual(str(buffers["Weight"].dtype), "bfloat16")

    def test_hc_down_cache_and_streaming_keep_symbolic_partial_rows(self):
        for capacity in (8, 128):
            with self.subTest(capacity=capacity):
                fn = explicit_rows(factory("hc-down-partial-8-4", capacity))
                (rows,) = [p for p in fn.params if str(p.dtype) == "int32"]
                buffers = {b.name: b for b in fn.buffer_map.values()}
                self.assertTrue(buffers["A"].shape[0].same_as(rows))
                self.assertTrue(buffers["Partials"].shape[1].same_as(rows))
                self.assertEqual(int(buffers["Partials"].shape[0]), 4)
                self.assertEqual(str(buffers["Partials"].dtype), "float32")

    def test_small_int8_projection_keeps_dynamic_rows_and_packed_operand_width(self):
        for capacity in (8, 128):
            with self.subTest(capacity=capacity):
                fn = explicit_rows(factory("dense-small-a8-8-48-2560-float16", capacity))
                (rows,) = [p for p in fn.params if str(p.dtype) == "int32"]
                buffers = {b.name: b for b in fn.buffer_map.values()}
                for name in ("A", "Output", "TokenScale"):
                    self.assertTrue(buffers[name].shape[0].same_as(rows))
                self.assertEqual(int(buffers["A"].shape[1]), 640)
                self.assertEqual(str(buffers["A"].dtype), "int32")
                self.assertEqual(int(buffers["Weight"].shape[0]), 48)


if __name__ == "__main__":
    unittest.main()
