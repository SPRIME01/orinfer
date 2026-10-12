"""CPU compiler checks for the affine-shape host ABI regression."""

import unittest

from kernels.model.rows import explicit_rows
from tools.model.flash_next.dynamic_batch import factory


class DynamicRowsTests(unittest.TestCase):
    def test_large_route_histograms_keep_capacity_strides_and_live_assignment_rows(self):
        for label in ("histogram-local-1024", "histogram-finish-1024"):
            fn = explicit_rows(factory(label, 1024))
            (rows,) = [p for p in fn.params if str(p.dtype) == "int32"]
            buffers = {b.name: b for b in fn.buffer_map.values()}
            self.assertTrue(buffers["IDs"].shape[0].same_as(rows))
            self.assertTrue(buffers["Slots"].shape[0].same_as(rows))
            partial = buffers.get("Partial", buffers.get("Offsets"))
            self.assertEqual([int(n) for n in partial.shape], [512, 10])
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
