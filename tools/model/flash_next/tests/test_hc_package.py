"""Protect HC head/main geometry when rebuilding an otherwise identical ABI."""

import unittest

from tools.model.flash_next.optimize_hc import up_rank
from tools.model.flash_next.mixed_hc import reduction_groups


class HcGeometryTests(unittest.TestCase):
    def test_joined_down_keeps_the_four_stream_accumulation_order(self):
        self.assertEqual(reduction_groups(320, 10240), 4)
        self.assertEqual(reduction_groups(10240, 320), 1)
        with self.assertRaisesRegex(ValueError, "projection"):
            reduction_groups(320, 2560)

    def buffers(self, rank):
        return {
            "w": {"dtype": "bf16", "shape": [10240, rank], "data": {"tensor": "weight"}},
            "a": {"dtype": "f16", "shape": [8, rank]},
            "n": {"dtype": "f16", "shape": [8, 4, 2560]},
            "o": {"dtype": "f16", "shape": [8, 2560]},
        }

    def roles(self):
        return {
            role: {"kind": "buffer", "name": name}
            for role, name in (("W", "w"), ("A", "a"), ("Normed", "n"), ("Out", "o"))
        }

    def test_shared_launch_geometry_does_not_determine_rank(self):
        self.assertEqual(up_rank(self.buffers(256), self.roles(), 3), 256)
        self.assertEqual(up_rank(self.buffers(320), self.roles(), 3), 320)

    def test_input_slice_cannot_extend_past_its_arena(self):
        bindings = self.roles()
        bindings["A"] = {"kind": "buffer_slice", "name": "a", "offset": 7 * 256 * 2}
        with self.assertRaisesRegex(ValueError, "A binding"):
            up_rank(self.buffers(256), bindings, 3)

    def test_reject_quantized_or_unregistered_weight_shape(self):
        for field, value in (("dtype", "i8"), ("shape", [10240, 384])):
            buffers = self.buffers(256)
            buffers["w"][field] = value
            with self.assertRaisesRegex(ValueError, "weight contract"):
                up_rank(buffers, self.roles(), 3)


if __name__ == "__main__":
    unittest.main()
