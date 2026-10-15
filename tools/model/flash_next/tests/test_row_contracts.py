"""Row counts and causal boundaries survive local fusion changes."""

import unittest

from tools.model.flash_next.draft_rows import causal_bindings, primitive_row_multiplier


class SharedRowsTests(unittest.TestCase):
    def test_assignment_and_stream_rows_are_distinct_from_token_rows(self):
        self.assertEqual(primitive_row_multiplier("rotate-a8-ffn-8", 80), 10)
        self.assertEqual(primitive_row_multiplier("dense-32-2560-2560", 32), 4)
        self.assertEqual(primitive_row_multiplier("dense-8-6144-2560", 8), 1)

    def test_causal_matching_tolerates_an_extra_projection_before_qsa(self):
        class Interfaces:
            @staticmethod
            def get(kernel):
                return kernel["abi"], None

        original = {}
        for width, sequence in [
            (8, ["projection", "prepare", "store", "index", "merge"]),
            (128, ["projection", "value", "prepare", "store", "index", "merge"]),
        ]:
            for i, abi in enumerate(sequence):
                name = f"mtp_warm_m{width}/layer48/k{i}"
                original[name] = dict(name=name, abi=[abi])
        found = causal_bindings(original, Interfaces(), 128, [1, 2, 3, 4])
        self.assertEqual([k["abi"] for k in found], [["prepare"], ["store"], ["index"], ["merge"]])
        del original["mtp_warm_m128/layer48/k4"]
        with self.assertRaisesRegex(ValueError, "Missing chronological"):
            causal_bindings(original, Interfaces(), 128, [1, 2, 3, 4])
