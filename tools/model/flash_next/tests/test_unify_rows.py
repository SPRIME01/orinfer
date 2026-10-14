"""Row counts and causal boundaries survive local fusion changes."""

import unittest

from tools.model.flash_next.draft_rows import causal_bindings, primitive_row_multiplier
from tools.model.flash_next.unify_rows import obsolete_computation


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

    def test_pruning_keeps_capture_controls_and_commit_factors(self):
        for name in [
            "verify_m4/layer3/k12",
            "mtp_head_m128/body/k2",
            "mtp_warm_m128/layer48/k3",
            "flash_draft_m8/post/k2",
        ]:
            self.assertTrue(obsolete_computation(name))
        for name in [
            "verify_m4/begin/k0",
            "verify_m4/head/k8",
            "verify_m4/end/k0",
            "restore_m4/layer0/k0",
            "draft_rows/capture_m128",
            "draft_rows/causal_m128/k3",
            "mtp_warm_m128/begin/k1",
        ]:
            self.assertFalse(obsolete_computation(name))
