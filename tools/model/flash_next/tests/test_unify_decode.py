"""Typed local fusions must consume only their declared adjacent operations."""

import unittest
from tools.model.flash_next.unify_decode import align_sections


class DecodeAlignmentTests(unittest.TestCase):
    def test_hc_and_shared_ffn_keep_unfused_order(self):
        small = [
            {"roles": {x: None for x in r}}
            for r in [
                ["A", "Hi", "Lo", "Normed", "Out"],
                ["A", "Gate", "Up", "GateScale", "UpScale", "TokenScale", "Output"],
                ["Block", "Residual", "Output"],
            ]
        ]
        large = [
            {"roles": {x: None for x in r}}
            for r in [
                ["A", "C", "W"],
                ["Mixed", "Normed", "Up"],
                ["A", "Weight", "WeightScale", "Output"],
                ["A", "Weight", "WeightScale", "Output"],
                ["Gate", "Up", "Output"],
                ["Block", "Residual", "Output"],
            ]
        ]
        pairs, skips = align_sections(small, large, lambda k: k["roles"])
        self.assertEqual([pair[1] for pair in pairs], [large[0], large[2], large[5]])
        self.assertEqual(skips, [large[1], large[3], large[4]])
        large[1]["roles"] = {"Other": None}
        with self.assertRaisesRegex(ValueError, "adjacent mix"):
            align_sections(small, large, lambda k: k["roles"])

    def test_extra_canonical_operation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unmatched"):
            align_sections([{}], [{}, {}], lambda k: {})
