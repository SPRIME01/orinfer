"""Measured selections cover tails without retaining losing implementations."""

import unittest
from tools.model.flash_next.prefill_tuning import ranges


class PrefillTuningTests(unittest.TestCase):
    def test_one_winner_has_no_other_runtime_implementation(self):
        result = ranges([dict(rows=n, times=dict(a=2.0, b=1.0)) for n in [1, 17, 257, 512]], 512)
        self.assertEqual(result["used"], ["b"])
        self.assertEqual(result["default"], "b")
        self.assertEqual(result["selections"], [])

    def test_boundaries_follow_measurements_instead_of_fixed_tiers(self):
        result = ranges(
            [
                dict(rows=1, times=dict(a=1.0, b=3.0)),
                dict(rows=37, times=dict(a=3.0, b=1.0)),
                dict(rows=512, times=dict(a=4.0, b=1.0)),
            ],
            512,
        )
        self.assertEqual(result["used"], ["a", "b"])
        choices = result["selections"]
        self.assertTrue(choices)
        self.assertFalse(any(c["max_rows"] in [127, 1023] for c in choices))

    def test_aligned_candidate_cannot_escape_its_legal_rows(self):
        result = ranges(
            [
                dict(rows=1, times=dict(general=2.0)),
                dict(rows=128, times=dict(general=3.0, aligned=1.0)),
                dict(rows=512, times=dict(general=3.0, aligned=1.0)),
            ],
            512,
        )
        self.assertEqual(result["default"], "general")
        self.assertEqual(
            result["selections"],
            [dict(min_rows=128, max_rows=512, multiple_of=128, implementation="aligned")],
        )
