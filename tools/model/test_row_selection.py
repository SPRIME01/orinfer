"""Measured selections cover tails without retaining losing implementations."""

import unittest
from tools.model.row_selection import ranges
from tools.model.row_selection import choices, samples, split_fixed


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


class DynamicProjectionChoicesTests(unittest.TestCase):
    def test_bounded_choices_cover_only_non_aot_rows_with_measured_configs(self):
        fixed = [1, 2, 3, 4, 8, 16, 32]
        table = dict(
            configs=[{"tile": "large"}, {"tile": "small"}],
            default="0",
            selections=[dict(min_rows=1, max_rows=23, implementation="1")],
        )
        selected = {}
        for suffix, config in choices(table, 64, fixed, bounded=True):
            low, high = map(int, suffix.removeprefix("/rows/").split("-"))
            for row in range(low, high + 1):
                self.assertNotIn(row, selected)
                selected[row] = config
        self.assertEqual(set(selected), set(range(1, 65)) - set(fixed))
        self.assertEqual(selected[23], {"tile": "small"})
        self.assertEqual(selected[24], {"tile": "large"})

    def test_fixed_specializations_are_not_overlapped_by_dynamic_choices(self):
        fixed = [1, 2, 3, 4, 8, 16, 32]
        table = dict(
            configs=[{"tile": "large"}, {"tile": "small"}],
            default="0",
            selections=[dict(min_rows=1, max_rows=64, implementation="1")],
        )
        published = list(choices(table, 128, fixed))
        self.assertEqual(published[0], ("", {"tile": "large"}))
        selected = {}
        for suffix, config in published[1:]:
            low, high = map(int, suffix.removeprefix("/rows/").split("-"))
            for row in range(low, high + 1):
                self.assertNotIn(row, fixed)
                self.assertNotIn(row, selected)
                selected[row] = config
        for row in set(range(1, 129)) - set(fixed):
            self.assertEqual(
                selected.get(row, published[0][1]), {"tile": "small" if row <= 64 else "large"}
            )

    def test_unused_default_is_removed_when_winner_only_covers_aot_points(self):
        table = dict(
            configs=[{"tile": "unused"}, {"tile": "dynamic"}],
            default="0",
            selections=[dict(min_rows=2, max_rows=8, implementation="1")],
        )
        self.assertEqual(list(choices(table, 8, [1])), [("", {"tile": "dynamic"})])
        self.assertEqual(list(choices(table, 8, range(1, 9))), [])

    def test_samples_cover_small_capacities_and_large_tile_boundaries(self):
        self.assertEqual(samples(1), [1])
        for capacity in [8, 128, 4096]:
            rows = samples(capacity)
            self.assertEqual(rows, sorted(set(rows)))
            self.assertEqual(rows[0], 1)
            self.assertEqual(rows[-1], capacity)
            for n in [512, 2048, 4096]:
                for delta in [-1, 0, 1]:
                    if 1 <= n + delta <= capacity:
                        self.assertIn(n + delta, rows)
        self.assertEqual(split_fixed(1, 8, [1, 4, 4, 8]), [(2, 3), (5, 7)])
