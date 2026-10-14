"""Typed local fusions must consume only their declared adjacent operations."""

import unittest
from tools.model.flash_next.unify_decode import share_dynamic_fallbacks
from tools.model.flash_next.decode_policy import (
    decode_shapes,
    selection_name,
    q4_configs,
    w8_configs,
)


class DecodeAlignmentTests(unittest.TestCase):
    def test_complete_row_table_reuses_its_dynamic_cubin_for_the_canonical_slot(self):
        kernels = [dict(name="slot", module="unused")]
        for shape in decode_shapes():
            for first, last in shape.ranges():
                kernels.append(dict(name=selection_name("slot", first, last), module=shape.key))
        contracts = [dict(name=k["name"], capacity=128) for k in kernels]
        self.assertEqual(share_dynamic_fallbacks(kernels, contracts), 1)
        self.assertEqual(kernels[0], dict(name="slot", module="dynamic"))
        self.assertEqual(contracts[0], dict(name="slot", capacity=128))
        kernels.pop()
        kernels[0]["module"] = "untouched"
        self.assertEqual(share_dynamic_fallbacks(kernels, contracts), 0)
        self.assertEqual(kernels[0]["module"], "untouched")

    def test_fused_range_rejects_an_overlapping_medium_tile(self):
        kernels = [
            dict(name="slot"),
            dict(name="slot/rows/1-128"),
            dict(name="slot/rows/9-16"),
        ]
        contracts = [dict(name=k["name"]) for k in kernels]
        with self.assertRaisesRegex(ValueError, "Overlapping.*slot"):
            share_dynamic_fallbacks(kernels, contracts)

    def test_specializations_and_symbolic_fallback_cover_each_live_row_once(self):
        shapes = decode_shapes()
        self.assertEqual([s.rows for s in shapes if not s.dynamic], [1, 2, 3, 4, 8, 16, 32])
        self.assertEqual(sum(s.dynamic for s in shapes), 1)
        covered = []
        for shape in shapes:
            for first, last in shape.ranges():
                covered.extend(range(first, last + 1))
                self.assertTrue(
                    selection_name("slot", first, last).endswith(f"/rows/{first}-{last}")
                )
        self.assertEqual(sorted(covered), list(range(1, 129)))
        for bounds in [(0, 8), (8, 4), (33, 129)]:
            with self.assertRaises(ValueError):
                selection_name("slot", *bounds)

    def test_one_q4_search_space_preserves_small_and_large_algorithms(self):
        for rows in [1, 3, 8, 32, 128]:
            configs = q4_configs(rows)
            self.assertEqual({c["algorithm"] for c in configs}, {"dp4a", "tensor_core"})
            self.assertTrue(all(c["block_m"] <= 8 for c in configs if c["algorithm"] == "dp4a"))

    def test_w8_search_includes_dp4a_for_large_matrices(self):
        for rows in [1, 4, 8, 32, 128]:
            for outputs in [48, 640, 2560, 6144]:
                candidates = w8_configs(rows, outputs)
                self.assertEqual(
                    {c.get("algorithm", "tensor_core") for c in candidates}, {"dp4a", "tensor_core"}
                )
                self.assertEqual(
                    {c["block_n"] for c in candidates if c.get("algorithm") == "dp4a"}, {4, 8, 16}
                )
