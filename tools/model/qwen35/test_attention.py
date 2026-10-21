"""One measured geometry covers owner counts without model-specific cutoffs."""

import unittest
from tools.model.qwen35.attention import candidates


class OwnerAttentionPolicyTests(unittest.TestCase):
    def test_geometry_is_derived_from_heads_and_reference_is_preserved(self):
        reference = dict(block_m=32, block_n=64, head_group=1, max_splits=8, decoded_kv=True)
        for query, kv in ((8, 2), (16, 4), (24, 4)):
            result = candidates(query, kv, reference)
            self.assertEqual(result[0]["block_m"], 32)
            self.assertEqual(result[0]["head_group"], 1)
            self.assertTrue(all(c["head_group"] == query // kv for c in result[1:]))
            self.assertTrue(all("decoded_kv" not in c for c in result))
        self.assertTrue(reference["decoded_kv"])
        with self.assertRaises(ValueError):
            candidates(7, 2, reference)
