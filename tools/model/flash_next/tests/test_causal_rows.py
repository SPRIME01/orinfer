"""Dynamic state strides use physical history/ring geometry, never a row tier."""

import unittest
from tools.model.flash_next.causal_rows import history_geometry, ring_geometry


class CausalGeometryTests(unittest.TestCase):
    def test_flattened_conv_and_ple_keep_the_chronological_axis(self):
        for shape in [[30720], [1, 3, 10240], [3, 10240]]:
            self.assertEqual(history_geometry(dict(shape=shape)), (3, 10240))
        self.assertEqual(history_geometry(dict(shape=[92160])), (9, 10240))
        with self.assertRaises(ValueError):
            history_geometry(dict(shape=[30721]))

    def test_hidden_ring_wrap_uses_the_published_capacity(self):
        for rows in [512, 4096]:
            self.assertEqual(ring_geometry(dict(shape=[rows, 10240])), (rows, 10240))
        with self.assertRaises(ValueError):
            ring_geometry(dict(shape=[4096 * 10240]))
