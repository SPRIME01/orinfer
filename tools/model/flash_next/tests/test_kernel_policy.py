"""Expert tile contracts shared by initial builds and execution upgrades."""

import unittest

from tools.model.flash_next.kernel_policy import expert_tile_config, expert_projection_config


class ExpertTilePolicyTests(unittest.TestCase):
    def test_single_token_uses_register_mma_and_exact_sign_table(self):
        config = expert_tile_config(1)
        self.assertEqual(config["block_m"], 8)
        self.assertEqual(config["block_n"], 128)
        self.assertTrue(config["shortbook"])
        self.assertTrue(config["byte_permute"])
        self.assertTrue(config["selectorbook"])

    def test_prefill_workspaces_bound_all_routing_distributions(self):
        for rows in range(1, 4097):
            config = expert_tile_config(rows)
            tiles = (rows * 10 + config["block_m"] - 1) // config["block_m"] + 512
            self.assertLessEqual(tiles, 1792)

    def test_short_prefill_keeps_integer_mma_and_smaller_padding(self):
        config = expert_tile_config(512)
        self.assertEqual(config["block_m"], 16)
        self.assertEqual(config["num_stages"], 1)
        self.assertTrue(config["selectorbook"])

    def test_invalid_capacity_is_rejected(self):
        for rows in (0, 4097, True, 1.5):
            with self.assertRaises(ValueError):
                expert_tile_config(rows)

    def test_decode_keeps_row_tiles_and_selects_integer_split_k_by_width(self):
        for rows in (1, 2, 4, 8):
            gate = expert_projection_config(rows, 2560)
            down = expert_projection_config(rows, 640)
            self.assertEqual((gate["block_n"], gate["split_k"]), (32, 2))
            self.assertEqual((down["block_n"], down["split_k"]), (64, 1))
            for config in (gate, down):
                self.assertEqual(config["block_m"], 8)
                self.assertTrue(config["warp_decode"])
                self.assertTrue(config["selectorbook"])
                self.assertEqual(config["num_stages"], 2)

    def test_xor_parity_is_limited_to_checked_decode_bases(self):
        for rows in (1, 3, 8):
            self.assertTrue(expert_projection_config(rows, 2560, parity_xor=True)["parity_xor"])
            self.assertNotIn("parity_xor", expert_projection_config(rows, 2560))
        self.assertEqual(
            expert_projection_config(512, 2560, parity_xor=True), expert_tile_config(512)
        )
        with self.assertRaises(ValueError):
            expert_projection_config(1, 2560, parity_xor=1)

    def test_prefill_keeps_existing_projection_policy(self):
        for rows in (16, 32, 128, 256, 512, 2048, 4096):
            self.assertEqual(expert_projection_config(rows, 2560), expert_tile_config(rows))
        for inputs in (0, True, 129, 640.0):
            with self.assertRaises(ValueError):
                expert_projection_config(1, inputs)


if __name__ == "__main__":
    unittest.main()
