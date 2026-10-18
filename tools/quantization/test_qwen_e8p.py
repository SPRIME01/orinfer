"""Precision policy for shared dense Qwen encoders and tied embeddings."""

import unittest
from tools.quantization.qwen_e8p import policy


class DensePolicyTests(unittest.TestCase):
    def test_main_projections_use_e8p_and_sensitive_operands_keep_w8(self):
        for name, shape, expected in (
            ("model.language_model.layers.0.mlp.down_proj.weight", [2048, 6144], "e8p-q4"),
            (
                "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
                [6144, 2048],
                "e8p-q4",
            ),
            ("model.language_model.embed_tokens.weight", [248320, 2048], "e8p-q4"),
            ("model.language_model.layers.0.linear_attn.out_proj.weight", [2048, 2048], "w8"),
            ("mtp.layers.0.mlp.up_proj.weight", [6144, 2048], "w8"),
            ("model.language_model.layers.0.linear_attn.in_proj_a.weight", [16, 2048], "original"),
            ("model.visual.pos_embed.weight", [2304, 1024], "original"),
            ("model.visual.blocks.0.attn.qkv.weight", [3072, 1024], "e8p-q4"),
        ):
            with self.subTest(name=name):
                self.assertEqual(policy(name, dict(dtype="BF16", shape=shape)), expected)


if __name__ == "__main__":
    unittest.main()
