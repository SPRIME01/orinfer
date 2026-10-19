"""Precision policy for shared dense Qwen encoders and tied embeddings."""

import unittest
from tools.quantization.qwen_e8p import policy, text_configuration


class DensePolicyTests(unittest.TestCase):
    def test_medium_dense_preserves_untied_output_head(self):
        config = dict(
            model_type="qwen3_5",
            tie_word_embeddings=False,
            text_config=dict(hidden_size=4096, intermediate_size=12288, num_hidden_layers=32),
        )
        text = text_configuration(config)
        self.assertIs(text, config["text_config"])
        self.assertIs(text["tie_word_embeddings"], False)
        for name in ("model.language_model.embed_tokens.weight", "lm_head.weight"):
            self.assertEqual(policy(name, dict(dtype="BF16", shape=[248320, 4096])), "e8p-q4")
        text["tie_word_embeddings"] = True
        self.assertIs(text_configuration(config)["tie_word_embeddings"], True)
        text["tie_word_embeddings"] = 0
        with self.assertRaises(ValueError):
            text_configuration(config)

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
