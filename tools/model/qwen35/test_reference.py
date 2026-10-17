"""CPU checks for the independent official BF16 reference's Torch dispatch."""

import unittest

import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from tools.eval.reference_bf16 import configure_torch_reference


class TorchReferenceTests(unittest.TestCase):
    def test_masked_hybrid_forward_matches_independent_histories(self):
        configure_torch_reference()
        torch.set_num_threads(2)
        torch.manual_seed(20261002)
        config = Qwen3_5TextConfig(
            vocab_size=97,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            linear_num_value_heads=4,
            linear_num_key_heads=2,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            linear_conv_kernel_dim=4,
            layer_types=["linear_attention", "full_attention"],
            max_position_embeddings=128,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
                "mrope_section": [3, 3, 2],
            },
        )
        config._attn_implementation = "sdpa"
        model = modeling.Qwen3_5TextModel(config).eval()
        histories = [[1, 8, 3, 7, 2], [6, 4, 9]]
        ids = torch.tensor([histories[0], histories[1] + [0, 0]])
        mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
        with torch.inference_mode():
            batched = model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
            self.assertTrue(bool(torch.isfinite(batched).all()))
            for row, history in enumerate(histories):
                single = model(input_ids=torch.tensor([history]), use_cache=False).last_hidden_state
                torch.testing.assert_close(
                    batched[row, : len(history)], single[0], atol=1e-5, rtol=1e-5
                )
