"""Standalone runtime encoding must match HF's configured added tokens."""

import json
from pathlib import Path
import tempfile
import unittest

from tokenizers import Tokenizer, models, pre_tokenizers

from tools.model.frontend import prepare_tokenizer


class FrontendTests(unittest.TestCase):
    def test_configured_tokens_are_recognized_without_changing_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer = Tokenizer(models.WordLevel({"?": 0, "<think>": 1, "</think>": 2}, "?"))
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            tokenizer.save(str(root / "tokenizer.json"))
            original = (root / "tokenizer.json").read_bytes()
            (root / "tokenizer_config.json").write_text(
                json.dumps(
                    dict(
                        added_tokens_decoder={
                            str(i): dict(content=s, normalized=False, special=False)
                            for i, s in ((1, "<think>"), (2, "</think>"))
                        }
                    )
                )
            )
            self.assertNotEqual(tokenizer.encode("<think></think>").ids, [1, 2])
            prepare_tokenizer(root, root / "prepared.json")
            prepared = Tokenizer.from_file(str(root / "prepared.json"))
            self.assertEqual(prepared.encode("<think></think>").ids, [1, 2])
            self.assertEqual(prepared.decode([1, 2], skip_special_tokens=True), "<think> </think>")
            self.assertEqual(prepared.get_vocab(), tokenizer.get_vocab())
            self.assertEqual((root / "tokenizer.json").read_bytes(), original)
            prepare_tokenizer(root, root / "prepared-again.json")
            self.assertEqual(
                (root / "prepared.json").read_bytes(), (root / "prepared-again.json").read_bytes()
            )
            (root / "tokenizer_config.json").write_text(
                json.dumps(dict(added_tokens_decoder={"5": dict(content="new-token")}))
            )
            with self.assertRaises(ValueError):
                prepare_tokenizer(root, root / "bad.json")

    def test_complete_tokenizer_is_copied_verbatim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer = Tokenizer(models.WordLevel({"?": 0}, "?"))
            tokenizer.save(str(root / "tokenizer.json"))
            prepare_tokenizer(root, root / "prepared.json")
            self.assertEqual(
                (root / "tokenizer.json").read_bytes(), (root / "prepared.json").read_bytes()
            )
