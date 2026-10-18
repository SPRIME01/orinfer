"""Freeze HF added-token overrides into the standalone deployment tokenizer."""

import json
from pathlib import Path


def prepare_tokenizer(source, destination):
    """Match HF token IDs without requiring tokenizer_config parsing online."""
    from tokenizers import AddedToken, Tokenizer

    source, destination = Path(source), Path(destination)
    raw = (source / "tokenizer.json").read_bytes()
    tokenizer = Tokenizer.from_str(raw.decode())
    config = source / "tokenizer_config.json"
    overrides = (
        json.loads(config.read_text()).get("added_tokens_decoder", {}) if config.exists() else {}
    )
    original = tokenizer.to_str()
    for key, values in sorted(overrides.items(), key=lambda item: int(item[0])):
        expected = int(key)
        token = AddedToken(**values)
        existing = tokenizer.token_to_id(token.content)
        if existing is not None and existing != expected:
            raise ValueError(f"Tokenizer ID conflict for {token.content}: {existing} != {expected}")
        if token.special:
            tokenizer.add_special_tokens([token])
        else:
            tokenizer.add_tokens([token])
        if tokenizer.token_to_id(token.content) != expected:
            raise ValueError(f"Tokenizer override would renumber {token.content}")
    # Preserve source bytes when its serialized tokenizer is already complete.
    target = tokenizer.to_str()
    destination.write_bytes(raw if target == original else (target + "\n").encode())
