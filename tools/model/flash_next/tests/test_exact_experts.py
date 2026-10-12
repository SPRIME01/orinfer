"""Lossless publication and exported expert geometry must fail on corruption."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from tools.model.flash_next.optimize_experts import live_rows, repack_weights, row_tile


class ExactExpertPackageTests(unittest.TestCase):
    def test_repack_preserves_source_shared_inode_and_all_codes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "source", root / "target"
            for model in (source, target):
                (model / "cache/weights").mkdir(parents=True)
            name = "W_blk.0.ffn_gate_up_exps.weight_packed"
            data = np.random.default_rng(20261002).integers(0, 65536, (2, 3, 17, 16), np.uint16)
            scale = np.ones((2, 17), np.float16)
            values = {name: data, "scale": scale}
            metadata = {f"orin.layout.{n}": "native_contiguous" for n in values}
            filename = "original.safetensors"
            path = source / "cache/weights" / filename
            save_file(values, path, metadata=metadata)
            (target / "cache/weights" / filename).hardlink_to(path)
            original = path.read_bytes()
            buffers = [
                dict(
                    name=n,
                    dtype="u16" if n == name else "f16",
                    shape=list(v.shape),
                    layout="native_contiguous",
                    data=dict(tensor=n, sha256=hashlib.sha256(v.tobytes()).hexdigest()),
                )
                for n, v in values.items()
            ]
            descriptor = dict(metadata=dict(buffers=buffers))
            package = dict(buffer_contracts=json.loads(json.dumps(buffers)))
            index = dict(weight_map={n: filename for n in values})
            self.assertEqual(repack_weights(source, target, descriptor, package, index), 1)
            self.assertEqual(path.read_bytes(), original)
            self.assertFalse((target / "cache/weights" / filename).exists())
            with safe_open(
                target / "cache/weights" / index["weight_map"][name], framework="np"
            ) as weights:
                transformed = weights.get_tensor(name)
                restored = (
                    transformed.reshape(2, 3, 2, 17, 8).transpose(0, 1, 3, 2, 4).reshape(data.shape)
                )
                np.testing.assert_array_equal(restored, data)
                np.testing.assert_array_equal(weights.get_tensor("scale"), scale)
            self.assertEqual(buffers[0]["shape"], [2, 6, 17, 8])
            self.assertEqual(package["buffer_contracts"][0]["shape"], buffers[0]["shape"])
            self.assertEqual(
                hashlib.sha256(transformed.tobytes()).hexdigest(), buffers[0]["data"]["sha256"]
            )
            buffers[0]["shape"] = list(data.shape)
            buffers[0]["data"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "identity"):
                repack_weights(
                    source,
                    target,
                    descriptor,
                    package,
                    dict(weight_map={n: filename for n in values}),
                )

    def test_program_and_row_stride_discovery_fail_closed(self):
        for program, rows in [
            ("decode", 1),
            ("prefill_m512", 512),
            ("verify_m3", 3),
            ("flash_dynamic_m4", 4),
            ("mtp_warm_m128", 128),
        ]:
            self.assertEqual(live_rows(program), rows)
        self.assertEqual(row_tile("meta[3] = min(16, (count - first));"), 16)
        for source in ("count = 8;", "min(8, (count)); min(16, (other));"):
            with self.assertRaises(ValueError):
                row_tile(source)
        with self.assertRaises(ValueError):
            live_rows("unknown_m4")


if __name__ == "__main__":
    unittest.main()
