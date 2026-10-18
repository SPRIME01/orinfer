"""CPU contracts for native Qwen Q4A8 construction and lossless shard adoption."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from safetensors.numpy import save_file
from tools.model.qwen35.build import Builder, CAPACITY, configuration
from tools.model.qwen35.storage import Weights


def config():
    text = dict(
        hidden_size=5120,
        intermediate_size=17408,
        num_hidden_layers=64,
        vocab_size=248320,
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        num_attention_heads=24,
        num_key_value_heads=4,
        head_dim=256,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        tie_word_embeddings=False,
        max_position_embeddings=262144,
        layer_types=["full_attention" if i % 4 == 3 else "linear_attention" for i in range(64)],
        rope_parameters=dict(
            rope_type="default",
            rope_theta=10000000,
            partial_rotary_factor=0.25,
            mrope_section=[11, 11, 10],
            mrope_interleaved=True,
        ),
    )
    return dict(
        model_type="qwen3_5",
        text_config=text,
        quantization_config=dict(
            quant_method="orinfer_q4_int8",
            version=1,
            rotation="signed-block128",
            scale="fp16-output-row",
            compute_dtype="int8_quality",
        ),
    )


class BuildTests(unittest.TestCase):
    def test_native_shard_layout_is_explicit_without_requantization(self):
        from tools.model.publication import file_hash
        from tools.model.safetensors_source import local_header

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            shard = source / "weights.safetensors"
            values = {"W.int8": np.arange(-16, 16, dtype=np.int8).reshape(4, 8)}
            save_file(values, shard)
            original = file_hash(shard)
            (source / "model.safetensors.index.json").write_text(
                json.dumps(
                    dict(
                        metadata={"orinfer.sha256": {shard.name: original}},
                        weight_map={"W.int8": shard.name},
                    )
                )
            )
            weights = Weights(source, root / "model")
            weights.immutable("W.int8")
            np.testing.assert_array_equal(weights.array("W.int8"), values["W.int8"])
            _, header = local_header(root / "model/cache/weights" / shard.name)
            self.assertEqual(header["__metadata__"]["orin.layout.W.int8"], "native_contiguous")
            self.assertEqual(file_hash(shard), original)

    def test_small_dense_geometry_and_tied_head(self):
        candidate = config()
        t = candidate["text_config"]
        t.update(
            hidden_size=2048,
            intermediate_size=6144,
            num_hidden_layers=24,
            linear_num_value_heads=16,
            num_attention_heads=8,
            num_key_value_heads=2,
            tie_word_embeddings=True,
            layer_types=t["layer_types"][:24],
        )
        candidate["quantization_config"].update(
            quant_method="orinfer_e8p_q4_int8",
            coefficients=[4, 1],
            basis_tensor="orinfer.e8p_q4.basis",
            layout="u32-low-coarse-high-residual",
            embedding="e8p_q4",
            vision="e8p_q4",
        )
        self.assertEqual(configuration(candidate, 262144)["hidden_size"], 2048)
        t["tie_word_embeddings"] = False
        with self.assertRaises(ValueError):
            configuration(candidate, 262144)

    def test_tied_head_binds_the_shared_compressed_weight(self):
        builder = Builder.__new__(Builder)
        builder.weights = SimpleNamespace(immutable=lambda n: n, packed_k=128)
        builder.q4_codec, builder.book = "e8p_q4", "book"
        launches = []
        builder.emit = lambda name, key, build, bindings, rows, **options: launches.append(bindings)
        builder.projection(
            "head/project",
            "embedding",
            248320,
            2048,
            "HeadLogits",
            head=True,
            rows=1,
            dynamic=False,
        )
        self.assertEqual(
            launches,
            [
                dict(
                    A="HeadA8",
                    W="embedding.packed",
                    S="embedding.scale",
                    AS="HeadAS",
                    Out="HeadLogits",
                    Book="book",
                )
            ],
        )

    def test_prefetch_views_bind_the_owning_kv_and_exported_abi(self):
        first, merge = (lambda: None), (lambda: None)
        first.__name__, merge.__name__ = "attention", "merge"
        selection = dict(
            configs=[
                dict(block_m=64, block_n=64, head_group=1, max_splits=8, threads=256, stages=1)
            ],
            default="0",
            selections=[],
        )
        builder = Builder.__new__(Builder)
        builder.context, builder.lock = 262144, None
        builder.qh, builder.kvh = 24, 4
        builder.row_tuner = SimpleNamespace(attention=lambda _: selection)
        launches = []
        builder.emit = lambda name, key, build, bindings, rows: launches.append(
            (build(), bindings, rows)
        )
        with (
            patch.dict(
                "sys.modules",
                {
                    "kernels.model.attention_rows": SimpleNamespace(
                        causal_attention_int8=first, merge_attention_int8=merge
                    )
                },
            ),
            patch("tools.model.qwen35.build.primitive", side_effect=lambda fn, *a, **k: k),
        ):
            builder.attention(
                "draft_attention",
                q="query",
                gate="gate",
                position="pos",
                length="length",
                output="output",
                prefix="Draft",
                rows=7,
            )
        options, bindings, rows = launches[0]
        self.assertEqual((options["threads"], options["stages"], rows), (256, 1, 7))
        for name, expected in (
            ("k", "KPages"),
            ("v", "VPages"),
            ("ks", "KPagesScale"),
            ("vs", "VPagesScale"),
        ):
            self.assertEqual(bindings[f"live_{name}"], "Draft" + expected)
        self.assertEqual(bindings["Lengths"], "length")
        self.assertNotIn("stages", launches[1][0])
        self.assertNotIn("live_k", launches[1][1])

    def test_projection_bounds_measured_choices_and_preserves_exact_aot(self):
        builder = Builder.__new__(Builder)
        builder.weights = SimpleNamespace(immutable=lambda name: name, packed_k=128)
        builder.q4_codec = "e8p_q4"
        builder.book = "selectors"
        builder.lock = None
        builder.compile_projection = lambda *args, **kw: [("/rows/5-7", {"block_m": 8}, object())]
        launches = []
        builder.emit = lambda name, key, build, bindings, rows, **options: launches.append(
            (name, rows, options.get("dynamic", True), bindings)
        )
        for kind in ("q4", "w8"):
            launches.clear()
            builder.projection(
                "projection", "weight", 65, 256, "output", stride=68, offset=3, kind=kind
            )
            self.assertEqual(
                [(name, rows, dynamic) for name, rows, dynamic, _ in launches],
                [("projection", CAPACITY, True)]
                + [(f"projection/rows/{r}-{r}", r, False) for r in (1, 2, 3, 4, 8, 16, 32)]
                + [("projection/rows/5-7", 128, True)],
            )
            output = "Out" if kind == "q4" else "Output"
            for _, _, _, bindings in launches:
                self.assertEqual(
                    bindings[output], dict(kind="buffer_slice", name="output", offset=6)
                )

    def test_equivalent_projection_strides_share_the_compiled_identity(self):
        builder = Builder.__new__(Builder)
        builder.weights = SimpleNamespace(immutable=lambda name: name, packed_k=128)
        builder.q4_codec, builder.book = "e8p_q4", "selectors"
        keys = []
        builder.emit = lambda name, key, *args, **kwargs: keys.append(key)
        for stride in (None, 64, 68):
            builder.projection(
                "projection", "weight", 64, 256, "output", stride=stride, rows=1, dynamic=False
            )
        self.assertEqual(keys[0], keys[1])
        self.assertNotEqual(keys[0], keys[2])

    def test_bounded_tuning_is_shared_across_layer_weights_but_not_strides(self):
        builder = Builder.__new__(Builder)
        builder.weights = SimpleNamespace(immutable=lambda name: name, packed_k=128)
        builder.q4_codec, builder.book, builder.lock = "e8p_q4", "selectors", None
        builder.emit = lambda *args, **kwargs: None
        with patch.object(builder, "compile_projection", return_value=[]) as compile:
            for weight, stride in [("layer0", None), ("layer1", None), ("layer2", 68)]:
                builder.projection("projection", weight, 64, 256, "output", stride=stride)
            self.assertEqual(compile.call_count, 2)

    def test_two_stage_codec_has_an_explicit_contract(self):
        candidate = config()
        quant = candidate["quantization_config"]
        quant.update(
            quant_method="orinfer_e8p_q4_int8",
            coefficients=[4, 1],
            basis_tensor="orinfer.e8p_q4.basis",
            layout="u32-low-coarse-high-residual",
        )
        self.assertEqual(configuration(candidate, 8192)["hidden_size"], 5120)
        for key, value in [
            ("coefficients", [3, 1]),
            ("coefficients", [4.0, 1.0]),
            ("basis_tensor", "other"),
            ("layout", "high-coarse"),
        ]:
            saved = quant[key]
            quant[key] = value
            with self.assertRaises(ValueError):
                configuration(candidate, 8192)
            quant[key] = saved

    def test_compiler_rejects_wrong_math_codec_and_context(self):
        self.assertEqual(configuration(config(), 262144)["num_hidden_layers"], 64)
        for component, key, value in [
            ("text_config", "head_dim", 128),
            ("text_config", "rms_norm_eps", 1e-5),
            ("quantization_config", "rotation", "none"),
        ]:
            d = config()
            d[component][key] = value
            with self.assertRaises(ValueError):
                configuration(d, 262144)
        for context in [0, 262145]:
            with self.assertRaises(ValueError):
                configuration(config(), context)

    def test_e8p_repack_is_lossless_and_publishes_one_weight_representation(self):
        from tools.model.compact import write_shard
        from tools.model.publication import file_hash
        from tools.model.safetensors_source import local_header

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            raw = np.arange(3 * 64, dtype=np.uint32).reshape(3, 64) * 1234567
            values = {
                "W.packed": raw,
                "W.scale": np.array([0.1, 0.2, 0.3], np.float16),
                "Sensitive": np.array([0x8000, 0x7F80, 0x3F81, 0x0001], np.uint16),
            }
            tensors = []
            for name, value in values.items():
                path = root / (name + ".bin")
                value.tofile(path)
                tensors.append(
                    dict(
                        name=name,
                        dtype="U32"
                        if name.endswith("packed")
                        else "F16"
                        if name.endswith("scale")
                        else "BF16",
                        shape=list(value.shape),
                        segments=[(path, 0, value.nbytes)],
                    )
                )
            shard = source / "layer.safetensors"
            write_shard(shard, tensors, {"orin.layout.W.packed": "e8p-q4-row-u32"})
            source_sha = file_hash(shard)
            index = dict(
                metadata={"orinfer.sha256": {shard.name: source_sha}},
                weight_map={name: shard.name for name in values},
            )
            (source / "model.safetensors.index.json").write_text(json.dumps(index))
            for packed_k in (64, 128):
                destination = root / str(packed_k)
                weights = Weights(source, destination, packed_k=packed_k)
                for name in values:
                    weights.immutable(name)
                actual = weights.array("W.packed")
                np.testing.assert_array_equal(actual.transpose(1, 0, 2).reshape(raw.shape), raw)
                self.assertEqual(weights.buffers["W.packed"]["layout"], f"e8p-q4-k{packed_k}-u32")
                self.assertEqual(
                    weights.buffers["W.packed"]["shape"], [512 // packed_k, 3, packed_k // 8]
                )
                for name in ("W.scale", "Sensitive"):
                    payload, _ = weights.raw(name)
                    self.assertEqual(payload, values[name].tobytes())
                    self.assertEqual(
                        weights.buffers[name]["data"]["sha256"], hashlib.sha256(payload).hexdigest()
                    )
                target = destination / "cache/weights/layer.safetensors"
                self.assertNotEqual(target.stat().st_ino, shard.stat().st_ino)
                begin, header = local_header(target)
                for name in ("W.scale", "Sensitive"):
                    self.assertEqual(
                        header["__metadata__"][f"orin.layout.{name}"], "native_contiguous"
                    )
                self.assertEqual(
                    target.stat().st_size - begin, sum(v.nbytes for v in values.values())
                )
                self.assertEqual(set(header) - {"__metadata__"}, set(values))
                self.assertEqual(list(target.parent.iterdir()), [target])
                self.assertEqual(file_hash(shard), source_sha)
            with self.assertRaisesRegex(ValueError, "identity changed"):
                index["metadata"]["orinfer.sha256"][shard.name] = "0" * 64
                (source / "model.safetensors.index.json").write_text(json.dumps(index))
                Weights(source, root / "bad", packed_k=128).immutable("W.packed")

    def test_repack_rejects_wrong_codec_or_incomplete_k_tile(self):
        for dtype, shape, layout in [
            (np.uint8, (3, 16), "e8p-q4-row-u32"),
            (np.uint32, (3, 15), "e8p-q4-row-u32"),
            (np.uint32, (3, 16), "q4-row-packed-little-nibbles"),
            (np.uint32, (1, 3, 16), "e8p-q4-row-u32"),
        ]:
            with (
                self.subTest(dtype=dtype, shape=shape, layout=layout),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                source = root / "source"
                source.mkdir()
                shard = source / "layer.safetensors"
                save_file(
                    {"W.packed": np.zeros(shape, dtype)},
                    str(shard),
                    metadata={"orin.layout.W.packed": layout},
                )
                index = dict(
                    metadata={
                        "orinfer.sha256": {
                            shard.name: hashlib.sha256(shard.read_bytes()).hexdigest()
                        }
                    },
                    weight_map={"W.packed": shard.name},
                )
                (source / "model.safetensors.index.json").write_text(json.dumps(index))
                with self.assertRaisesRegex(ValueError, "Invalid source E8P weight layout"):
                    Weights(source, root / "destination", packed_k=128).immutable("W.packed")

    def test_shards_are_linked_and_packed_layout_and_scale_bits_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            destination = root / "model"
            packed = np.arange(2 * 64, dtype=np.uint8).reshape(2, 64)
            scale = np.array([0.001, 0.02], np.float16)
            save_file(
                {"W.packed": packed, "W.scale": scale},
                str(source / "layer.safetensors"),
                metadata={
                    "orin.layout.W.packed": "q4-row-packed-little-nibbles",
                    "orin.layout.W.scale": "native_contiguous",
                },
            )
            index = dict(
                metadata={
                    "orinfer.sha256": {
                        "layer.safetensors": hashlib.sha256(
                            (source / "layer.safetensors").read_bytes()
                        ).hexdigest()
                    }
                },
                weight_map={"W.packed": "layer.safetensors", "W.scale": "layer.safetensors"},
            )
            (source / "model.safetensors.index.json").write_text(json.dumps(index))
            weights = Weights(source, destination)
            weights.immutable("W.packed")
            weights.immutable("W.scale")
            self.assertEqual(weights.buffers["W.packed"]["layout"], "q4-row-packed-little-nibbles")
            self.assertEqual(
                weights.buffers["W.scale"]["data"]["sha256"],
                hashlib.sha256(scale.tobytes()).hexdigest(),
            )
            target = destination / "cache/weights/layer.safetensors"
            self.assertEqual(target.stat().st_ino, (source / "layer.safetensors").stat().st_ino)
            weights.constant("Control", np.arange(8, dtype=np.int32))
            signs = np.array([-1, 1, -1, 1], np.int8)
            weights.constant("Signs", signs)
            selectors = np.arange(256, dtype=np.uint32).reshape(256, 1)
            weights.constant("Selectors", selectors)
            weights.finish()
            from safetensors import safe_open

            with safe_open(
                destination / "cache/weights/runtime.safetensors", framework="numpy"
            ) as runtime:
                np.testing.assert_array_equal(runtime.get_tensor("Signs"), signs)
                self.assertEqual(runtime.metadata()["orin.layout.Signs"], "native_contiguous")
                np.testing.assert_array_equal(runtime.get_tensor("Selectors"), selectors)
            self.assertEqual(weights.buffers["Selectors"]["dtype"], "u32")
            self.assertEqual(
                index["metadata"]["orinfer.sha256"]["layer.safetensors"],
                hashlib.sha256(target.read_bytes()).hexdigest(),
            )
            raw = bytearray((source / "layer.safetensors").read_bytes())
            raw[-1] ^= 1
            (source / "layer.safetensors").write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "identity changed"):
                Weights(source, root / "other").immutable("W.packed")


if __name__ == "__main__":
    unittest.main()
