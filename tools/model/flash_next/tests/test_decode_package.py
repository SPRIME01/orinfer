"""Rebuilt native launch order must preserve buffer slices and reject scalars."""

import unittest

from tools.model.flash_next.optimize_decode import (
    pointer_bindings,
    replace_binding,
    rotation_geometry,
    validate_gdn_prefill_state,
)


class DecodePackageTests(unittest.TestCase):
    def test_explicit_dynamic_binding_retains_offsets_and_validates_row_scalars(self):
        original = dict(
            name="joined",
            args=[
                dict(kind="buffer_slice", name="Input", offset=256),
                dict(kind="i32", value=1024),
            ],
        )
        host = dict(
            ordered_arguments=[
                dict(ctype="ctypes.c_void_p", value="A.data_ptr()"),
                dict(ctype="ctypes.c_int32", value="m"),
            ],
            symbol="dynamic",
            launch_expressions=dict(
                gridDimX="(m+31)//32",
                gridDimY="1",
                gridDimZ="1",
                blockDimX="128",
                blockDimY="1",
                blockDimZ="1",
                sharedMemBytes="0",
            ),
        )
        with self.assertRaises(ValueError):
            pointer_bindings(original, host)
        pointers = pointer_bindings(original, host, allow_scalars=True)
        binding = replace_binding(original, host, {}, pointers, dimensions=dict(m=1024))
        self.assertEqual(binding["grid"], [32, 1, 1])
        self.assertEqual(binding["args"], original["args"])

    def test_gdn_matrix_and_flat_arena_have_the_same_extent(self):
        for shape in ([48, 128, 128], [48 * 128 * 128]):
            validate_gdn_prefill_state(dict(dtype="f32", shape=shape), dict(kind="buffer"))
        for dtype, shape, offset in (
            ("f16", [48, 128, 128], 0),
            ("f32", [48, 128, 127], 0),
            ("f32", [48, 128, 128], 4),
        ):
            with self.assertRaisesRegex(ValueError, "FP32"):
                validate_gdn_prefill_state(dict(dtype=dtype, shape=shape), dict(offset=offset))

    def test_reordered_launch_preserves_slice_without_mutating_original(self):
        original = dict(
            name="verify_m3/layer0/k30",
            args=[
                dict(kind="buffer_slice", name="Input", offset=256),
                dict(kind="buffer", name="Output"),
            ],
        )
        host = dict(
            ordered_arguments=[
                dict(ctype="ctypes.c_void_p", value="A.data_ptr()"),
                dict(ctype="ctypes.c_void_p", value="C.data_ptr()"),
            ]
        )
        pointers = pointer_bindings(original, host)
        rebuilt = dict(
            ordered_arguments=list(reversed(host["ordered_arguments"])),
            symbol="routed_kernel",
            launch_expressions=dict(
                gridDimX="32",
                gridDimY="10",
                gridDimZ="1",
                blockDimX="128",
                blockDimY="1",
                blockDimZ="1",
                sharedMemBytes="16384",
            ),
        )
        binding = replace_binding(original, rebuilt, {}, pointers)
        self.assertEqual(binding["args"], list(reversed(original["args"])))
        self.assertEqual(binding["grid"], [32, 10, 1])
        binding["args"][1]["offset"] = 512
        self.assertEqual(original["args"][0]["offset"], 256)
        self.assertEqual(pointers["A"]["offset"], 256)

    def test_symbolic_or_incomplete_abi_is_rejected(self):
        original = dict(args=[dict(kind="buffer", name="Input")])
        for host in [
            dict(ordered_arguments=[]),
            dict(ordered_arguments=[dict(ctype="ctypes.c_int32", value="m")]),
        ]:
            with self.assertRaises(ValueError):
                pointer_bindings(original, host)

    def test_mtp_last_row_slice_and_full_verification_have_distinct_extents(self):
        buffers = {
            "hidden": dict(name="hidden", dtype="f16", shape=[3, 128]),
            "signs": dict(name="Q4_output_signs", dtype="i8", shape=[128]),
            "activation": dict(name="activation", dtype="i8", shape=[3, 128]),
            "scale": dict(name="scale", dtype="f16", shape=[3]),
        }
        pointers = {
            "X": dict(kind="buffer_slice", name="hidden", offset=512),
            "Signs": dict(kind="buffer", name="signs"),
            "Q": dict(kind="buffer", name="activation"),
            "S": dict(kind="buffer", name="scale"),
        }
        kernel = dict(grid=[3, 1, 1])
        self.assertEqual(
            rotation_geometry("mtp_head_m3", kernel, pointers, buffers), (1, 128, False)
        )
        with self.assertRaisesRegex(ValueError, "workspace"):
            rotation_geometry("verify_m3", kernel, pointers, buffers)
        pointers["X"]["offset"] = 0
        self.assertEqual(rotation_geometry("verify_m3", kernel, pointers, buffers), (3, 128, False))
        pointers["X"]["offset"] = 768
        with self.assertRaisesRegex(ValueError, "workspace"):
            rotation_geometry("mtp_head_m3", kernel, pointers, buffers)


if __name__ == "__main__":
    unittest.main()
