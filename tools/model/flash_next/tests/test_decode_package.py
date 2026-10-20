"""Rebuilt native launch order must preserve buffer slices and reject scalars."""

import unittest
from tools.model.flash_next.mixed_verify import share_verification_state

from tools.model.host_compiler import (
    pointer_bindings,
    replace_binding,
)


class DecodePackageTests(unittest.TestCase):
    def test_verification_profiles_share_capacity_without_changing_shapes_or_lengths(self):
        data = dict(
            metadata=dict(
                mtp={},
                buffers=[
                    dict(name="VerifyM2_Saved_0_gdn_K", dtype="f16", shape=[16, 2, 128]),
                    dict(name="VerifyM8_Saved_0_gdn_K", dtype="f16", shape=[16, 8, 128]),
                    dict(name="State_0_gdn", dtype="f32", shape=[48, 128, 128]),
                ],
                reset_buffers=["VerifyM2_Saved_0_gdn_K"],
            ),
            buffer_scopes={
                "VerifyM2_Saved_0_gdn_K": "workspace",
                "VerifyM8_Saved_0_gdn_K": "workspace",
                "State_0_gdn": "sequence",
            },
        )
        package = dict(
            kernels=[
                dict(args=[dict(kind="buffer_slice", name="VerifyM2_Saved_0_gdn_K", offset=256)])
            ]
        )
        share_verification_state(data, package)
        self.assertTrue(data["metadata"]["mtp"]["shared_verification_state"])
        self.assertEqual(len(data["metadata"]["buffers"]), 2)
        shared = next(
            b for b in data["metadata"]["buffers"] if b["name"].startswith("VerifyShared")
        )
        self.assertEqual(shared["shape"], [16, 8, 128])
        self.assertEqual(
            package["kernels"][0]["args"],
            [dict(kind="buffer_slice", name=shared["name"], offset=256)],
        )
        self.assertEqual(data["metadata"]["reset_buffers"], [shared["name"]])
        share_verification_state(data, package)
        self.assertEqual(len(data["metadata"]["buffers"]), 2)

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


if __name__ == "__main__":
    unittest.main()
