"""Native catalog factoring must preserve real host parameter order and launches."""

import copy
import unittest

from tools.model.catalog import expand, factor, rust_source


class CatalogTests(unittest.TestCase):
    def package(self):
        identity = dict(file="assets.safetensors", tensor="a" * 64, sha256="a" * 64)
        kernel = dict(
            name="layer0/projection",
            module=identity,
            source=identity,
            host_abi=identity,
            symbol="main",
            grid=[128, 1, 1],
            block=[128, 1, 1],
            shared_memory_bytes=1024,
            cooperative=False,
            args=[dict(kind="buffer", name="Weight0"), dict(kind="i32", value=128)],
            readonly_arguments=[True, False],
        )
        second = copy.deepcopy(kernel)
        second["name"] = "layer1/projection"
        second["args"][0]["name"] = "Weight1"
        return dict(
            kernels=[kernel, second],
            plan_kernel_contracts=[kernel["name"]],
            dynamic_batch_kernels=[
                dict(
                    name=kernel["name"],
                    capacity=128,
                    grid=[
                        dict(op="rows"),
                        dict(op="constant", value=1),
                        dict(op="constant", value=1),
                    ],
                    arguments=[dict(index=1, value=dict(op="rows"))],
                )
            ],
        )

    def test_tensor_roles_are_native_and_definitions_are_shared(self):
        package = self.package()
        factored, bindings = factor(package)
        self.assertNotIn("kernels", factored)
        self.assertNotIn("dynamic_batch_kernels", factored)
        self.assertNotIn("plan_kernel_contracts", factored)
        self.assertEqual(len(factored["catalog"]["assets"]), 1)
        self.assertEqual(len(factored["catalog"]["definitions"]), 1)
        kernels, dynamic, contracts = expand(factored["catalog"], bindings)
        self.assertEqual(kernels, package["kernels"])
        self.assertEqual(dynamic, package["dynamic_batch_kernels"])
        self.assertEqual(contracts, package["plan_kernel_contracts"])
        self.assertIn('"Weight1"', rust_source(factored["catalog"], bindings))

    def test_abi_launch_or_access_changes_cannot_share_a_definition(self):
        for field, value in (
            ("block", [64, 1, 1]),
            ("shared_memory_bytes", 0),
            ("readonly_arguments", [False, False]),
            ("symbol", "other"),
        ):
            p = self.package()
            p["kernels"][1][field] = value
            factored, _ = factor(p)
            self.assertEqual(len(factored["catalog"]["definitions"]), 2)
        p = self.package()
        p["kernels"][1]["args"][1] = dict(kind="u32", value=128)
        factored, bindings = factor(p)
        self.assertEqual(len(factored["catalog"]["definitions"]), 2)
        bindings[1]["args"][1]["kind"] = "i32"
        with self.assertRaisesRegex(ValueError, "compiled ABI"):
            expand(factored["catalog"], bindings)

    def test_native_mapping_identity_covers_roles_offsets_and_dynamic_launch(self):
        original = self.package()
        before, _ = factor(original)
        for change in ("role", "offset", "grid", "dynamic"):
            p = self.package()
            if change == "role":
                p["kernels"][0]["args"][0]["name"] = "WrongWeight"
            elif change == "offset":
                p["kernels"][0]["args"][0] = dict(kind="buffer_slice", name="Weight0", offset=64)
            elif change == "grid":
                p["kernels"][0]["grid"][0] = 64
            else:
                p["dynamic_batch_kernels"][0]["capacity"] = 64
            after, _ = factor(p)
            self.assertNotEqual(
                before["catalog"]["binding_signature"], after["catalog"]["binding_signature"]
            )

    def test_duplicate_or_missing_contract_is_rejected(self):
        p = self.package()
        p["kernels"].append(p["kernels"][0])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            factor(p)
        p = self.package()
        p["plan_kernel_contracts"].append("missing")
        with self.assertRaisesRegex(ValueError, "Missing"):
            factor(p)
