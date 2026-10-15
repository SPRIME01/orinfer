"""Length arithmetic and legal fixed AOT exclusions during convergence."""

import unittest
from tools.model.flash_next.converge_kernels import divide_rows, split_fixed
from tools.model.flash_next.row_tuning import intervals


class ConvergenceTests(unittest.TestCase):
    def test_merger_scales_grid_and_scalar_in_one_row_contract(self):
        def evaluate(expr, rows):
            if expr["op"] == "rows":
                return rows
            if expr["op"] == "constant":
                return expr["value"]
            a, b = evaluate(expr["lhs"], rows), evaluate(expr["rhs"], rows)
            return a + b if expr["op"] == "add" else a // b

        scalar = divide_rows(dict(op="rows"), 4)
        grid = divide_rows(
            dict(
                op="divide",
                lhs=dict(op="add", lhs=dict(op="rows"), rhs=dict(op="constant", value=63)),
                rhs=dict(op="constant", value=64),
            ),
            4,
        )
        for patches in [4, 140, 256, 540, 8192, 32768]:
            self.assertEqual(evaluate(scalar, patches), patches // 4)
            self.assertEqual(evaluate(grid, patches), (patches // 4 + 63) // 64)
        for patches in [1, 2, 3]:
            self.assertGreater(evaluate(grid, patches), 0)

    def test_exact_precompilations_do_not_steal_neighbouring_rows(self):
        self.assertEqual(split_fixed(1, 40, {"1": "a", "2": "b", "32": "c"}), [(3, 31), (33, 40)])

    def test_measured_winner_drops_unused_candidates(self):
        r = intervals(
            [dict(rows=1, times=dict(a=1, b=2)), dict(rows=512, times=dict(a=1, b=3))], 512
        )
        self.assertEqual(r["used"], ["a"])
        self.assertEqual(r["intervals"], [[1, 512, "a"]])

    def test_coalescing_preserves_expert_assignment_multiplier(self):
        import copy
        from tools.model.flash_next.converge_kernels import coalesce_single_implementations

        name = "flash_decode/layer0/post/k18/rows/1-4"
        kernel = dict(name=name, module=dict(sha256="same"), args=[dict(kind="i32", value=40)])
        expression = dict(op="multiply", lhs=dict(op="rows"), rhs=dict(op="constant", value=10))
        launch = dict(
            name=name,
            capacity=4,
            grid=[expression, dict(op="constant", value=1), dict(op="constant", value=1)],
            arguments=[dict(index=0, value=expression)],
        )
        original = copy.deepcopy(launch)
        package = dict(
            kernels=[kernel], dynamic_batch_kernels=[launch], plan_kernel_contracts=[name]
        )

        class Compiler:
            def host(self, k):
                return dict(ordered_arguments=[dict(ctype="ctypes.c_int32", value="rows")])

        coalesce_single_implementations(package, Compiler())
        self.assertEqual(package["kernels"][0]["args"], [dict(kind="i32", value=40)])
        original["name"] = name.split("/rows/")[0]
        self.assertEqual(package["dynamic_batch_kernels"], [original])
