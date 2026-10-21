"""Shared tuning preserves rejection, replay validation and cache identity."""

from pathlib import Path
import tempfile
import unittest

from tools.build.files import atomic_json
from tools.model.tuning import compile_candidates, measure, read_record, record_path


class TuningTests(unittest.TestCase):
    def test_corrupt_stale_and_incomplete_records_are_recomputed(self):
        with tempfile.TemporaryDirectory() as folder:
            identity = dict(seed=17, geometry=[128, 512])
            path = record_path(Path(folder), identity)
            self.assertEqual(
                path, record_path(Path(folder), dict(reversed(list(identity.items()))))
            )
            self.assertIsNone(read_record(path, identity))
            path.write_text("{truncated")
            self.assertIsNone(read_record(path, identity))
            for value in ([], None, {"identity": {}}, {"identity": identity}):
                atomic_json(path, value)
                self.assertIsNone(read_record(path, identity, lambda r: r["checked_rows"]))
            record = dict(identity=identity, checked_rows=[1, 7, 128])
            atomic_json(path, record)
            self.assertEqual(read_record(path, identity), record)

    def test_failed_candidates_keep_their_config_and_do_not_shift_valid_indices(self):
        def build(config):
            if config == 2:
                raise ValueError("unsupported tile")
            return config * 10

        kernels, configs, rejected = compile_candidates([1, 2, 3], build, tolerate_failures=True)
        self.assertEqual((kernels, configs), ([10, 30], [1, 3]))
        self.assertEqual(rejected[0]["config"], 2)
        with self.assertRaisesRegex(ValueError, "unsupported tile"):
            compile_candidates([2], build)
        with self.assertRaisesRegex(ValueError, "All tile candidates"):
            compile_candidates([2], build, tolerate_failures=True)

    def test_changed_input_is_restored_after_replay_failure(self):
        state, calls = [0], []

        def benchmark(run, **kwargs):
            calls.append(kwargs)
            return {"median_ms": 0.01}, type("Graph", (), {"replay": lambda self: run()})()

        def change():
            state[0] ^= 1

        def check():
            self.assertEqual(state[0], 1)
            raise AssertionError("changed input")

        with self.assertRaisesRegex(AssertionError, "changed input"):
            measure(
                lambda: None,
                budget_ms=0.25,
                max_repetitions=8,
                change=change,
                check=check,
                benchmark=benchmark,
            )
        self.assertEqual(state[0], 0)
        self.assertEqual(calls, [dict(repetitions=1, warmup=1), dict(repetitions=8, warmup=0)])
