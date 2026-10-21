"""Offline metadata failures preserve committed files and release GPU locks."""

import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.build.files import atomic_json, file_hash, source_hash
from tools.build.gpu import gpu_lock


class BuildFilesTests(unittest.TestCase):
    def test_failed_metadata_publication_preserves_previous_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "state.json"
            atomic_json(path, {"name": "音频"})
            original = path.read_bytes()
            with self.assertRaises(TypeError):
                atomic_json(path, {"bad": object()})
            self.assertEqual(path.read_bytes(), original)
            with patch("tools.build.files.os.fsync", side_effect=OSError("disk failure")):
                with self.assertRaisesRegex(OSError, "disk failure"):
                    atomic_json(path, {"version": 2})
            self.assertEqual(path.read_bytes(), original)
            with patch.object(Path, "replace", side_effect=OSError("rename failure")):
                with self.assertRaisesRegex(OSError, "rename failure"):
                    atomic_json(path, {"version": 3})
            self.assertEqual(list(root.iterdir()), [path])
            self.assertEqual(file_hash(str(path)), hashlib.sha256(original).hexdigest())

    def test_source_identity_tracks_names_contents_and_stable_order(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            a, b = root / "a.py", root / "b.py"
            a.write_text("a")
            b.write_text("b")
            before = source_hash(root, [a, b])
            self.assertEqual(before, source_hash(root, [b, a, b]))
            a.write_text("changed")
            self.assertNotEqual(before, source_hash(root, [a, b]))
            a.write_text("a")
            c = a.rename(root / "c.py")
            self.assertNotEqual(before, source_hash(root, [c, b]))

    def test_gpu_lock_covers_exception_and_releases_for_other_processes(self):
        fake = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False))
        with tempfile.TemporaryDirectory() as folder, patch.dict(sys.modules, torch=fake):
            path = Path(folder) / "gpu.lock"

            def available():
                return (
                    subprocess.run(
                        [
                            sys.executable,
                            "-c",
                            "import fcntl,sys; f=open(sys.argv[1],'a'); "
                            "fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)",
                            str(path),
                        ],
                        capture_output=True,
                    ).returncode
                    == 0
                )

            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                with gpu_lock(path):
                    self.assertFalse(available())
                    raise RuntimeError("cancelled")
            self.assertTrue(available())
