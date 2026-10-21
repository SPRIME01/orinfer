import hashlib
import json
from pathlib import Path
import tempfile
import tarfile
import unittest
from unittest.mock import patch
from tools.release.package import archive, execution_assets, verify


class ReleaseTests(unittest.TestCase):
    def test_reproducible_archive_and_download_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = root / "bundle"
            bundle.mkdir()
            (bundle / "source.txt").write_text("fixed source")
            archive(bundle, root / "one.tar.xz", 100)
            archive(bundle, root / "two.tar.xz", 100)
            self.assertTrue((root / "one.tar.xz").read_bytes().startswith(b"\xfd7zXZ\0"))
            with tarfile.open(root / "one.tar.xz") as stream:
                entry = stream.getmember("bundle/source.txt")
                self.assertEqual((entry.uid, entry.gid, entry.mtime), (0, 0, 100))
            self.assertEqual((root / "one.tar.xz").read_bytes(), (root / "two.tar.xz").read_bytes())
            digest = hashlib.sha256((root / "one.tar.xz").read_bytes()).hexdigest()
            (root / "SHA256SUMS").write_text(f"{digest}  one.tar.xz\n")
            verify(root)
            (root / "one.tar.xz").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify(root)


class ExecutionAssetsTests(unittest.TestCase):
    def test_model_variants_have_distinct_assets_and_tampered_identity_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            header = bytearray(20)
            header[:6] = b"\x7fELF\x02\x01"
            header[16:20] = bytes((3, 0, 183, 0))
            identity = dict(
                abi_version=1,
                package="orinfer-models",
                version="0.1.2",
                library=dict(file="lib/model.so", sha256=hashlib.sha256(header).hexdigest()),
            )
            archives = []
            for index, architecture in enumerate(("qwen3_5", "flash_next", "qwen3_5")):
                manifest = dict(
                    schema_version=1,
                    runtime_abi=1,
                    architecture=architecture,
                    compute_policy="int8_quality",
                    target="sm_87",
                    catalog=dict(
                        assets=[], definitions=[], dynamic=[], binding_signature=str(index) * 64
                    ),
                    execution=identity,
                )
                raw = json.dumps(manifest).encode()
                package = root / hashlib.sha256(raw).hexdigest()
                (package / "lib").mkdir(parents=True)
                (package / "package.json").write_bytes(raw)
                (package / "lib/model.so").write_bytes(header)
                suffix = ".tar.gz" if architecture == "qwen3_5" else ".tar.xz"
                path = root / (f"{architecture}-{index}" + suffix)
                from tools.model.package import archive as archive_package

                if suffix == ".tar.gz":
                    with tarfile.open(path, "w:gz") as stream:
                        stream.add(package, arcname=package.name)
                else:
                    archive_package(package, path)
                archives.append(path)
            with patch("tools.release.package.execution_identity", return_value=identity):
                assets = execution_assets(archives, root / "cache", Path("/bin/true"), "v0.1.2")
                names = [a["metadata"]["file"] for a in assets]
                self.assertEqual(len(set(names)), 3)
                self.assertIn("flash_next", names[1])
                self.assertNotIn("27b", names[1])
                self.assertEqual(
                    [a["metadata"]["architecture"] for a in assets],
                    ["qwen3_5", "flash_next", "qwen3_5"],
                )
                from tools.model.package import install

                for asset in assets:
                    output = root / asset["metadata"]["file"]
                    self.assertTrue(output.name.endswith(".tar.xz"))
                    archive(asset["directory"], output, 100)
                    self.assertTrue(output.read_bytes().startswith(b"\xfd7zXZ\0"))
                    self.assertEqual(
                        install(output, root / "reinstalled"), asset["metadata"]["digest"]
                    )
            with patch(
                "tools.release.package.execution_identity",
                return_value=dict(identity, version="wrong"),
            ):
                with self.assertRaisesRegex(ValueError, "differs from package"):
                    execution_assets(archives, root / "bad-cache", Path("/bin/true"), "v0.1.2")
