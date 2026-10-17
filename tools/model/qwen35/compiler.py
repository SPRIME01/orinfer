"""Reuse Flash Next's content cache and host-ABI emitter for native Qwen AOT."""

import hashlib
from pathlib import Path
from tools.model.flash_next.build_cache import BuildCache
from tools.model.flash_next.compiler import Compiler
from tools.model.flash_next.converge_kernels import divide_rows
from tools.operators.common import export_kernel

ROOT = Path(__file__).resolve().parents[3]


class Aot:
    def __init__(self, root, output, cache, buffers, *, compile_lock):
        self.root, self.output, self.lock = root, output, compile_lock
        self.output.mkdir(parents=True, exist_ok=True)
        self.cache = BuildCache(cache)
        self.compiler = Compiler(root, root, output, buffers)
        self.kernels, self.dynamic = [], []
        self.objects = {}
        sha = hashlib.sha256()
        for directory in [ROOT / "kernels", ROOT / "tools/model/qwen35"]:
            for file in sorted(directory.rglob("*.py")):
                if file.name.startswith(("test_", "check_")):
                    continue
                sha.update(str(file.relative_to(ROOT)).encode())
                sha.update(file.read_bytes())
        for file in [
            ROOT / "tools/model/compile.py",
            ROOT / "tools/model/row_tuning.py",
            ROOT / "tools/model/row_selection.py",
            ROOT / "tools/model/flash_next/compiler.py",
            ROOT / "tools/model/flash_next/decode_tuning.py",
            ROOT / "tools/model/flash_next/decode_policy.py",
            ROOT / "tools/model/flash_next/converge_kernels.py",
            ROOT / "tools/operators/abi.py",
            ROOT / "tools/operators/common.py",
        ]:
            sha.update(file.read_bytes())
        import torch
        import tilelang

        self.identity = dict(
            source=sha.hexdigest(),
            tilelang=tilelang.__version__,
            cuda=torch.version.cuda,
            target="sm_87",
        )

    def emit(self, name, key, build, bindings, rows=4096, *, dynamic=True, row_divisor=1):
        if key not in self.objects:

            def construct(folder):
                with self.lock():
                    kernel = build()
                    export_kernel(kernel, folder)

            artifact, hit = self.cache.get(dict(**self.identity, operator=key), construct)
            self.objects[key] = artifact
            print("AOT", key, "cached" if hit else "compiled", flush=True)
        artifact = self.objects[key]
        kernel, launch = self.compiler.emit(
            dict(name=name),
            name,
            {
                k: (dict(kind="buffer", name=v) if isinstance(v, str) else v)
                for k, v in bindings.items()
            },
            rows,
            key=key,
            artifact=artifact,
        )
        self.kernels.append(kernel)
        if dynamic:
            if row_divisor != 1:
                launch = divide_rows(launch, row_divisor)
                launch["capacity"] = rows * row_divisor
            self.dynamic.append(launch)
        return dict(kind="kernel", name=name)
