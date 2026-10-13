"""Publish independent MTP draft rows using shared projections and private QSA.

Reuse warm-plan cubins for stateless stages. A warm plan's causal QSA never
processes independent requests: each request executes the one-token QSA with
its own position, index, pending block and INT8 KV. Weight precision is retained.
"""

import argparse
import copy
import math
from pathlib import Path

from tools.model.flash_next.mixed_verify import Interfaces, share_verification_state
from tools.model.flash_next.optimize_decode import replace_binding
from tools.model.publication import (
    atomic_model,
    clone_model,
    commit_package,
    file_hash,
    link_or_copy,
    load_model,
)

PROFILES = (2, 4, 8)
PRIVATE = (9, 10, *range(12, 29))
SECTIONS = {"begin": range(3, 8), "pre": (*range(9), 11), "post": range(29, 54)}


def upgrade(model, destination, output):
    import tilelang
    from kernels.model.q4_projection import q4_gemv
    from kernels.model.rotation_a8 import rotate_activation_a8
    from tools.operators.abi import parse_host
    from tools.operators.common import configure, export_kernel

    configure()
    data, origin, package = load_model(model)
    metadata = data["metadata"]
    spec = metadata.get("mtp")
    if not spec or not spec.get("batch_verification_tokens") or spec.get("batch_draft_profiles"):
        raise ValueError("Expected a joined target-verification package without draft batching")
    operator = clone_model(model, destination, origin)
    original = {k["name"]: k for k in package["kernels"]}
    buffers = {b["name"]: b for b in metadata["buffers"]}
    api = Interfaces(origin)
    additions = []
    widths = {"f16": 2, "bf16": 2, "f32": 4, "i32": 4, "u32": 4, "u16": 2, "i8": 1}

    def arena(name, dtype, shape):
        if name in buffers:
            raise ValueError("Duplicate draft arena")
        b = dict(
            name=name,
            dtype=dtype,
            shape=list(shape),
            alignment=256,
            layout="native_contiguous",
            access="read_write",
        )
        buffers[name] = b
        data["buffer_scopes"][name] = "workspace"

    def clone(old, name, aliases):
        k = copy.deepcopy(old)
        k["name"] = name
        for arg in k["args"]:
            if arg.get("name") in aliases:
                arg["name"] = aliases[arg["name"]]
        additions.append(k)
        return k

    for rows in PROFILES:
        prefix = f"flash_draft_m{rows}"
        warm = f"mtp_warm_m{rows}"
        shared = []
        for stage, indices in SECTIONS.items():
            source = "begin" if stage == "begin" else "layer48"
            shared.extend(original[f"{warm}/{source}/k{i}"] for i in indices)
        shared.extend(original[f"mtp_head_m{rows}/body/k{i}"] for i in range(4))
        aliases = {}
        for k in shared:
            for arg in k["args"]:
                name = arg.get("name")
                if not name or data["buffer_scopes"][name] == "weights":
                    continue
                if data["buffer_scopes"][name] != "workspace":
                    raise ValueError(f"Draft shared stage references private state: {name}")
                if name not in aliases:
                    aliases[name] = f"DraftBatchM{rows}_{name}"
                    arena(aliases[name], buffers[name]["dtype"], buffers[name]["shape"])
        for stage, indices in SECTIONS.items():
            source = "begin" if stage == "begin" else "layer48"
            for index, i in enumerate(indices):
                clone(original[f"{warm}/{source}/k{i}"], f"{prefix}/{stage}/k{index}", aliases)
        for tokens in range(1, rows + 1):
            for index, i in enumerate(PRIVATE):
                old = original[f"mtp_warm_m{tokens}/layer48/k{i}"]
                parallel = original[f"{warm}/layer48/k{i}"]
                if api.get(old)[0] != api.get(parallel)[0]:
                    raise ValueError("Private causal QSA interface differs")
                left, right = api.pointers(old), api.pointers(parallel)
                boundary = {
                    left[a]["name"]: aliases[v["name"]]
                    for a, v in right.items()
                    if v["name"] in aliases
                }
                clone(old, f"{prefix}/warm_m{tokens}/private/k{index}", boundary)
                if tokens == 1:
                    clone(old, f"{prefix}/private/k{index}", boundary)
        for i in range(4):
            clone(original[f"mtp_head_m{rows}/body/k{i}"], f"{prefix}/head/k{i}", aliases)
        mixed = api.pointers(original[f"mtp_head_m{rows}/body/k3"])["Out"]["name"]
        controls = {
            "activation": ("i32", (rows, 640)),
            "scale": ("f16", (rows,)),
            "logits": ("f32", (rows, metadata["vocab"])),
            "pairs": ("i32", (rows, 2)),
        }
        roles = {role: f"DraftBatchM{rows}_{role}" for role in controls}
        for role, (dtype, shape) in controls.items():
            arena(roles[role], dtype, shape)
        for label, factory, bindings, slot in [
            (
                "rotate",
                lambda: rotate_activation_a8.__wrapped__(rows, 2560),
                {
                    "X": aliases[mixed],
                    "Signs": "Q4_output_signs",
                    "Q": roles["activation"],
                    "S": roles["scale"],
                },
                4,
            ),
            (
                "head",
                lambda: q4_gemv.__wrapped__(rows, metadata["vocab"], 2560, output_dtype="float32"),
                {
                    "A": roles["activation"],
                    "W": "Q4_output_packed",
                    "S": "Q4_output_scale",
                    "AS": roles["scale"],
                    "Out": roles["logits"],
                },
                5,
            ),
        ]:
            asset = output / "aot" / f"draft-{rows}-{label}"
            kernel = tilelang.compile(
                factory(),
                out_idx=[],
                execution_backend="nvrtc",
                target={"kind": "cuda", "arch": "sm_87"},
            )
            export_kernel(kernel, asset)
            assets = {}
            for field, filename in [
                ("module", "kernel.cubin"),
                ("source", "kernel.cu"),
                ("host_abi", "host.txt"),
            ]:
                target = operator / "draft-batch" / f"m{rows}-{label}" / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                link_or_copy(asset / filename, target)
                assets[field] = dict(
                    file=str(target.relative_to(operator)), sha256=file_hash(target)
                )
            (host,) = parse_host((asset / "host.txt").read_text())
            additions.append(
                replace_binding(
                    dict(name=f"{prefix}/head/k{slot}"),
                    host,
                    assets,
                    {a: dict(kind="buffer", name=b) for a, b in bindings.items()},
                )
            )
        # Existing full-row greedy kernels consume all independently generated logits.
        for i in (6, 7):
            old = original[f"flash_batch_m{rows}/head/k{i}"]
            rename = {
                f"BatchM{rows}_Logits": roles["logits"],
                f"BatchM{rows}_Selected": roles["pairs"],
            }
            clone(old, f"{prefix}/head/k{i}", rename)
        # Rust derives lane offsets from these dedicated row-major workspace roles.
        for k in additions:
            if k["name"].startswith(prefix + "/") and "/private/" in k["name"]:
                for arg in k["args"]:
                    name = arg.get("name", "")
                    if name.startswith(f"DraftBatchM{rows}_"):
                        b = buffers[name]
                        if math.prod(b["shape"]) * widths[b["dtype"]] % rows:
                            raise ValueError("Private QSA boundary is not row-aligned")
    for tokens in range(1, max(PROFILES) + 1):
        for role in ("Embedding", "Condition"):
            data["buffer_scopes"][f"DraftM{tokens}_{role}"] = "sequence"
    spec["batch_draft_profiles"] = list(PROFILES)
    spec["batch_commit"] = True
    spec["batch_warm"] = True
    spec["batch_programs"] = sorted(
        set(spec.get("batch_programs", []))
        | {
            f"flash_draft_m{rows}/warm_m{tokens}/private"
            for rows in PROFILES
            for tokens in range(1, rows + 1)
        }
        | {
            f"flash_draft_m{rows}/{stage}"
            for rows in PROFILES
            for stage in ("begin", "pre", "private", "post", "head")
        }
        | {
            f"flash_verify_segment_m{tokens}/layer{layer}/private"
            for tokens in spec["batch_verification_tokens"]
            for layer in range(48)
        }
        | {
            name
            for tokens in spec["batch_verification_tokens"]
            for name in (
                f"flash_verify_segment_m{tokens}/layer1/ple",
                f"flash_verify_capture_m{tokens}",
            )
        }
    )
    metadata["buffers"] = list(buffers.values())
    package["kernels"].extend(additions)
    share_verification_state(data, package)
    package["buffer_contracts"] = [
        {k: v for k, v in b.items() if k != "data"} for b in metadata["buffers"]
    ]
    commit_package(destination, operator, data, package)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    args = parser.parse_args()
    with atomic_model(args.model_output) as staging:
        upgrade(args.model, staging, args.output)


if __name__ == "__main__":
    main()
