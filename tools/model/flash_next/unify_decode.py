"""Publish one decode-row template with local small-batch implementations.

All variants bind the canonical operand roles and the same bounded workspace.
Only launch scalars and local tiles depend on live rows; no serving compilation.
"""

import argparse
import copy
import math
import re
import shutil
from pathlib import Path

from tools.model.compact import compact_package
from tools.model.flash_next.optimize_decode import pointer_bindings, replace_binding
from tools.model.publication import (
    AssetReader,
    atomic_model,
    clone_model,
    file_hash,
    link_or_copy,
    load_model,
    write_json,
)
from tools.model.upgrade_dynamic_batch import contract
from tools.operators.abi import parse_host, validate_parameter_count

CAPACITY = 128
PREFIX = "flash_decode"
ARENA = "FlashDecode_"


def align_sections(small, large, pointers):
    """Align local fusions by their typed operands, retaining the model's order."""
    pairs, skipped = [], []
    i = j = 0
    while i < len(small):
        s, l = small[i], large[j]
        a = pointers(s)
        if {"Hi", "Lo", "Normed", "Out"} <= a.keys():
            mix = large[j + 1]
            if set(pointers(mix)) != {"Mixed", "Normed", "Up"}:
                raise ValueError("HC fusion must consume the adjacent mix")
            pairs.append((s, l, mix))
            skipped.append(mix)
            j += 2
        elif {"Gate", "Up", "GateScale", "UpScale", "TokenScale"} <= a.keys():
            up, act = large[j + 1 : j + 3]
            if set(pointers(act)) != {"Gate", "Up", "Output"}:
                raise ValueError("Shared FFN fusion must consume adjacent gate/up/SwiGLU")
            pairs.append((s, l, act))
            skipped.extend([up, act])
            j += 3
        else:
            pairs.append((s, l, None))
            j += 1
        i += 1
    if j != len(large):
        raise ValueError("Unmatched canonical decode operations")
    return pairs, skipped


def upgrade(model, destination, output):
    import tilelang
    from kernels.model.rows import explicit_rows
    from kernels.model.hyperconnection import hc_down_finish
    from kernels.model.bf16_single import bf16_single
    from kernels.model.hc_bf16_cache import hc_up13_mix
    from kernels.model.e8p_exact import e8p_exact_grouped, split_a8
    from kernels.model.q4_projection import q4_gemv
    from kernels.model.int8_projection import int8_small_projection
    from kernels.model.flash_next import scalar_bf16_projection
    from tools.model.package import model_library
    from tools.operators.common import configure, export_kernel

    configure()
    data, origin, package = load_model(model)
    if data["architecture"] != "flash_next" or package["prefill_profiles"] != [
        dict(tokens=4096, min_tokens=1, kind="flash_dynamic_recurrent")
    ]:
        raise ValueError("Expected Flash Next with the published dynamic prefill template")
    originals = {k["name"]: k for k in package["kernels"]}
    if f"{PREFIX}/begin/k0" in originals or "flash_dynamic_m128/begin/k0" not in originals:
        raise ValueError("Expected an unmodified dynamic Flash decode package")
    root = clone_model(model, destination, origin)
    reader, hosts, compiled = AssetReader(origin), {}, {}
    buffers = {b["name"]: b for b in data["metadata"]["buffers"]}
    dynamics = {c["name"]: c for c in package["dynamic_batch_kernels"]}
    additions, contracts = [], []

    def host(k):
        digest = k["host_abi"]["sha256"]
        if digest not in hosts:
            hosts[digest] = parse_host(reader.read(k["host_abi"]).decode())[0]
        return hosts[digest]

    def pointers(k):
        return pointer_bindings(k, host(k), allow_scalars=True)

    def arena(name):
        return name.replace("BatchM128_", ARENA).replace("ComputeQ4_R128_", "FlashDecodeQ4_")

    def remap(p, mapping=None):
        result = copy.deepcopy(p)
        for a in result.values():
            a["name"] = arena(a["name"]) if mapping is None else mapping.get(a["name"], a["name"])
        return result

    def scratch(name, dtype, shape):
        b = dict(
            name=name,
            dtype=dtype,
            shape=shape,
            layout="native_contiguous",
            alignment=256,
            access="read_write",
        )
        if name in buffers and buffers[name] != b:
            raise ValueError("Conflicting decode workspace")
        buffers[name] = b
        data["buffer_scopes"][name] = "workspace"
        return name

    # Copy only the canonical arena contract. Weight tensors remain immutable aliases.
    for name, b in list(buffers.items()):
        new = arena(name)
        if new != name:
            v = copy.deepcopy(b)
            v["name"] = new
            buffers[new] = v
            data["buffer_scopes"][new] = "workspace"
    parts = scratch("FlashDecodeHCPartials", "f32", [4, CAPACITY, 320])

    def emit(old, name, args=None, build=None, key=None, multiplier=1, *, symbolic=True):
        bound = remap(pointers(old)) if args is None else args
        if build is not None:
            if key not in compiled:
                print("COMPILE", key, flush=True)
                fn = explicit_rows(build()) if symbolic else build()
                generated = tilelang.compile(
                    fn,
                    out_idx=[],
                    execution_backend="nvrtc",
                    target={"kind": "cuda", "arch": "sm_87"},
                    pass_configs={
                        "tl.disable_thread_storage_sync": True,
                        "tl.disable_shared_memory_reuse": True,
                        "tl.if_stmt_binding_inline_replayable_binds": False,
                    }
                    if key.startswith("exact-")
                    else {},
                )
                folder = output / "aot" / key
                export_kernel(generated, folder)
                assets = {}
                for field, filename in [
                    ("module", "kernel.cubin"),
                    ("source", "kernel.cu"),
                    ("host_abi", "host.txt"),
                ]:
                    target = root / "unified-decode" / key / filename
                    target.parent.mkdir(parents=True, exist_ok=True)
                    link_or_copy(folder / filename, target)
                    assets[field] = dict(
                        file=str(target.relative_to(root)), sha256=file_hash(target)
                    )
                abi = parse_host((folder / "host.txt").read_text())[0]
                validate_parameter_count((folder / "kernel.cu").read_text(), abi)
                compiled[key] = abi, assets
            abi, assets = compiled[key]
        else:
            abi = host(old)
            assets = {f: old[f] for f in ("module", "source", "host_abi")}
        k = replace_binding(
            dict(old, name=name),
            abi,
            assets,
            bound,
            dimensions=dict(m=CAPACITY * multiplier, rows=CAPACITY * multiplier),
        )
        c = contract(k, abi, CAPACITY, multiplier)
        additions.append(k)
        contracts.append(c)
        return k

    shared = [k for k in package["kernels"] if k["name"].startswith("flash_dynamic_m128/")]
    canonical = {}
    for old in shared:
        suffix = old["name"].removeprefix("flash_dynamic_m128/")
        name = f"{PREFIX}/{suffix}"
        p = pointers(old)
        multiplier = (
            10
            if any(
                a["value"] in ("m", "rows") and old["args"][i].get("value") == 1280
                for i, a in enumerate(host(old)["ordered_arguments"])
            )
            else 1
        )
        down = set(p) == {"A", "C", "W"} and "hc_" in p["W"]["name"] and "_down." in p["W"]["name"]
        if down:
            n, k = buffers[p["W"]["name"]]["shape"]
            emit(
                old,
                name,
                args=dict(A=remap(p)["A"], Weight=p["W"], Output=dict(kind="buffer", name=parts)),
                build=lambda n=n: bf16_single.__wrapped__(
                    n, 2560, 4, rows=CAPACITY, dynamic_rows=True, row_major_grid=True
                ),
                key=f"hc-down-{n}",
            )
        elif set(p) == {"Activated", "Down"}:
            n = math.prod(buffers[p["Activated"]["name"]]["shape"]) // CAPACITY
            emit(
                old,
                name,
                args=dict(Out=remap(p)["Activated"], Partials=dict(kind="buffer", name=parts)),
                build=lambda n=n: hc_down_finish.__wrapped__(CAPACITY, n, dynamic_rows=True),
                key=f"hc-finish-{n}",
            )
        elif (
            set(p) == {"A", "Output", "Weight"}
            and buffers[p["Weight"]["name"]]["dtype"] == "bf16"
            and buffers[p["Weight"]["name"]]["shape"] == [512, 2560]
        ):
            emit(
                old,
                name,
                build=lambda: bf16_single.__wrapped__(
                    512, 2560, rows=CAPACITY, dynamic_rows=True, row_major_grid=True
                ),
                key="router-exact",
                args=remap(p),
            )
        elif {"P", "Counts", "TileExpert", "TileRow", "C"} <= p.keys():
            e, groups, n, words = buffers[p["P"]["name"]]["shape"]
            k = groups * words * 8
            tiles = math.prod(buffers[p["TileExpert"]["name"]]["shape"])
            books = {
                pointers(v)["Book"]["name"]
                for v in originals.values()
                if v["symbol"] == "exact_e8p_int4_kernel"
                and v["name"].startswith("flash_batch_m8/")
                and pointers(v)["P"]["name"] == p["P"]["name"]
            }
            if len(books) != 1:
                raise ValueError("Missing certified exact expert book")
            planes = {
                role: dict(kind="buffer", name=scratch(f"FlashDecodeE8P_K{k}_{role}", dtype, shape))
                for role, dtype, shape in [
                    ("Low", "u32", [CAPACITY * 10, k // 8]),
                    ("High", "u32", [CAPACITY * 10, k // 8]),
                    ("Sum", "i32", [CAPACITY * 10]),
                ]
            }
            bound = remap(
                {
                    "AS": p["AS"],
                    "Book": dict(kind="buffer", name=next(iter(books))),
                    "Counts": p["Counts"],
                    "Offsets": p["Offsets"],
                    "Out": p["C"],
                    "P": p["P"],
                    "TC": p["TileCount"],
                    "TE": p["TileExpert"],
                    "TR": p["TileRow"],
                    "WS": p["WS"],
                    **planes,
                }
            )
            emit(
                old,
                name,
                args=bound,
                build=lambda e=e, n=n, k=k, tiles=tiles: e8p_exact_grouped.__wrapped__(
                    CAPACITY * 10, e, tiles, n, k, block_m=16, dynamic_rows=True
                ),
                key=f"exact-wide-{n}-{k}",
                multiplier=10,
            )
            emit(
                old,
                name + "/a8_split",
                args=dict(A=remap(p)["A"], **planes),
                build=lambda k=k: split_a8.__wrapped__(CAPACITY * 10, k, dynamic_rows=True),
                key=f"split-{k}",
                multiplier=10,
            )
        else:
            emit(old, name, multiplier=multiplier)
        canonical[suffix] = additions[-1]

    # A local medium-row tile keeps the validated 16-row geometry. It uses
    # canonical operands; routing and expert launches always use live counts.
    for old in shared:
        suffix = old["name"].removeprefix("flash_dynamic_m128/")
        medium = originals.get("flash_dynamic_m16/" + suffix)
        if medium is None or medium["module"]["sha256"] == old["module"]["sha256"]:
            continue
        p = pointers(medium)
        if set(p) == {"A", "Output", "Weight"} and buffers[p["Weight"]["name"]]["dtype"] == "bf16":
            continue
        if set(p) == {"A", "C", "W"} and "hc_" in p["W"]["name"] and "_down." in p["W"]["name"]:
            continue
        if set(p) == {"Activated", "Down"}:
            continue
        if "Counts" in p or not any(
            a["ctype"] == "ctypes.c_int32" and a["value"] == "m"
            for a in host(medium)["ordered_arguments"]
        ):
            continue
        canonical_args = remap(pointers(old))
        if set(p) != set(canonical_args):
            raise ValueError("Medium tile changes canonical operands")
        emit(medium, f"{PREFIX}/{suffix}/medium", canonical_args)

    small_mapping = {}
    aliases = {
        "Output": "C",
        "Weight": "W",
        "Out": "C",
        "Partials": "Down",
        "TC": "TileCount",
        "TE": "TileExpert",
        "TR": "TileRow",
    }
    section_names = sorted(
        {
            k["name"].rsplit("/", 1)[0].removeprefix("flash_batch_m8/")
            for k in package["kernels"]
            if k["name"].startswith("flash_batch_m8/")
            and k["name"].rsplit("/", 1)[1].startswith("k")
            and "/private/" not in k["name"]
        }
    )
    aligned, omitted = [], []
    for section in section_names:

        def nodes(b):
            prefix = f"flash_batch_m{b}/{section}/"
            return sorted(
                [
                    k
                    for name, k in originals.items()
                    if name.startswith(prefix)
                    and name.removeprefix(prefix).startswith("k")
                    and "/" not in name.removeprefix(prefix)
                ],
                key=lambda k: int(k["name"].rsplit("/k", 1)[1]),
            )

        pairs, skip = align_sections(nodes(8), nodes(128), pointers)
        aligned.extend(pairs)
        omitted.extend(skip)
        for s, l, extra in pairs:
            a, b = pointers(s), pointers(l)
            if extra is not None:
                b.update(pointers(extra))
                if "Hi" in a:
                    b["Out"] = b["Mixed"]
                else:
                    b["Gate"], b["GateScale"] = b["Weight"], b["WeightScale"]
            for role, arg in a.items():
                if buffers[arg["name"]].get("data"):
                    continue
                target = b.get(role, b.get(aliases.get(role)))
                if role == "Out" and "Activated" in b:
                    target = b["Activated"]
                if role in ("Output", "Partials") and (
                    s["symbol"] == "bf16_single_kernel"
                    and "hc_" in a["Weight"]["name"]
                    or "Partials" in a
                ):
                    target = dict(name=parts)
                if target is not None:
                    new = arena(target["name"])
                    previous = small_mapping.setdefault(arg["name"], new)
                    if previous != new:
                        raise ValueError(
                            f"Conflicting small-row operand mapping: {arg['name']} {previous} {new}"
                        )

    # The compact expert representation has separate transient nibble planes.
    for name, b in list(buffers.items()):
        if name.startswith("ComputeE8P_R80_"):
            shape = [CAPACITY * 10, *b["shape"][1:]]
            small_mapping[name] = scratch(
                name.replace("ComputeE8P_R80_", "FlashDecodeE8P_"), b["dtype"], shape
            )
    for name in list(small_mapping):
        if name.startswith("BatchM8_") and small_mapping[name] == ARENA + "Scratch366":
            small_mapping[name] = parts

    for s, l, extra in aligned:
        suffix = l["name"].removeprefix("flash_batch_m128/")
        slot = f"{PREFIX}/{suffix}"
        dyn = originals.get(s["name"].replace("flash_batch_m8/", "flash_dynamic_m8/"))
        p = remap(pointers(s), small_mapping)
        if any(a["name"].startswith(("BatchM8_", "ComputeQ4_R8_")) for a in p.values()):
            raise ValueError(f"Unmapped small-row workspace at {s['name']}: {p}")
        if s["symbol"] == "hc_up13_vector_project_kernel":
            source = reader.read(s["source"]).decode()
            bases = set(re.findall(r"hc_expand13_four\([^;]+?,\s*\(uint\)(\d+)\)", source))
            if len(bases) != 1:
                raise ValueError("Missing certified BF16 cache exponent")
            base = int(next(iter(bases)))
            rank = buffers[p["Hi"]["name"]]["shape"][1] * 64
            patches = math.prod(buffers[p["Patches"]["name"]]["shape"]) if "Patches" in p else 0
            p.setdefault(
                "Patches",
                dict(kind="buffer", name=p["Hi"]["name"].removesuffix("_Hi") + "_Patches"),
            )
            emit(
                s,
                slot + "/small",
                p,
                lambda rank=rank, base=base, patches=patches: hc_up13_mix.__wrapped__(
                    8, 2560, rank, dynamic_rows=True, base_exp=base, patch_count=patches
                ),
                f"hc13-{rank}-{base}-{patches}",
            )
        elif s["symbol"] == "exact_e8p_int4_kernel":
            e, groups, n, words = buffers[p["P"]["name"]]["shape"]
            k = groups * words * 8
            tiles = (CAPACITY * 10 + 7) // 8 + min(512, CAPACITY * 10)
            emit(
                s,
                slot + "/small",
                p,
                lambda e=e, n=n, k=k, tiles=tiles: e8p_exact_grouped.__wrapped__(
                    CAPACITY * 10, e, tiles, n, k, block_m=8, dynamic_rows=True
                ),
                f"exact-{n}-{k}",
                10,
            )
            split = originals[s["name"] + "/a8_split"]
            emit(
                split,
                slot + "/small/a8_split",
                remap(pointers(split), small_mapping),
                lambda k=k: split_a8.__wrapped__(CAPACITY * 10, k, dynamic_rows=True),
                f"split-{k}",
                10,
            )
            for tag, rows in [("pair", 2), ("four", 4), ("oct", 8)]:
                fixed = originals[s["name"].replace("flash_batch_m8/", f"flash_batch_m{rows}/")]
                emit(fixed, slot + "/" + tag, p)
                fixed_split = originals[fixed["name"] + "/a8_split"]
                emit(
                    fixed_split,
                    slot + "/" + tag + "/a8_split",
                    remap(pointers(split), small_mapping),
                )
        elif set(p) == {"A", "AS", "Out", "S", "W"}:
            n, width = buffers[p["W"]["name"]]["shape"]
            k = width * 4
            dt = "float32" if buffers[p["Out"]["name"]]["dtype"] == "f32" else "float16"
            for tag, rows in [("tiny", 2), ("quad", 4), ("small", 8)]:
                emit(
                    s,
                    slot + "/" + tag,
                    p,
                    lambda rows=rows, n=n, k=k, dt=dt: q4_gemv.__wrapped__(
                        rows, n, k, output_dtype=dt, dynamic_rows=True
                    ),
                    f"q4-{tag}-{n}-{k}-{dt}",
                )
            # Common shapes eliminate inner-loop row predicates without
            # duplicating the forward plan or changing canonical operands.
            for tag, rows in [("pair", 2), ("four", 4), ("oct", 8)]:
                emit(
                    s,
                    slot + "/" + tag,
                    p,
                    lambda rows=rows, n=n, k=k, dt=dt: q4_gemv.__wrapped__(
                        rows, n, k, output_dtype=dt
                    ),
                    f"q4-fixed-{rows}-{n}-{k}-{dt}",
                    symbolic=False,
                )
        elif (
            set(p) == {"A", "Output", "Weight", "TokenScale", "WeightScale"}
            and buffers[p["Weight"]["name"]]["shape"][0] <= 128
        ):
            n, k = buffers[p["Weight"]["name"]]["shape"]
            dt = "float32" if buffers[p["Output"]["name"]]["dtype"] == "f32" else "float16"
            for tag, rows in [("tiny", 2), ("quad", 4), ("small", 8)]:
                emit(
                    s,
                    slot + "/" + tag,
                    p,
                    lambda rows=rows, n=n, k=k, dt=dt: int8_small_projection.__wrapped__(
                        rows, n, k, dt, dynamic_rows=True
                    ),
                    f"w8-narrow-{rows}-{n}-{k}-{dt}",
                )
        elif (
            set(p) == {"A", "Output", "Weight"}
            and buffers[p["Weight"]["name"]]["dtype"] == "bf16"
            and buffers[p["Weight"]["name"]]["shape"][0] == 1
        ):
            k = buffers[p["Weight"]["name"]]["shape"][1]
            for tag, rows in [("tiny", 2), ("quad", 4), ("small", 8)]:
                emit(
                    s,
                    slot + "/" + tag,
                    p,
                    lambda rows=rows, k=k: scalar_bf16_projection.__wrapped__(
                        rows, k, dynamic_rows=True
                    ),
                    f"bf16-scalar-{rows}-{k}",
                )
        elif dyn is not None:
            p = remap(pointers(dyn), small_mapping)
            multiplier = (
                10
                if dynamics[dyn["name"]]["arguments"]
                and any(
                    old["args"][a["index"]].get("value") == 80
                    for a in dynamics[dyn["name"]]["arguments"]
                    for old in [dyn]
                )
                else 1
            )
            emit(dyn, slot + "/small", p, multiplier=multiplier)
        else:
            raise ValueError(f"Missing symbolic small-row implementation: {s['name']}")
    for old in omitted:
        # A marker is bound to the canonical identity; the native variant omits it.
        slot = f"{PREFIX}/" + old["name"].removeprefix("flash_batch_m128/")
        marker = copy.deepcopy(canonical[old["name"].removeprefix("flash_batch_m128/")])
        marker["name"] = slot + "/small_skip"
        additions.append(marker)
        c = copy.deepcopy(next(c for c in contracts if c["name"] == slot))
        c["name"] = marker["name"]
        contracts.append(c)

    # Reuse single-request causal code with the already dynamic request table.
    for old in package["kernels"]:
        if old["name"].startswith("flash_private_m128/"):
            emit(old, old["name"].replace("flash_private_m128/", "flash_decode_state/"))

    from kernels.model import decode_packet, ple
    from kernels.model.independent_rows import independent_rows

    private_stride = 48 * len(data["metadata"]["batch_layout"]["state_columns"])
    table = "FlashBatchPointers"
    buffers[table]["shape"] = [CAPACITY * (private_stride + decode_packet.IO_COLUMNS)]
    pointer = dict(kind="buffer", name=table)
    embedding = dict(kind="buffer", name=ARENA + "M1_Embedding")
    ple_input = dict(kind="buffer", name=ARENA + "M1_Ple")
    init = pointers(originals["flash_dynamic_m128/begin/k0"])
    hidden = remap(init)["Residual"]
    vision = data["metadata"].get("vision") is not None
    feature_rows = buffers["Features"]["shape"][0] if vision else 1
    ring = buffers["MtpHiddenRing"]["shape"][0] if data["metadata"].get("mtp") else 1
    emit(
        shared[0],
        PREFIX + "/packet/inputs",
        dict(Pointers=pointer, Embedding=embedding, Ple=ple_input),
        lambda: decode_packet.inputs.__wrapped__(
            private_stride, data["metadata"]["max_context"], feature_rows, vision
        ),
        "packet-inputs",
    )
    emit(
        shared[0],
        PREFIX + "/packet/end",
        dict(Pointers=pointer, Hidden=hidden),
        lambda: decode_packet.end.__wrapped__(
            private_stride, ring, bool(data["metadata"].get("mtp"))
        ),
        "packet-end",
    )
    emit(
        shared[0],
        PREFIX + "/packet/outputs",
        dict(
            Pointers=pointer,
            Logits=dict(kind="buffer", name=ARENA + "Logits"),
            Selected=dict(kind="buffer", name=ARENA + "Selected"),
        ),
        lambda: decode_packet.outputs.__wrapped__(private_stride, data["metadata"]["vocab"]),
        "packet-outputs",
    )
    for index, build in enumerate(
        [lambda: ple.ple_conv.__wrapped__(1, 10240), lambda: ple.ple_history.__wrapped__(1, 10240)]
    ):
        old = originals[f"flash_batch_m128/layer1/private/k{index}"]
        p = pointers(old)
        roles = {
            name: (
                "weight"
                if buffers[arg["name"]].get("data")
                else "private"
                if arg["name"] == "State_ple"
                else "row",
                "State_ple" if arg["name"] == "State_ple" else name,
            )
            for name, arg in p.items()
        }
        fn, identities = independent_rows(
            build(),
            CAPACITY,
            roles,
            {"State_ple": 0},
            row_stride=decode_packet.IO_COLUMNS,
            dynamic=True,
        )
        bound = {name: arg for name, arg in remap(p).items() if name in identities}
        bound["Pointers"] = dict(
            kind="buffer_slice", name=table, offset=(CAPACITY * private_stride + 9) * 8
        )
        emit(old, f"flash_decode_ple/k{index}", bound, lambda fn=fn: fn, f"packet-ple-{index}")

    removed = {
        k["name"]
        for k in package["kernels"]
        if k["name"].startswith(("flash_batch_m", "flash_dynamic_m", "flash_private_m"))
    }
    package["kernels"] = [k for k in package["kernels"] if k["name"] not in removed] + additions
    package["dynamic_batch_kernels"] = [
        c for c in package["dynamic_batch_kernels"] if c["name"] not in removed
    ] + contracts
    package["plan_kernel_contracts"] = sorted(
        set(package["plan_kernel_contracts"]) - removed | {k["name"] for k in additions}
    )
    used = {a["name"] for k in package["kernels"] for a in k["args"] if "name" in a}
    discarded = {
        n
        for n in buffers
        if n.startswith(
            ("BatchM", "ComputeQ4_R", "ComputeE8P_R", ARENA, "FlashDecodeQ4_", "FlashDecodeE8P_")
        )
        and n not in used
        and not buffers[n].get("data")
    }
    data["metadata"]["buffers"] = [b for n, b in buffers.items() if n not in discarded]
    for n in discarded:
        data["buffer_scopes"].pop(n, None)
    package["buffer_contracts"] = [
        {k: v for k, v in b.items() if k != "data"} for b in data["metadata"]["buffers"]
    ]
    path = root / "lib/model.so"
    path.unlink()
    shutil.copyfile(model_library(), path)
    package["execution"]["library"]["sha256"] = file_hash(path)
    digest = compact_package(root, package, destination / "cache/packages")
    shutil.rmtree(root)
    data["execution_package"] = digest
    write_json(destination / "cache/model.json", data)
    write_json(
        output / "results.json",
        dict(
            capacity=CAPACITY,
            definitions=len(compiled),
            bindings=len(additions),
            removed_bindings=len(removed),
            removed_buffers=len(discarded),
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--model-output", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--compile-cache", type=Path)
    args = p.parse_args()
    if args.compile_cache:
        from tools.model.publication import seed_compile_cache

        seed_compile_cache(args.compile_cache, args.output / "cache/0.1.15")
    with atomic_model(args.model_output) as staging:
        upgrade(args.model.resolve(), staging, args.output)


if __name__ == "__main__":
    main()
