"""Publish one decode-row template with autotuned AOT row specializations.

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
from tools.model.flash_next.decode_policy import (
    CAPACITY,
    SPECIALIZATIONS,
    decode_shapes,
    selection_name,
)
from tools.model.flash_next.decode_tuning import DecodeTuner
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

PREFIX = "flash_decode"
ARENA = "FlashDecode_"


def share_dynamic_fallbacks(kernels, contracts):
    """Reuse the tuned dynamic cubin as the canonical slot's definition too."""
    positions = {k["name"]: i for i, k in enumerate(kernels)}
    launches = {c["name"]: i for i, c in enumerate(contracts)}
    groups = {}
    for kernel in kernels:
        if "/rows/" in kernel["name"]:
            slot, bounds = kernel["name"].split("/rows/")
            first, last = map(int, bounds.split("-"))
            groups.setdefault(slot, []).append((first, last, kernel))
    reused = 0
    for slot, choices in groups.items():
        ordered = sorted(choices, key=lambda v: (v[0], v[1]))
        if any(a[1] >= b[0] for a, b in zip(ordered, ordered[1:])):
            raise ValueError(
                f"Overlapping decode row choices at {slot}: {[(a, b) for a, b, _ in ordered]}"
            )
        if slot not in positions or ordered[0][0] != 1 or ordered[-1][1] != CAPACITY:
            continue
        if any(a[1] + 1 != b[0] for a, b in zip(ordered, ordered[1:])):
            continue
        # The final range belongs to the single runtime-row fallback.
        first, _, fallback = ordered[-1]
        if first != SPECIALIZATIONS[-1] + 1:
            continue
        kernel = copy.deepcopy(fallback)
        kernel["name"] = slot
        kernels[positions[slot]] = kernel
        launch = copy.deepcopy(contracts[launches[fallback["name"]]])
        launch["name"] = slot
        contracts[launches[slot]] = launch
        reused += 1
    return reused


def upgrade(model, destination, output, tuning_cache=None):
    import tilelang
    from kernels.model.rows import explicit_rows
    from kernels.model.hyperconnection import hc_down_finish, hc_injection
    from kernels.model.bf16_single import bf16_single
    from kernels.model.hc_bf16_cache import hc_up13_mix
    from kernels.model.flash_next import scalar_bf16_projection
    from kernels.model.e8p_exact import e8p_exact_grouped, split_a8
    from kernels.model import moe
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
    reader, hosts, compiled, sources = AssetReader(origin), {}, {}, {}
    staged_reader = AssetReader(root)
    buffers = {b["name"]: b for b in data["metadata"]["buffers"]}
    additions, contracts = [], []
    tuner = DecodeTuner(output, tuning_cache)
    shapes = decode_shapes()
    tuned, expert_tiles = {}, {}

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

    def emit(
        old, name, args=None, build=None, key=None, multiplier=1, *, symbolic=True, kernel=None
    ):
        bound = remap(pointers(old)) if args is None else args
        if build is not None or kernel is not None:
            if key not in compiled:
                print("COMPILE", key, flush=True)
                if kernel is None:
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
                else:
                    generated = kernel
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
        hosts[assets["host_abi"]["sha256"]] = abi
        if assets["source"]["sha256"] not in sources:
            sources[assets["source"]["sha256"]] = staged_reader.read(assets["source"]).decode()
        k = replace_binding(
            dict(old, name=name),
            abi,
            assets,
            bound,
            dimensions=dict(m=CAPACITY * multiplier, rows=CAPACITY * multiplier),
        )
        prototype = re.search(
            r"\b" + re.escape(k["symbol"]) + r"\s*\(([^)]*)\)",
            sources[assets["source"]["sha256"]],
        )
        if prototype is None:
            raise ValueError("Missing generated CUDA access prototype")
        declarations = prototype[1].split(",")
        if len(declarations) != len(k["args"]):
            raise ValueError("Mismatched CUDA access prototype")
        k["readonly_arguments"] = [
            "*" not in declaration or bool(re.search(r"\bconst\b", declaration))
            for declaration in declarations
        ]
        c = contract(k, abi, CAPACITY, multiplier)
        additions.append(k)
        contracts.append(c)
        return k

    def tuned_projection(old, slot, bound, kind, n, k, dtype="float16"):
        for shape in shapes:
            key = f"tuned-{kind}-{shape.key}-{n}-{k}-{dtype}"
            if key not in tuned:
                tuned[key] = tuner.compile(kind, shape.rows, n, k, dtype, dynamic=shape.dynamic)
            kernel, _ = tuned[key]
            for lo, hi in shape.ranges():
                if lo <= hi:
                    emit(
                        old,
                        selection_name(slot, lo, hi),
                        bound,
                        key=key,
                        symbolic=shape.dynamic,
                        kernel=kernel,
                    )

    hc_caches = {}
    for binding in originals.values():
        if binding["symbol"] == "hc_up13_vector_project_kernel":
            hc_caches.setdefault(pointers(binding)["Hi"]["name"], binding)

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
        elif set(p) == {"A", "C", "W"} and "hc_" in p["W"]["name"] and "_inject." in p["W"]["name"]:
            n, k = buffers[p["W"]["name"]]["shape"]
            emit(
                old,
                name,
                args=dict(A=remap(p)["A"], W=p["W"], Out=remap(p)["C"]),
                build=lambda n=n, k=k: hc_injection.__wrapped__(CAPACITY, n, k, dynamic_rows=True),
                key=f"hc-injection-{n}-{k}",
            )
        elif (
            set(p) == {"A", "Output", "Weight"}
            and buffers[p["Weight"]["name"]]["dtype"] == "bf16"
            and buffers[p["Weight"]["name"]]["shape"][0] == 1
        ):
            k = buffers[p["Weight"]["name"]]["shape"][1]
            emit(
                old,
                name,
                args=remap(p),
                build=lambda k=k: scalar_bf16_projection.__wrapped__(
                    CAPACITY, k, dynamic_rows=True
                ),
                key=f"bf16-scalar-{k}",
            )
        elif set(p) == {"Expert", "SlotMap", "Prob", "Shared", "SharedGate", "Output"}:
            emit(
                old,
                name,
                args=remap(p),
                build=lambda: moe.moe_combine.__wrapped__(
                    CAPACITY, 2560, CAPACITY * 10, dynamic_rows=True
                ),
                key="moe-combine",
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
            # The selected expert tile can be eight or sixteen rows. Both
            # layouts share these bounded maps, sized for the smaller tile.
            tile_capacity = (CAPACITY * 10 + 7) // 8 + min(e, CAPACITY * 10)
            for role in ("TileExpert", "TileRow"):
                buffers[arena(p[role]["name"])]["shape"] = [tile_capacity]
            tiles = math.prod(buffers[p["TileExpert"]["name"]]["shape"])
            books = {
                pointers(v)["Book"]["name"]
                for v in originals.values()
                if v["symbol"] == "exact_e8p_int4_kernel"
                and buffers[pointers(v)["Book"]["name"]]["shape"] == [1, 256, 1]
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
        canonical[suffix] = next(k for k in reversed(additions) if k["name"] == name)

    # Every row count uses the canonical operands and helpers. Only integer
    # matrix tiles are selected by the offline table; there are no helper bands.
    expert = next(k for k in canonical.values() if "exact-" in k["module"]["file"])
    expert_p = pointers(expert)
    e, groups, n, words = buffers[expert_p["P"]["name"]]["shape"]
    k = groups * words * 8
    for shape in shapes:
        key = f"tuned-expert-{shape.key}-{n}-{k}"
        tuned[key] = tuner.compile("expert", shape.rows, n, k, dynamic=shape.dynamic, experts=e)
        expert_tiles[shape.key] = tuned[key][1]["block_m"]

    skipped = []
    for old in shared:
        suffix = old["name"].removeprefix("flash_dynamic_m128/")
        slot = f"{PREFIX}/{suffix}"
        p = remap(pointers(old))
        if set(p) == {"A", "C", "W"} and "hc_" in p["W"]["name"] and "_up." in p["W"]["name"]:
            section, index = suffix.rsplit("/k", 1)
            mix_suffix = f"{section}/k{int(index) + 1}"
            mix = pointers(canonical[mix_suffix])
            if set(mix) != {"Mixed", "Normed", "Up"}:
                raise ValueError("HC up must consume the adjacent mix")
            hi_name = "HC13_" + p["W"]["name"] + "_Hi"
            certificate = hc_caches.get(hi_name)
            if certificate is None:
                raise ValueError("Missing certified lossless HC cache")
            cache = pointers(certificate)
            source = reader.read(certificate["source"]).decode()
            bases = set(re.findall(r"hc_expand13_four\([^;]+?,\s*\(uint\)(\d+)\)", source))
            if len(bases) != 1:
                raise ValueError("Missing certified BF16 cache exponent")
            base = int(next(iter(bases)))
            rank = buffers[hi_name]["shape"][1] * 64
            patches = (
                math.prod(buffers[cache["Patches"]["name"]]["shape"]) if "Patches" in cache else 0
            )
            bound = dict(
                A=p["A"],
                Hi=cache["Hi"],
                Lo=cache["Lo"],
                Patches=cache.get(
                    "Patches", dict(kind="buffer", name=hi_name.removesuffix("_Hi") + "_Patches")
                ),
                Normed=mix["Normed"],
                Out=mix["Mixed"],
            )
            emit(
                old,
                selection_name(slot, 1, CAPACITY),
                bound,
                lambda rank=rank, base=base, patches=patches: hc_up13_mix.__wrapped__(
                    CAPACITY,
                    2560,
                    rank,
                    dynamic_rows=True,
                    base_exp=base,
                    patch_count=patches,
                ),
                f"hc13-{rank}-{base}-{patches}",
            )
            finish_suffix = f"{section}/k{int(index) - 1}"
            if set(pointers(canonical[finish_suffix])) != {"Out", "Partials"}:
                raise ValueError("HC up must follow its partial reduction")
            skipped.append(mix_suffix)
        elif {"P", "Counts", "TE", "TR", "Out"} <= pointers(canonical[suffix]).keys():
            bound = pointers(canonical[suffix])
            e, groups, n, words = buffers[bound["P"]["name"]]["shape"]
            k = groups * words * 8
            for shape in shapes:
                key = f"tuned-expert-{shape.key}-{n}-{k}"
                if key not in tuned:
                    tuned[key] = tuner.compile(
                        "expert",
                        shape.rows,
                        n,
                        k,
                        dynamic=shape.dynamic,
                        experts=e,
                        block_m=expert_tiles[shape.key],
                    )
                kernel, _ = tuned[key]
                for lo, hi in shape.ranges():
                    emit(
                        old,
                        selection_name(slot, lo, hi),
                        bound,
                        key=key,
                        multiplier=10,
                        symbolic=shape.dynamic,
                        kernel=kernel,
                    )
        elif set(p) == {"A", "AS", "Out", "S", "W"}:
            n, width = buffers[p["W"]["name"]]["shape"]
            dtype = "float32" if buffers[p["Out"]["name"]]["dtype"] == "f32" else "float16"
            tuned_projection(old, slot, p, "q4", n, width * 4, dtype)
        elif set(p) == {"IDs", "Counts", "RelativeSlot"}:
            section, index = suffix.rsplit("/k", 1)
            offsets_suffix, tiles_suffix = [
                f"{section}/k{int(index) + offset}" for offset in (1, 2)
            ]
            bound = {
                **p,
                **pointers(canonical[offsets_suffix]),
                **pointers(canonical[tiles_suffix]),
            }
            for shape in shapes:
                bm = expert_tiles[shape.key]
                for lo, hi in shape.ranges():
                    emit(
                        old,
                        selection_name(slot, lo, hi),
                        bound,
                        lambda bm=bm: moe.route_metadata.__wrapped__(
                            CAPACITY, block_m=bm, dynamic_rows=True
                        ),
                        f"route-metadata-{bm}",
                    )
            skipped.extend([offsets_suffix, tiles_suffix])
        elif set(p) in (
            {"Counts", "RowOffsets", "TileOffsets", "TileCount"},
            {"Counts", "TileOffsets", "TileExpert", "TileRow"},
        ):
            continue
        elif set(p) == {"A", "Output", "Weight", "TokenScale", "WeightScale"}:
            n, k = buffers[p["Weight"]["name"]]["shape"]
            dtype = "float32" if buffers[p["Output"]["name"]]["dtype"] == "f32" else "float16"
            if p["Weight"]["name"].endswith("ffn_up_shexp.weight_weight"):
                continue
            if p["Weight"]["name"].endswith("ffn_gate_shexp.weight_weight"):
                section, index = suffix.rsplit("/k", 1)
                up_suffix, act_suffix = [f"{section}/k{int(index) + offset}" for offset in (1, 2)]
                up, act = pointers(canonical[up_suffix]), pointers(canonical[act_suffix])
                if (
                    set(act) != {"Gate", "Up", "Output"}
                    or act["Gate"] != p["Output"]
                    or act["Up"] != up["Output"]
                ):
                    raise ValueError("Shared FFN fusion changes adjacent operands")
                bound = dict(
                    A=p["A"],
                    Gate=p["Weight"],
                    Up=up["Weight"],
                    GateScale=p["WeightScale"],
                    UpScale=up["WeightScale"],
                    TokenScale=p["TokenScale"],
                    Output=act["Output"],
                )
                tuned_projection(old, slot, bound, "swiglu", n, k)
                skipped.extend([up_suffix, act_suffix])
            else:
                tuned_projection(old, slot, p, "w8", n, k, dtype)

    # Rotation quantizes each source once, then produces the exact planes in
    # expert order. Down consumes the expert SwiGLU rows without a pack pass.
    from kernels.model.rotation_a8 import rotate_activation_a8

    operands = {
        suffix: remap(pointers(old))
        for suffix, old in ((k["name"].removeprefix("flash_dynamic_m128/"), k) for k in shared)
    }
    for suffix, projection in canonical.items():
        bound = pointers(projection)
        if not {"Low", "High", "Sum", "P", "AS"} <= bound.keys():
            continue
        source = operands[suffix]
        section = suffix.rsplit("/k", 1)[0] + "/k"
        dispatches = [
            (name, p)
            for name, p in operands.items()
            if name.startswith(section)
            and set(p) == {"A", "D", "DS", "IDs", "RelativeSlot", "RowOffsets", "Scale", "SlotMap"}
            and p["D"] == source["A"]
        ]
        dispatch = dispatches[0] if dispatches else None
        if len(dispatches) > 1:
            raise ValueError("Ambiguous expert dispatch producer")
        input_arg = dispatch[1]["A"] if dispatch else source["A"]
        rotations = [
            (name, p)
            for name, p in operands.items()
            if name.startswith(section)
            and set(p) == {"Q", "S", "Signs", "X"}
            and p["Q"] == input_arg
        ]
        if len(rotations) != 1:
            raise ValueError("Missing exact expert rotation producer")
        rotation_suffix, producer = rotations[0]
        fused = dict(producer, **{role: bound[role] for role in ("Low", "High", "Sum")})
        if dispatch:
            fused.update(
                {
                    role: dispatch[1][role]
                    for role in ("IDs", "RelativeSlot", "RowOffsets", "DS", "SlotMap")
                }
            )
            skipped.append(dispatch[0])
        width, rows = (2560, CAPACITY) if dispatch else (640, CAPACITY * 10)
        emit(
            canonical[rotation_suffix],
            selection_name(f"{PREFIX}/{rotation_suffix}", 1, CAPACITY),
            fused,
            lambda rows=rows, width=width, dispatch=bool(dispatch): (
                rotate_activation_a8.__wrapped__(
                    rows,
                    width,
                    swiglu=not dispatch,
                    dynamic_rows=True,
                    output_layout="e8p_scatter" if dispatch else "e8p_rows",
                )
            ),
            f"expert-rotate-{'up' if dispatch else 'down'}",
            multiplier=1 if dispatch else 10,
        )
        skipped.append(suffix + "/a8_split")

    for suffix in skipped:
        slot = f"{PREFIX}/{suffix}"
        marker = copy.deepcopy(next(k for k in additions if k["name"] == slot))
        marker["name"] = selection_name(slot, 1, CAPACITY, skip=True)
        additions.append(marker)
        c = copy.deepcopy(next(c for c in contracts if c["name"] == slot))
        c["name"] = marker["name"]
        contracts.append(c)

    # Wrap the exact compact single-request convolution, including its in-place
    # history update. Its pointer-table scratch offsets are certified by the
    # existing compiled wrapper; no private tensor becomes a shared row arena.
    from kernels.model.independent_rows import independent_rows
    from kernels.operators.op08_gdn_conv_prep import gdn_conv_prep

    columns = {name: i for i, name in enumerate(data["metadata"]["batch_layout"]["state_columns"])}
    for old in package["kernels"]:
        if not old["name"].startswith("flash_private_m128/"):
            continue
        name = old["name"].replace("flash_private_m128/", "flash_decode_state/")
        layer = int(name.split("/")[1].removeprefix("layer"))
        if layer % 4 != 3 and name.endswith("/k0"):
            source = reader.read(old["source"]).decode()
            offsets = {}
            for role in ("Q", "K", "V"):
                matches = re.findall(
                    r"\bvoid\* "
                    + role
                    + r"_row = orin_row_pointer\(\([^;]+?\+ \(uint64_t\)(\d+)\)\);",
                    source,
                )
                if len(matches) != 1:
                    raise ValueError("Missing certified private convolution scratch offset")
                offsets[role] = int(matches[0])
                extent = (16 if role in ("Q", "K") else 48) * 128 * 2
                if offsets[role] + extent > math.prod(buffers["FlashPrivateScratch"]["shape"]):
                    raise ValueError("Convolution scratch exceeds its private arena")
            roles = dict(
                X=("row", "X"),
                W=("weight", "W"),
                History=("private", "State_conv"),
                lengths=("private", "Length"),
                positions=("private", "Position"),
                positions_out=("private", "PositionOut"),
                **{role: ("private", "FlashPrivateScratch") for role in offsets},
            )
            fn = gdn_conv_prep.__wrapped__(
                B=1,
                tokens=1,
                compact_decode=True,
                inplace_history=True,
                normalize_round_fp16=False,
                conv_product_round_fp16=False,
                weight_dtype="float32",
            )
            fn, _ = independent_rows(
                fn,
                CAPACITY,
                roles,
                columns,
                row_stride=48 * len(columns),
                dynamic=True,
                offsets=offsets,
            )
            fn = fn.with_attr("global_symbol", "rows_gdn_conv")
            emit(
                old,
                name,
                args=remap(pointers(old)),
                build=lambda fn=fn: fn,
                key="gdn-compact-inplace",
                symbolic=True,
            )
        else:
            emit(old, name)

    from kernels.model import decode_packet, ple

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

    shared_fallbacks = share_dynamic_fallbacks(additions, contracts)
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
            specializations=[shape.rows for shape in shapes if not shape.dynamic],
            tuned_shapes=len(tuner.results),
            shared_dynamic_fallbacks=shared_fallbacks,
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
    p.add_argument("--tuning-cache", type=Path, help="Reusable offline SM87 autotune results")
    args = p.parse_args()
    if args.compile_cache:
        from tools.model.publication import seed_compile_cache

        seed_compile_cache(args.compile_cache, args.output / "cache/0.1.15")
    with atomic_model(args.model_output) as staging:
        upgrade(args.model.resolve(), staging, args.output, args.tuning_cache)


if __name__ == "__main__":
    main()
