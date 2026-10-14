"""Build one draft row frame; chronological QSA and capture retain their owners."""

import copy
import math
import re

CAPACITY = 512
PREFIX = "draft_rows"


def primitive_row_multiplier(label, input_rows):
    if label.startswith("rotate-a8-ffn-"):
        return 10
    if label.startswith("dense-") and input_rows == 32:
        return 4
    return 1


def causal_bindings(original, api, width, reference_indices):
    """Find chronological operations by their ABI, independent of local fusion."""
    candidates = sorted(
        (
            (int(k["name"].rsplit("/k", 1)[1]), k)
            for k in original.values()
            if re.fullmatch(f"mtp_warm_m{width}/layer48/k[0-9]+", k["name"])
        ),
        key=lambda item: item[0],
    )
    found, start = [], 0
    for index in reference_indices:
        reference = original[f"mtp_warm_m8/layer48/k{index}"]
        names = api.get(reference)[0]
        for offset in range(start, len(candidates)):
            kernel = candidates[offset][1]
            if api.get(kernel)[0] == names:
                found.append(kernel)
                start = offset + 1
                break
        else:
            raise ValueError(f"Missing chronological draft ABI at {width}/{index}: {names}")
    return found


def publish(data, package, compiler):
    import torch
    from kernels.model import flash_mtp
    from kernels.model.bf16_single import bf16_single
    from kernels.model.e8p_exact import split_a8, e8p_exact_grouped
    from kernels.model.moe import route_metadata
    from tools.model.flash_next.prepare import DTYPES
    from kernels.model.hc_bf16_cache import hc_up13_mix
    from tools.model.flash_next.batching import shape_planner, kind
    from tools.model.flash_next.dynamic_batch import factory
    from tools.model.flash_next.mixed_verify import Interfaces
    from tools.model.flash_next.mixed_draft import PRIVATE
    from tools.model.flash_next.decode_policy import decode_shapes
    from tools.model.flash_next.decode_tuning import DecodeTuner, make_prim

    metadata, buffers = data["metadata"], compiler.buffers
    original = {k["name"]: k for k in package["kernels"]}
    api = Interfaces(compiler.reader.base)
    planner, logical = shape_planner(
        metadata, buffers, original, compiler.output, draft=True, templates_only=True
    )
    planner.position = planner.capacity - 8
    plan = planner.plan(8)
    slots, additions, contracts = {}, [], []
    fixed_storage, tile_storage = set(), set()
    routing = {}
    for (_, args), label in zip(plan["ops"], plan["labels"]):
        label = label.split(":")[-1]
        if label.startswith("histogram-"):
            routing.update(IDs=args[0], Counts=args[1], RelativeSlot=args[2])
        elif label.startswith("offsets-"):
            routing.update(
                Counts=args[0], RowOffsets=args[1], TileOffsets=args[2], TileCount=args[3]
            )
            fixed_storage.update(t.untyped_storage()._cdata for t in args)
        elif label.startswith("tiles-"):
            routing.update(TileExpert=args[2], TileRow=args[3])
            tile_storage.update(t.untyped_storage()._cdata for t in args[2:])
    shared_old_names = {
        a["name"]
        for k in original.values()
        if k["name"].startswith(
            ("flash_draft_m8/begin/", "flash_draft_m8/pre/", "flash_draft_m8/post/")
        )
        for a in k["args"]
        if "name" in a
    }
    shared_old_names = {
        n.removeprefix("DraftBatchM8_") for n in shared_old_names if n.startswith("DraftBatchM8_")
    }
    next_buffer = 0
    retained = []

    def allocate(name, dtype, shape):
        buffers[name] = dict(
            name=name,
            dtype=dtype,
            shape=shape,
            layout="native_contiguous",
            alignment=256,
            access="read_write",
        )
        data["buffer_scopes"][name] = "workspace"
        return dict(kind="buffer", name=name)

    def tensor(t):
        nonlocal next_buffer
        if not isinstance(t, torch.Tensor):
            raise ValueError("Draft row operands must be tensors")
        storage = t.untyped_storage()._cdata
        if storage in logical:
            name = logical[storage]
            offset = t.storage_offset() * t.element_size()
            return (
                dict(kind="buffer_slice", name=name, offset=offset)
                if offset
                else dict(kind="buffer", name=name)
            )
        if storage not in slots:
            count = t.untyped_storage().nbytes() // t.element_size()
            size = (
                count
                if storage in fixed_storage
                else (CAPACITY * 10 + 7) // 8 + 512
                if storage in tile_storage
                else count * CAPACITY // 8
            )
            if storage not in fixed_storage and storage not in tile_storage and count % 8:
                raise ValueError(
                    f"Draft scratch at {label} must scale with rows: {t.shape}, {count}"
                )
            name = f"DraftRows_Scratch{next_buffer}"
            next_buffer += 1
            slots[storage] = allocate(name, DTYPES[str(t.dtype).removeprefix("torch.")][0], [size])
            retained.append(t)
        result = dict(slots[storage])
        offset = t.storage_offset() * t.element_size()
        if offset:
            result.update(kind="buffer_slice", offset=offset)
        return result

    for key, name in [
        ("embedding", "DraftRows_Embedding"),
        ("condition", "DraftRows_Condition"),
        ("residual", "DraftRows_Hidden"),
    ]:
        t = plan[key]
        slots[t.untyped_storage()._cdata] = allocate(name, "f16", [CAPACITY, *t.shape[1:]])

    def emit(stage, old, pointers, build, key, *, multiplier=1, name=None, symbolic=True):
        if name is None:
            index = sum(
                bool(re.fullmatch(f"{PREFIX}/{stage}/k[0-9]+", k["name"])) for k in additions
            )
            name = f"{PREFIX}/{stage}/k{index}"
        k, launch = compiler.emit(
            old,
            name,
            pointers,
            CAPACITY,
            build=build,
            key=key,
            multiplier=multiplier,
            symbolic=symbolic,
        )
        additions.append(k)
        contracts.append(launch)
        return name

    tuner = DecodeTuner(compiler.output / "draft-tuning", getattr(compiler, "tuning_cache", None))
    common_books = [
        compiler.pointers(k)["Book"]
        for k in original.values()
        if k["symbol"] == "exact_e8p_int4_kernel" and k["name"].startswith("flash_decode/")
    ]
    if not common_books or len({b["name"] for b in common_books}) != 1:
        raise ValueError("Draft and target must share the certified exact E8P book")
    exact = next(k for k in original.values() if k["symbol"] == "exact_e8p_int4_kernel")
    private_index = 0
    frame_aliases = {}
    for index, ((_, args), label) in enumerate(zip(plan["ops"], plan["labels"])):
        if index >= plan["body_count"]:
            break
        label = label.split(":")[-1]
        if kind(label):
            source = original[f"mtp_warm_m8/layer48/k{PRIVATE[private_index]}"]
            names, _ = api.get(source)
            names = [n for n in names if n != "Coordinates"]
            if len(names) != len(args):
                raise ValueError(f"Causal draft ABI differs at {label}: {names}")
            p = compiler.pointers(source)
            for name, t in zip(names, args):
                if name in p and p[name]["name"] in shared_old_names:
                    frame_aliases[p[name]["name"]] = tensor(t)
            private_index += 1
            continue
        # The first five operations condition the draft input. All subsequent
        # projections before QSA completes belong to pre, including index-QK.
        stage = "begin" if index < 5 else "pre" if private_index < len(PRIVATE) else "post"
        if label.startswith(("offsets-", "tiles-")):
            continue
        if label.startswith("dense-quant-"):
            values = [tensor(args[0]), tensor(args[2]), tensor(args[3])]
        else:
            values = [
                None if label.startswith(("expert-gu-", "expert-down-")) and i == 3 else tensor(t)
                for i, t in enumerate(args)
            ]
        weight = next((v for v in values if v and buffers[v["name"]].get("data")), None)
        old = exact
        build, bindings, key, multiplier, projection_slot = None, None, label, 1, None
        if label.startswith("histogram-"):
            bindings = {name: tensor(t) for name, t in routing.items()}

            def build():
                return route_metadata.__wrapped__(CAPACITY, block_m=8, dynamic_rows=True)

            key = "route-m8-draft"
        elif label.startswith("mtp-norm-"):
            width = args[0].shape[-1]

            def build(width=width):
                return flash_mtp.norm.__wrapped__(CAPACITY, width, dynamic_rows=True)

            bindings = dict(zip(("X", "Gamma", "Out"), values))
        elif label.startswith("mtp-fuse-"):

            def build():
                return flash_mtp.fuse.__wrapped__(CAPACITY, 2560, dynamic_rows=True)

            bindings = dict(zip(("Embedding", "Hidden", "Out"), values))
        elif label.startswith("dense-quant-"):
            width = args[0].shape[-1]
            old = next(
                k
                for k in original.values()
                if k["name"].startswith("flash_decode/")
                and set(compiler.pointers(k)) == {"X", "Q", "S"}
                and math.prod(buffers[compiler.pointers(k)["Q"]["name"]]["shape"]) == 1024 * width
            )
            bindings = dict(X=values[0], Q=values[1], S=values[2])
        elif label.startswith("hc-down-partial-"):

            def build():
                return bf16_single.__wrapped__(320, 2560, 4, rows=CAPACITY, dynamic_rows=True)

            bindings = dict(zip(("A", "Weight", "Output"), values))
        elif label.startswith("hc-up-mix-"):
            assert weight is not None
            hi = "HC13_" + weight["name"] + "_Hi"
            certificate = next(
                k for k in original.values() if any(a.get("name") == hi for a in k["args"])
            )
            cache = compiler.pointers(certificate)
            source = compiler.reader.read(certificate["source"]).decode()
            bases = set(re.findall(r"hc_expand13_four\([^;]+?,\s*\(uint\)(\d+)\)", source))
            if len(bases) != 1:
                raise ValueError("Missing draft BF16 cache exponent certificate")
            base = int(next(iter(bases)))
            patches = (
                math.prod(buffers[cache["Patches"]["name"]]["shape"]) if "Patches" in cache else 0
            )

            def build(base=base, patches=patches):
                return hc_up13_mix.__wrapped__(
                    CAPACITY, 2560, 320, dynamic_rows=True, base_exp=base, patch_count=patches
                )

            bindings = dict(
                A=values[0],
                Normed=values[2],
                Out=values[3],
                Hi=cache["Hi"],
                Lo=cache["Lo"],
            )
            if "Patches" in cache:
                bindings["Patches"] = cache["Patches"]
            key = f"hc13-320-{base}-{patches}"
        elif label.startswith(("dense-a8-", "dense-small-a8-")):
            n, width = args[1].shape
            dtype = str(args[-1].dtype).removeprefix("torch.")
            bindings = dict(zip(("A", "Weight", "WeightScale", "TokenScale", "Output"), values))
            slot = emit(
                stage,
                old,
                bindings,
                lambda n=n, width=width, dtype=dtype: make_prim(
                    dict(
                        kind="w8", rows=CAPACITY, outputs=n, inputs=width, dtype=dtype, dynamic=True
                    ),
                    dict(block_m=16, block_n=64, block_k=128),
                ),
                f"w8-{n}-{width}-{dtype}",
            )
            # Integer tiles share the target's offline tuning policy and rounding.
            for shape in decode_shapes():
                rows = CAPACITY if shape.dynamic else shape.rows
                _, config = tuner.compile("w8", rows, n, width, dtype, dynamic=shape.dynamic)
                ranges = [(33, CAPACITY)] if shape.dynamic else shape.ranges()
                if shape.dynamic:
                    ranges = [(a, b if b < 128 else CAPACITY) for a, b in shape.ranges()]
                for first, last in ranges:
                    emit(
                        stage,
                        old,
                        bindings,
                        lambda rows=rows, n=n, width=width, dtype=dtype, dynamic=shape.dynamic, config=config: (
                            make_prim(
                                dict(
                                    kind="w8",
                                    rows=rows,
                                    outputs=n,
                                    inputs=width,
                                    dtype=dtype,
                                    dynamic=dynamic,
                                ),
                                config,
                            )
                        ),
                        f"w8-{rows}-{shape.dynamic}-{n}-{width}-{config}",
                        name=f"{slot}/rows/{first}-{last}",
                        symbolic=shape.dynamic,
                    )
            continue
        elif label.startswith(("expert-gu-", "expert-down-")):
            e, groups, n, words = buffers[values[1]["name"]]["shape"]
            width = groups * words * 8
            low = allocate(f"DraftRows_Planes{index}Low", "u32", [CAPACITY * 10, width // 8])
            high = allocate(f"DraftRows_Planes{index}High", "u32", [CAPACITY * 10, width // 8])
            sums = allocate(f"DraftRows_Planes{index}Sum", "i32", [CAPACITY * 10])
            projection_slot = f"{PREFIX}/{stage}/k" + str(
                sum(bool(re.fullmatch(f"{PREFIX}/{stage}/k[0-9]+", k["name"])) for k in additions)
            )
            emit(
                stage,
                old,
                dict(A=values[0], Low=low, High=high, Sum=sums),
                lambda width=width: split_a8.__wrapped__(CAPACITY * 10, width, dynamic_rows=True),
                f"split-{width}",
                multiplier=10,
                name=projection_slot + "/a8_split",
            )
            bindings = dict(
                Low=low,
                High=high,
                Sum=sums,
                P=values[1],
                Book=common_books[0],
                WS=values[4],
                AS=values[5],
                Counts=values[6],
                Offsets=values[7],
                TE=values[8],
                TR=values[9],
                TC=values[10],
                Out=values[11],
            )
            tile = 8
            count = (CAPACITY * 10 + tile - 1) // tile + 512

            def build(n=n, width=width, count=count):
                return e8p_exact_grouped.__wrapped__(
                    CAPACITY * 10, 512, count, n, width, block_m=8, dynamic_rows=True
                )

            multiplier = 10
            key = f"expert-m8-n{n}-k{width}-draft"
        else:
            multiplier = primitive_row_multiplier(label, args[0].shape[0])

            def build(label=label, multiplier=multiplier):
                return factory(
                    label, CAPACITY if label.startswith("rotate-a8-ffn-") else CAPACITY * multiplier
                )

            # Parameter order comes from the new primitive rather than a CUDA
            # source filename or the historical kernel's reordered launch ABI.
            prim = build()
            names = [prim.buffer_map[p].name for p in prim.params if p.dtype == "handle"]
            bindings = dict(zip(names, values))
        if bindings is None or (build is None and old is exact):
            raise ValueError(f"Missing draft row primitive: {label}")
        emit(stage, old, bindings, build, key, multiplier=multiplier, name=projection_slot)

    if private_index != len(PRIVATE):
        raise ValueError("Incomplete draft causal boundary mapping")
    # Rebind chronological and independent causal programs to the same row frame.
    for k in package["kernels"]:
        match = re.fullmatch(r"flash_draft_m8/(?:private|warm_m\d+/private)/k\d+", k["name"])
        if not match:
            continue
        clone = copy.deepcopy(k)
        clone["name"] = clone["name"].replace("flash_draft_m8/", PREFIX + "/")
        for arg in clone["args"]:
            source = arg.get("name", "").removeprefix("DraftBatchM8_")
            if source in frame_aliases and arg.get("name", "").startswith("DraftBatchM8_"):
                arg.update(frame_aliases[source])
        additions.append(clone)
    # Single warm QSA owns its full chronological segment. Only interface buffers
    # cross into the shared frame; all private intermediate state stays intact.
    for width in [1, 2, 3, 4, 5, 6, 7, 8, 16, 128, 512]:
        for j, source in enumerate(causal_bindings(original, api, width, PRIVATE)):
            k = copy.deepcopy(source)
            k["name"] = f"{PREFIX}/causal_m{width}/k{j}"
            ref = compiler.pointers(original[f"mtp_warm_m8/layer48/k{PRIVATE[j]}"])
            current = compiler.pointers(k)
            for role, arg in current.items():
                source = ref.get(role, {}).get("name")
                if source in frame_aliases:
                    actual = arg["name"]
                    for value in k["args"]:
                        if value.get("name") == actual:
                            value.update(frame_aliases[source])
            additions.append(k)
    # Target and draft heads share one row program and the same Q4 output
    # projection. Only the immutable HC parameters and input condition differ.
    for k in list(package["kernels"]):
        if not k["name"].startswith("flash_decode/head/"):
            continue
        head = copy.deepcopy(k)
        head["name"] = head["name"].replace("flash_decode/head/", PREFIX + "/head/")
        for arg in head["args"]:
            name = arg.get("name", "")
            if name == "FlashDecode_Residual":
                arg["name"] = "DraftRows_Hidden"
            elif name.startswith("W_output_hc_"):
                arg["name"] = "D" + name
            elif name.startswith("HC13_W_output_hc_"):
                arg["name"] = name.replace("HC13_W_", "HC13_DW_", 1)
        hp = compiler.pointers(head)
        if {"A", "Hi", "Lo", "Patches", "Normed", "Out"} == hp.keys():
            certificate = next(
                k
                for k in original.values()
                if any(v.get("name") == hp["Hi"]["name"] for v in k["args"])
            )
            source = compiler.reader.read(certificate["source"]).decode()
            bases = set(re.findall(r"hc_expand13_four\([^;]+?,\s*\(uint\)(\d+)\)", source))
            if len(bases) != 1:
                raise ValueError("Missing draft head BF16 cache certificate")
            base = int(next(iter(bases)))
            patches = math.prod(buffers[hp["Patches"]["name"]]["shape"])
            bm = 32 if "/rows/33-" in head["name"] else 8
            result, launch = compiler.emit(
                head,
                head["name"],
                hp,
                128,
                build=lambda base=base, patches=patches, bm=bm: hc_up13_mix.__wrapped__(
                    128,
                    2560,
                    320,
                    dynamic_rows=True,
                    base_exp=base,
                    patch_count=patches,
                    block_m=bm,
                ),
                key=f"hc13-draft-head-{base}-{patches}-{bm}",
            )
        else:
            result, launch = compiler.emit(head, head["name"], hp, 128)
        additions.append(result)
        contracts.append(launch)
    selector = copy.deepcopy(original["mtp_head_m1/body/k8"])
    selector["name"] = PREFIX + "/selections"
    pair = compiler.pointers(selector)["Pair"]["name"]
    for arg in selector["args"]:
        if arg.get("name") == pair:
            arg["name"] = "FlashDecode_Selected"
    additions.append(selector)
    for width in [1, 2, 3, 4, 5, 6, 7, 8, 16, 128, 512]:
        capture = copy.deepcopy(
            original[f"mtp_head_m{width}/body/k{9 if width <= 8 or width == 512 else 10}"]
        )
        capture["name"] = f"{PREFIX}/capture_m{width}"
        for arg in capture["args"]:
            if arg.get("name") == f"DraftM{width}_HC":
                arg["name"] = "DraftRows_Hidden"
        additions.append(capture)
    package["kernels"].extend(additions)
    package["dynamic_batch_kernels"].extend(contracts)
    package["plan_kernel_contracts"] = sorted(
        set(package["plan_kernel_contracts"]) | {k["name"] for k in additions}
    )
    spec = data["metadata"]["mtp"]
    spec["batch_draft_profiles"] = [8]
    for warm in spec["warm_plans"]:
        warm["head_program"] = "mtp_head"
    spec["batch_programs"] = [
        p for p in spec["batch_programs"] if not p.startswith("flash_draft_m")
    ]
    spec["batch_programs"].extend(
        f"{PREFIX}/{stage}" for stage in ["begin", "pre", "post", "head", "private"]
    )
    spec["batch_programs"].extend(f"{PREFIX}/warm_m{n}/private" for n in range(1, 9))
