"""Losslessly repack E8P banks and compile exact A8/INT4 expert execution.

The model keeps one compressed bank. Large tiles and unsupported bases retain
INT8 arithmetic over the same K64 storage; small tiles use exact nibble products.
"""

import argparse
import copy
import hashlib
import math
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np
from safetensors.numpy import save_file

from tools.model.compact import compact_package, header, write_shard
from tools.model.flash_next.compiler import pointer_bindings, replace_binding
from tools.model.publication import (
    AssetReader,
    atomic_model,
    clone_model,
    file_hash,
    link_or_copy,
    load_model,
    source_path,
    write_json,
)
from tools.operators.abi import parse_host, validate_parameter_count
from tools.quantization.vq import (
    e8p_nibble_table,
    e8p_repack_k64,
    e8p_selector_table,
    e8p_selector_supports_parity_xor,
)


def live_rows(program):
    if program == "decode":
        return 1
    match = re.fullmatch(r"(?:prefill|verify|mtp_warm|flash_batch|flash_dynamic)_m(\d+)", program)
    if match is None:
        raise ValueError(f"Unregistered expert program: {program}")
    return int(match[1])


def row_tile(source):
    # Existing exported modules contain this explicit count bound. Fail closed
    # if its spelling/geometry changes, rather than guess routing tile strides.
    matches = {int(m) for m in re.findall(r"min\((8|16|32|64), \(", source)}
    if len(matches) != 1:
        raise ValueError("Cannot establish routed expert row stride")
    return matches.pop()


def repack_weights(model, destination, data, package, index):
    root = model / "cache/weights"
    target = destination / "cache/weights"
    buffers = {b["data"]["tensor"]: b for b in data["metadata"]["buffers"] if b.get("data")}
    contracts = {b["name"]: b for b in package["buffer_contracts"]}
    files = {}
    for name, filename in index["weight_map"].items():
        files.setdefault(filename, []).append(name)
    total = 0
    with tempfile.TemporaryDirectory(dir=target, prefix=".repack-") as temporary:
        temporary = Path(temporary)
        for number, (filename, names) in enumerate(files.items()):
            if not any(n.endswith("_exps.weight_packed") for n in names):
                continue
            source = source_path(root.resolve(), filename)
            begin, specs = header(source)
            entries, layouts = [], {}
            for name in names:
                b, info = buffers[name], specs[name]
                first, end = info["data_offsets"]
                if b["shape"] != info["shape"] or b["layout"] != specs.get("__metadata__", {}).get(
                    f"orin.layout.{name}"
                ):
                    raise ValueError("Weight source contract mismatch")
                segments = [(source, begin + first, end - first)]
                if name.endswith("_exps.weight_packed"):
                    if b["dtype"] != "u16" or len(b["shape"]) != 4 or b["shape"][-1] != 16:
                        raise ValueError("Expected original K128 E8P bank")
                    e, g, n, _ = b["shape"]
                    raw = np.memmap(
                        source, mode="r", offset=begin + first, dtype=np.uint16, shape=(e, g, n, 16)
                    )
                    output = temporary / f"{total}.bin"
                    original, transformed = hashlib.sha256(), hashlib.sha256()
                    with output.open("xb") as stream:
                        for expert in range(e):
                            original.update(raw[expert].tobytes())
                            block = e8p_repack_k64(raw[expert : expert + 1]).tobytes()
                            transformed.update(block)
                            stream.write(block)
                    del raw
                    if original.hexdigest() != b["data"]["sha256"]:
                        raise ValueError(f"Invalid expert weight identity: {name}")
                    b["shape"] = [e, g * 2, n, 8]
                    b["data"]["sha256"] = transformed.hexdigest()
                    contracts[b["name"]]["shape"] = b["shape"].copy()
                    segments = [(output, 0, end - first)]
                    total += 1
                    print("repacked", name, flush=True)
                entries.append(
                    dict(
                        name=name,
                        dtype=info["dtype"],
                        shape=b["shape"],
                        segments=segments,
                        sha256=b["data"]["sha256"],
                    )
                )
                layouts[f"orin.layout.{name}"] = b["layout"]
            new_file = f"exact-e8p-{number:05d}.safetensors"
            write_shard(target / new_file, entries, layouts)
            (target / filename).unlink()
            for name in names:
                index["weight_map"][name] = new_file
            for item in entries:
                path = item["segments"][0][0]
                if path.is_relative_to(temporary):
                    path.unlink()
    return total


def upgrade(model, destination, report):
    from kernels.model.e8p_exact import e8p_exact_grouped, split_a8
    from kernels.model.e8p_prefill import e8p_prefill_grouped, split_prefill_a8
    from kernels.model.integer_vq import integer_vq_grouped
    from tools.model.package import model_library
    from tools.operators.common import configure, export_kernel

    configure()
    data, origin, package = load_model(model)
    if data["architecture"] != "flash_next" or data["compute_policy"] != "int8_quality":
        raise ValueError("Expected native Flash Next INT8-quality model")
    operator = clone_model(model, destination, origin)
    reader, hosts, exports, compiled, books = AssetReader(origin), {}, {}, {}, {}
    buffers = {b["name"]: b for b in data["metadata"]["buffers"]}
    index_path = destination / "cache/weights/model.safetensors.index.json"
    import json

    index = json.loads(index_path.read_text())

    def tensor(name):
        b = buffers[name]
        path = source_path(
            (model / "cache/weights").resolve(), index["weight_map"][b["data"]["tensor"]]
        )
        begin, spec = header(path)
        info = spec[b["data"]["tensor"]]
        first, end = info["data_offsets"]
        with path.open("rb") as stream:
            stream.seek(begin + first)
            raw = stream.read(end - first)
        if hashlib.sha256(raw).hexdigest() != b["data"]["sha256"]:
            raise ValueError("E8P table identity mismatch")
        return np.frombuffer(raw, np.uint32).reshape(b["shape"])

    def add_buffer(name, dtype, shape, value=None):
        if name in buffers:
            raise ValueError(f"Duplicate exact E8P storage: {name}")
        contract = dict(
            name=name,
            dtype=dtype,
            shape=shape,
            layout="native_contiguous",
            alignment=256,
            access="read" if value is not None else "read_write",
        )
        b = copy.deepcopy(contract)
        if value is not None:
            digest = hashlib.sha256(value.tobytes()).hexdigest()
            file = f"{name}.safetensors"
            save_file(
                {name: value},
                destination / "cache/weights" / file,
                metadata={f"orin.layout.{name}": "native_contiguous"},
            )
            index["weight_map"][name] = file
            index.setdefault("metadata", {})["total_size"] += value.nbytes
            data["metadata"]["weight_bytes"] += value.nbytes
            b["data"] = dict(tensor=name, sha256=digest)
        data["metadata"]["buffers"].append(b)
        data["buffer_scopes"][name] = "weights" if value is not None else "workspace"
        package["buffer_contracts"].append(contract)
        buffers[name] = b
        return dict(kind="buffer", name=name)

    def book(pointer):
        name = pointer["name"]
        if name not in books:
            raw = tensor(name)
            if raw.shape == (1, 256, 1):
                selector = raw[0]
            elif raw.shape == (1, 512, 4):
                basis = (
                    np.ascontiguousarray(raw[0, :256, [0, 2]].T)
                    .view(np.int8)
                    .reshape(256, 8)
                    .astype(np.int16)
                    - 1
                ).astype(np.int8)
                selector = e8p_selector_table(basis)
            elif raw.shape == (1, 256, 2):
                selector = e8p_selector_table(raw.view(np.int8).reshape(256, 8))
            else:
                raise ValueError("Unsupported E8P table")
            if e8p_selector_supports_parity_xor(selector):
                value = e8p_nibble_table(selector)[None]
                digest = hashlib.sha256(value.tobytes()).hexdigest()
                label = f"W_e8p_nibble_{digest}"
                books[name] = (
                    dict(kind="buffer", name=label)
                    if label in buffers
                    else add_buffer(label, "u32", [1, 256, 1], value)
                )
            else:
                books[name] = None
        return books[name]

    def compile_kernel(key, build):
        if key not in compiled:
            target = operator / "exact-e8p" / key
            export_kernel(build(), report / "aot" / key)
            assets = {}
            for field, filename in [
                ("module", "kernel.cubin"),
                ("source", "kernel.cu"),
                ("host_abi", "host.txt"),
            ]:
                path = target / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                link_or_copy(report / "aot" / key / filename, path)
                assets[field] = dict(file=str(path.relative_to(operator)), sha256=file_hash(path))
            host = parse_host((target / "host.txt").read_text())[0]
            validate_parameter_count((target / "kernel.cu").read_text(), host)
            compiled[key], exports[key] = host, assets
        return compiled[key], exports[key]

    additions, fast, fallback = [], 0, 0
    dynamic = {d["name"]: d for d in package.get("dynamic_batch_kernels", [])}
    # Tables must be read before changing the index to the new weight shards.
    for i, kernel in enumerate(package["kernels"]):
        packed = next(
            (
                buffers[a["name"]]
                for a in kernel["args"]
                if a.get("name", "").endswith("_exps.weight_packed")
            ),
            None,
        )
        if packed is None:
            continue
        old_host = hosts.setdefault(
            kernel["host_abi"]["sha256"], parse_host(reader.read(kernel["host_abi"]).decode())[0]
        )
        pointers = pointer_bindings(kernel, old_host)
        if set(pointers) != {
            "A",
            "AS",
            "Book",
            "C",
            "Counts",
            "Offsets",
            "P",
            "TileCount",
            "TileExpert",
            "TileRow",
            "WS",
        }:
            raise ValueError("Unexpected grouped expert pointer contract")
        rows = live_rows(kernel["name"].split("/")[0]) * 10
        e, g, n, words = packed["shape"]
        if words != 16:
            raise ValueError("Expected K128 source bank")
        k = g * 128
        source = reader.read(kernel["source"]).decode()
        bm = row_tile(source)
        # The accepted short-prefill launch is output-tile first.
        group_n = kernel["name"].startswith("prefill_m512/")
        tiles = kernel["grid"][1 if group_n else 0]
        nibble = (
            book(pointers["Book"])
            if bm in (8, 16) and kernel["name"].split("/")[0] in ("decode", "prefill_m512")
            else None
        )
        if nibble is not None:
            prefill_fast = kernel["name"].startswith("prefill_m512/") and bm == 16
            suffix = "-kmajor-p4" if prefill_fast else ""
            workspace = f"E8P_{kernel['name'].split('/')[0]}_K{k}"
            args = {}
            for role, dtype, shape in [
                ("Low", "u32", [k // 64, rows, 8] if prefill_fast else [rows, k // 8]),
                ("High", "u32", [k // 64, rows, 8] if prefill_fast else [rows, k // 8]),
                ("Sum", "i32", [rows]),
            ]:
                label = f"{workspace}_{role}"
                args[role] = (
                    dict(kind="buffer", name=label)
                    if label in buffers
                    else add_buffer(label, dtype, shape)
                )
            host, assets = compile_kernel(
                f"split-r{rows}-k{k}{suffix}",
                lambda rows=rows, k=k, prefill_fast=prefill_fast: (
                    split_prefill_a8(rows, k) if prefill_fast else split_a8(rows, k)
                ),
            )
            split = replace_binding(
                dict(name=kernel["name"] + "/a8_split"), host, assets, dict(A=pointers["A"], **args)
            )
            additions.append(split)
            if kernel["name"].startswith("flash_dynamic_"):
                # The splitter covers the fixed capacity arena. Only Counts
                # select live assignments; inactive planes are never consumed.
                package["dynamic_batch_kernels"].append(
                    dict(
                        name=split["name"],
                        capacity=rows // 10,
                        grid=[dict(op="constant", value=v) for v in split["grid"]],
                        arguments=[],
                    )
                )
            bound = dict(
                **args,
                P=pointers["P"],
                Book=nibble,
                WS=pointers["WS"],
                AS=pointers["AS"],
                Counts=pointers["Counts"],
                Offsets=pointers["Offsets"],
                TE=pointers["TileExpert"],
                TR=pointers["TileRow"],
                TC=pointers["TileCount"],
                Out=pointers["C"],
            )
            host, assets = compile_kernel(
                f"exact-r{rows}-bm{bm}-n{n}-k{k}-t{tiles}{suffix}",
                lambda rows=rows, e=e, tiles=tiles, n=n, k=k, bm=bm, prefill_fast=prefill_fast: (
                    e8p_prefill_grouped(rows, e, tiles, n, k)
                    if prefill_fast
                    else e8p_exact_grouped(rows, e, tiles, n, k, block_m=bm)
                ),
            )
            package["kernels"][i] = replace_binding(kernel, host, assets, bound)
            fast += 1
        else:
            shape = buffers[pointers["Book"]["name"]]["shape"]
            selector, short = shape == [1, 256, 1], shape in ([1, 256, 1], [1, 512, 4])
            bn = math.ceil(n / kernel["grid"][0 if group_n else 1])
            opts = dict(
                kind="e8p",
                shared_table=True,
                block_m=bm,
                block_n=bn,
                shortbook=short,
                selectorbook=selector,
                byte_permute=short and not selector,
                num_stages=1 if short and bm <= 32 else 2,
                group_n=group_n,
                packed_k=64,
            )
            if "e8p_mma(" in source:
                opts.update(
                    warp_decode=True,
                    split_k=2 if k >= 2048 else 1,
                    parity_xor=book(pointers["Book"]) is not None,
                )
            host, assets = compile_kernel(
                f"fallback-r{rows}-bm{bm}-n{n}-k{k}-t{tiles}-s{shape[1]}-{shape[2]}-warp{int(opts.get('warp_decode', False))}-xor{int(opts.get('parity_xor', False))}",
                lambda rows=rows, e=e, tiles=tiles, n=n, k=k, opts=opts: integer_vq_grouped(
                    rows, e, tiles, n, k, **opts
                ),
            )
            package["kernels"][i] = replace_binding(kernel, host, assets, pointers)
            fallback += 1
        if kernel["name"] in dynamic:
            contract = dynamic[kernel["name"]]
            if contract["arguments"]:
                raise ValueError("Expected capacity-bounded dynamic expert ABI")
            contract["grid"] = [dict(op="constant", value=v) for v in package["kernels"][i]["grid"]]
    package["kernels"].extend(additions)
    package["plan_kernel_contracts"] = sorted(
        set(package.get("plan_kernel_contracts", []))
        | {k["name"] for k in additions}
        | {k["name"].removesuffix("/a8_split") for k in additions}
    )
    if fast == 0 or fallback == 0:
        raise ValueError("Missing exact or fallback expert consumers")
    banks = repack_weights(model, destination, data, package, index)
    library = operator / package["execution"]["library"]["file"]
    library.unlink()
    shutil.copyfile(model_library(), library)
    package["execution"]["library"]["sha256"] = file_hash(library)
    write_json(index_path, index)
    digest = compact_package(operator, package, destination / "cache/packages")
    shutil.rmtree(operator)
    data["execution_package"] = digest
    write_json(destination / "cache/model.json", data)
    write_json(
        report / "upgrade.json",
        dict(
            execution_package=digest,
            fast=fast,
            fallback=fallback,
            repacked_banks=banks,
            compiled_variants=len(compiled),
            weight_bytes=data["metadata"]["weight_bytes"],
            arithmetic_unchanged=True,
            single_compressed_bank=True,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine", type=Path)
    args = parser.parse_args()
    with atomic_model(args.model_output, args.engine) as staging:
        upgrade(args.model.resolve(strict=True), staging, args.output)
    print("EXACT E8P PACKAGE READY", args.model_output, flush=True)


if __name__ == "__main__":
    main()
