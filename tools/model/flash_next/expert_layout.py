"""Lossless E8P expert-bank layouts and routed row-stride validation."""

import hashlib
from pathlib import Path
import re
import tempfile

import numpy as np

from tools.model.compact import header, write_shard
from tools.model.publication import source_path
from tools.quantization.vq import e8p_repack_k64


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
