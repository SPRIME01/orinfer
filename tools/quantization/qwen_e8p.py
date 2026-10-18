"""Resumable two-stage E8P Q4A8 encoding of original dense Qwen3.5 weights.

Each source shard is consumed independently. GPU ownership is released between
bounded row blocks, and completed blocks are checksummed. Tied embeddings and
output heads share one Q4 bank. Vision matrices also use Q4; sensitive
projections, positional tables and small parameters retain W8 or BF16.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
from safetensors.numpy import save_file

from tools.model.compact import write_shard
from tools.model.publication import file_hash
from tools.quantization.flash_next import atomic_json as write_json
from tools.model.safetensors_source import Source, local_header
from tools.quantization.flash_next_aux import int8_rows
from tools.quantization.qwen35 import gpu_lock


def policy(name, info):
    shape = info["shape"]
    if (
        info["dtype"] != "BF16"
        or len(shape) != 2
        or shape[1] % 128
        or min(shape) < 128
        or name == "model.visual.pos_embed.weight"
        or "norm" in name
    ):
        return "original"
    if name.startswith("mtp.") or name.endswith(
        ("linear_attn.out_proj.weight", "self_attn.o_proj.weight")
    ):
        return "w8"
    return "e8p-q4"


def encode(args):
    source_dir, out = args.source_dir.resolve(), args.output.resolve()
    if source_dir == out:
        raise ValueError("Source and encoded directories must differ")
    config = json.loads((source_dir / "config.json").read_text())
    if config.get("model_type") != "qwen3_5" or config.get("quantization_config"):
        raise ValueError("Expected original floating dense Qwen3.5 checkpoint")
    t = config["text_config"]
    if (t.get("hidden_size"), t.get("intermediate_size"), t.get("num_hidden_layers")) not in (
        (2048, 6144, 24),
        (5120, 17408, 64),
    ):
        raise ValueError("Unsupported dense Qwen geometry")
    origin = json.loads((source_dir / "source.json").read_text())
    source = Source(source_dir / "model.safetensors.index.json", directory=source_dir)
    out.mkdir(parents=True, exist_ok=True)
    payloads = out / ".payloads"
    payloads.mkdir(exist_ok=True)
    implementation = hashlib.sha256()
    for name in (
        "qwen_e8p.py",
        "e8p_q4_gpu.py",
        "e8p_q4.py",
        "e8p_gpu.py",
        "e8p.py",
        "vq.py",
        "rotation.py",
    ):
        implementation.update(Path(__file__).with_name(name).read_bytes())
    implementation.update((Path(__file__).parents[2] / "kernels/model/e8p_encode.py").read_bytes())
    contract = json.dumps(
        dict(
            source=origin,
            source_index=file_hash(source_dir / "model.safetensors.index.json"),
            encoder=implementation.hexdigest(),
            seed=20261002,
            coefficients=[4, 1],
            search="beam8",
            rows=args.rows,
            policy="qwen35-dense-e8p-q4a8-v2",
        ),
        sort_keys=True,
    )
    progress = out / "progress.json"
    state = (
        json.loads(progress.read_text())
        if progress.exists()
        else dict(
            contract=contract,
            complete=False,
            shards={},
            quantized={},
            parameters=0,
            started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
    )
    if state["contract"] != contract:
        raise ValueError("Quantization identity changed")
    write_json(progress, state)
    vision = config.get("vision_config", {})
    widths = sorted(
        {
            k
            for k in (
                t["hidden_size"],
                t["intermediate_size"],
                2 * t["hidden_size"],
                vision.get("hidden_size", 0),
                vision.get("intermediate_size", 0),
                4 * vision.get("hidden_size", 0),
            )
            if k > 0 and k % 128 == 0
        }
    )
    signs = {
        k: np.random.default_rng(20261002 + k).choice(np.array([-1, 1], np.int8), k) for k in widths
    }
    encoder = None
    from tools.operators.common import configure
    from tools.quantization.e8p_q4_gpu import Encoder
    from tools.quantization.vq import rotate

    shards = sorted(set(source.weight_map.values()))
    if args.shard and args.shard not in shards:
        raise ValueError("Unknown source shard")
    for shard in shards:
        if args.shard and shard != args.shard:
            continue
        if shard in state["shards"]:
            if file_hash(out / shard) != state["shards"][shard]["sha256"]:
                raise ValueError("Completed shard changed")
            continue
        path = source_dir / shard
        begin, header = local_header(path)
        entries, layouts, garbage = [], {}, []
        parameters = 0
        for name, info in sorted(header.items()):
            if name == "__metadata__":
                continue
            shape = info["shape"]
            parameters += math.prod(shape)
            kind = policy(name, info)
            first, last = info["data_offsets"]
            if kind == "original":
                layouts["orin.layout." + name] = "native_contiguous"
                entries.append(
                    dict(
                        name=name,
                        dtype=info["dtype"],
                        shape=shape,
                        segments=[(path, begin + first, last - first)],
                    )
                )
                continue
            n, k = math.prod(shape[:-1]), shape[-1]
            key = hashlib.sha256(name.encode()).hexdigest()
            pc, sc, checkpoint = (payloads / (key + s) for s in (".codes", ".scale", ".json"))
            saved = (
                json.loads(checkpoint.read_text())
                if checkpoint.exists()
                else dict(
                    name=name,
                    shape=shape,
                    kind=kind,
                    rows=0,
                    blocks=[],
                    compute_s=0.0,
                    sse=0.0,
                    norm=0.0,
                )
            )
            if (saved["name"], saved["shape"], saved["kind"]) != (name, shape, kind):
                raise ValueError("Matrix checkpoint changed")
            codes_per_row = k // 2 if kind == "e8p-q4" else k
            if saved["rows"]:
                with pc.open("rb") as pf, sc.open("rb") as sf:
                    for block in saved["blocks"]:
                        pf.seek(block["first"] * codes_per_row)
                        sf.seek(block["first"] * 2)
                        data = pf.read(block["rows"] * codes_per_row) + sf.read(block["rows"] * 2)
                        if hashlib.sha256(data).hexdigest() != block["sha256"]:
                            raise ValueError("Encoded block changed")
            raw = np.memmap(path, mode="r", offset=begin + first, dtype="<u2", shape=(n, k))
            with (
                pc.open("r+b" if pc.exists() else "w+b") as pf,
                sc.open("r+b" if sc.exists() else "w+b") as sf,
            ):
                pf.truncate(saved["rows"] * codes_per_row)
                sf.truncate(saved["rows"] * 2)
                pf.seek(0, 2)
                sf.seek(0, 2)
                for row in range(saved["rows"], n, args.rows):
                    rows = min(args.rows, n - row)
                    floating = (np.asarray(raw[row : row + rows]).astype(np.uint32) << 16).view(
                        np.float32
                    )
                    started = time.monotonic()
                    if kind == "e8p-q4":
                        with gpu_lock(args.gpu_lock):
                            if encoder is None:
                                configure()
                                encoder = Encoder()
                            fitted = encoder.refine(floating, encoder.fit(floating, signs[k]))
                        data, scales = fitted.packed.tobytes(), fitted.scales.tobytes()
                        reference = rotate(floating, signs[k])
                        error = fitted.dequantize() - reference
                    else:
                        codes, scale = int8_rows(floating)
                        data, scales = codes.tobytes(), scale.tobytes()
                        reference = floating
                        error = (
                            codes.astype(np.float32) * scale.astype(np.float32)[:, None] - reference
                        )
                    pf.write(data)
                    sf.write(scales)
                    pf.flush()
                    sf.flush()
                    os.fsync(pf.fileno())
                    os.fsync(sf.fileno())
                    saved["rows"] = row + rows
                    saved["compute_s"] += time.monotonic() - started
                    saved["sse"] += float(np.square(error).sum(dtype=np.float64))
                    saved["norm"] += float(np.square(reference).sum(dtype=np.float64))
                    saved["blocks"].append(
                        dict(first=row, rows=rows, sha256=hashlib.sha256(data + scales).hexdigest())
                    )
                    write_json(checkpoint, saved)
                    state["current"] = dict(
                        name=name, rows=saved["rows"], total_rows=n, compute_s=saved["compute_s"]
                    )
                    write_json(progress, state)
                    if row % (args.rows * 16) == 0:
                        print(
                            "ROWS",
                            name,
                            saved["rows"],
                            n,
                            "seconds",
                            round(saved["compute_s"], 2),
                            flush=True,
                        )
            del raw
            packed_name = name + (".packed" if kind == "e8p-q4" else ".int8")
            entries.extend(
                [
                    dict(
                        name=packed_name,
                        dtype="U32" if kind == "e8p-q4" else "I8",
                        shape=[*shape[:-1], k // 8] if kind == "e8p-q4" else shape,
                        segments=[(pc, 0, pc.stat().st_size)],
                    ),
                    dict(
                        name=name + ".scale",
                        dtype="F16",
                        shape=shape[:-1],
                        segments=[(sc, 0, sc.stat().st_size)],
                    ),
                ]
            )
            layouts["orin.layout." + packed_name] = (
                "e8p-q4-row-u32" if kind == "e8p-q4" else "native_contiguous"
            )
            layouts["orin.layout." + name + ".scale"] = "native_contiguous"
            state["quantized"][name] = {k: v for k, v in saved.items() if k != "blocks"}
            garbage.extend([pc, sc, checkpoint])
            print(
                "FITTED",
                name,
                kind,
                "relative_mse",
                saved["sse"] / max(saved["norm"], 1e-30),
                flush=True,
            )
        temporary = out / (shard + ".partial")
        write_shard(temporary, entries, dict(contract=contract, **layouts))
        temporary.replace(out / shard)
        state["shards"][shard] = dict(
            sha256=file_hash(out / shard), parameters=parameters, names=[e["name"] for e in entries]
        )
        state.pop("current", None)
        write_json(progress, state)
        for path in garbage:
            path.unlink()
        print("SHARD", len(state["shards"]), len(shards), shard, flush=True)
    if set(state["shards"]) != set(shards):
        return
    from tools.quantization.e8p import basis

    common = {f"orinfer.rotation.k{k}.signs": s for k, s in signs.items()}
    common["orinfer.e8p_q4.basis"] = basis()
    save_file(common, str(out / "quantization.safetensors"))
    mapping = {n: f for f, r in state["shards"].items() for n in r["names"]}
    mapping.update({n: "quantization.safetensors" for n in common})
    hashes = {f: r["sha256"] for f, r in state["shards"].items()}
    hashes["quantization.safetensors"] = file_hash(out / "quantization.safetensors")
    sizes = sum((out / f).stat().st_size - local_header(out / f)[0] for f in hashes)
    write_json(
        out / "model.safetensors.index.json",
        dict(metadata={"total_size": sizes, "orinfer.sha256": hashes}, weight_map=mapping),
    )
    for path in source_dir.iterdir():
        if (
            path.is_file()
            and path.suffix not in (".safetensors",)
            and path.name
            in (
                "source.json",
                "generation_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
                "merges.txt",
                "vocab.json",
                "preprocessor_config.json",
                "processor_config.json",
                "LICENSE",
                "README.md",
            )
        ):
            shutil.copyfile(path, out / path.name)
    config["quantization_config"] = dict(
        quant_method="orinfer_e8p_q4_int8",
        version=1,
        coefficients=[4, 1],
        rotation="signed-block128",
        scale="fp16-output-row",
        compute_dtype="int8_quality",
        basis_tensor="orinfer.e8p_q4.basis",
        layout="u32-low-coarse-high-residual",
        policy="qwen35-dense-e8p-q4a8-v2",
        embedding="e8p_q4",
        vision="e8p_q4",
        source=origin["repo"],
        source_revision=origin["revision"],
    )
    write_json(out / "config.json", config)
    state.update(
        complete=True,
        parameters=sum(r["parameters"] for r in state["shards"].values()),
        weight_bytes=sizes,
    )
    write_json(progress, state)
    print("COMPLETE", sizes, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--shard")
    p.add_argument("--rows", type=int, default=512)
    p.add_argument("--gpu-lock", type=Path, default=Path("artifacts/gpu-experiment.lock"))
    args = p.parse_args()
    if not 1 <= args.rows <= 1024:
        p.error("Row block must be in 1..1024")
    encode(args)


if __name__ == "__main__":
    main()
