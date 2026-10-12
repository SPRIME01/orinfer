"""Rebuild Flash Next projections without changing weight payloads.

Reads native safetensors weights and standalone/bundled execution assets. The
new immutable package remains bundled, with the same registered execution ABI.
"""

import argparse
import copy
import hashlib
import math
from pathlib import Path
import re
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.model.compact import compact_package
from tools.model.publication import (
    AssetReader,
    atomic_model,
    clone_model,
    file_hash,
    load_model,
    source_path,
    write_json,
)
from tools.model.flash_next.kernel_policy import (
    expert_tile_config,
    expert_projection_config,
    router_tile,
)
from tools.operators.abi import evaluate, parse_host, validate_parameter_count


def pointer_bindings(kernel, host, *, allow_scalars=False):
    """Retain the original offsets; exported argument order is authoritative."""
    if len(kernel["args"]) != len(host["ordered_arguments"]):
        raise ValueError("Original host ABI differs from the native package")
    pointers = {}
    for actual, argument in zip(kernel["args"], host["ordered_arguments"]):
        value = argument["value"]
        if allow_scalars and argument["ctype"] == "ctypes.c_int32" and actual["kind"] == "i32":
            continue
        if argument["ctype"] != "ctypes.c_void_p" or not value.endswith(".data_ptr()"):
            raise ValueError("Small static projection must use pointer-only ABI")
        if actual["kind"] not in ("buffer", "buffer_slice"):
            raise ValueError("Invalid projection pointer")
        pointers[value.removesuffix(".data_ptr()")] = copy.deepcopy(actual)
    return pointers


def replace_binding(original, host, assets, pointers, *, dimensions=None):
    launch = host["launch_expressions"]
    args = []
    for argument in host["ordered_arguments"]:
        value = argument["value"]
        if dimensions is not None and argument["ctype"] == "ctypes.c_int32":
            args.append(dict(kind="i32", value=evaluate(value, dimensions)))
            continue
        if argument["ctype"] != "ctypes.c_void_p" or not value.endswith(".data_ptr()"):
            raise ValueError("Unexpected rebuilt projection ABI")
        args.append(copy.deepcopy(pointers[value.removesuffix(".data_ptr()")]))
    return dict(
        name=original["name"],
        **assets,
        symbol=host["symbol"],
        grid=[evaluate(launch["gridDim" + axis], dimensions or {}) for axis in "XYZ"],
        block=[evaluate(launch["blockDim" + axis], dimensions or {}) for axis in "XYZ"],
        shared_memory_bytes=evaluate(launch["sharedMemBytes"], dimensions or {}),
        cooperative=False,
        args=args,
    )


def rotation_geometry(program, kernel, pointers, buffers):
    signs = buffers[pointers["Signs"]["name"]]
    if signs["dtype"] != "i8" or len(signs["shape"]) != 1:
        raise ValueError("Invalid rotation sign buffer")
    width = signs["shape"][0]
    # MTP's last-token head reads a slice of a multi-token hidden buffer.
    # Verification heads consume all rows; these two contracts differ.
    live_rows = 1 if program.startswith("mtp_head_m") else kernel["grid"][0]
    swiglu = signs["name"].endswith("ffn_down_exps.weight_signs")
    for argument, needed, dtype, item_bytes in (
        ("X", live_rows * width * (2 if swiglu else 1), "f16", 2),
        ("Q", live_rows * width, "i8", 1),
        ("S", live_rows, "f16", 2),
    ):
        b = buffers[pointers[argument]["name"]]
        capacity = math.prod(b["shape"]) * item_bytes
        if (
            b["dtype"] != dtype
            or pointers[argument].get("offset", 0) + needed * item_bytes > capacity
        ):
            raise ValueError("Rotation exceeds its workspace binding")
    return live_rows, width, swiglu


COMPONENTS = (
    "experts",
    "prefill_experts",
    "rotations",
    "routers",
    "routing",
    "shared_gates",
    "hc_down",
    "hc_up",
    "small_projections",
    "gdn_sequences",
    "gdn_prefill",
    "gdn_conv",
)


def validate_gdn_prefill_state(buffer, binding):
    """Native state arenas may flatten the same contiguous FP32 matrix."""
    size = math.prod(buffer["shape"])
    offset = binding.get("offset", 0)
    if buffer["dtype"] != "f32" or size != 48 * 128 * 128 or type(offset) is not int or offset != 0:
        raise ValueError("Expected contiguous FP32 GDN prefill state")


def upgrade(model, destination, report, components=COMPONENTS):
    import json
    import numpy as np
    from safetensors import safe_open
    from kernels.model.flash_next import dense_projection, scalar_bf16_projection
    from kernels.model.integer_vq import integer_vq_grouped
    from kernels.model.hyperconnection import hc_down_partial, hc_up_mix
    from kernels.model.int8_projection import int8_small_projection
    from kernels.model.gdn_sequence import gdn_sequence
    from kernels.operators.op08_gdn_conv_prep import gdn_conv_prep
    from kernels.model.rotation_a8 import rotate_activation_a8
    from kernels.model.moe import expert_offsets, expert_tiles, router_topk
    from tools.operators.common import configure, export_kernel
    from tools.quantization.vq import (
        e8p_sign_table,
        e8p_selector_table,
        e8p_selector_supports_parity_xor,
    )
    from safetensors.numpy import save_file

    configure()
    report.mkdir(parents=True, exist_ok=True)
    data, origin, package = load_model(model)
    if data["architecture"] != "flash_next" or data["compute_policy"] != "int8_quality":
        raise ValueError("Expected a Flash Next INT8 quality package")
    buffers = {b["name"]: b for b in data["metadata"]["buffers"]}
    text = json.loads((model / "config.json").read_text())["text_config"]
    experts, hidden = text["num_experts"], text["hidden_size"]
    if experts != 512 or hidden != 2560:
        raise ValueError("Unsupported Flash Next geometry")
    operator = clone_model(model, destination, origin)
    reader = AssetReader(origin)
    index = json.loads((model / "cache/weights/model.safetensors.index.json").read_text())
    exports, host_cache, tables, selector_tables = {}, {}, {}, {}

    def host(kernel):
        identity = kernel["host_abi"]["sha256"]
        if identity not in host_cache:
            (parsed,) = parse_host(reader.read(kernel["host_abi"]).decode())
            host_cache[identity] = parsed
        return host_cache[identity]

    def compile_kernel(key, factory):
        if key not in exports:
            folder = operator / "decode-aot" / key
            export_kernel(factory(), folder)
            (parsed,) = parse_host((folder / "host.txt").read_text())
            validate_parameter_count((folder / "kernel.cu").read_text(), parsed)
            assets = {
                field: dict(
                    file=str((folder / filename).relative_to(operator)),
                    sha256=file_hash(folder / filename),
                )
                for field, filename in (
                    ("module", "kernel.cubin"),
                    ("source", "kernel.cu"),
                    ("host_abi", "host.txt"),
                )
            }
            exports[key] = parsed, assets
        return exports[key]

    def short_table(original):
        name = original["name"]
        if name not in tables:
            b = buffers[name]
            tensor = b["data"]["tensor"]
            path = source_path((model / "cache/weights").resolve(), index["weight_map"][tensor])
            with safe_open(path, framework="np") as weights:
                raw = weights.get_tensor(tensor)
            if (
                b["shape"] != [1, 256, 2]
                or hashlib.sha256(raw.tobytes()).hexdigest() != b["data"]["sha256"]
            ):
                raise ValueError("Invalid E8P basis identity")
            expanded = e8p_sign_table(raw.view(np.int8).reshape(256, 8))
            digest = hashlib.sha256(expanded.tobytes()).hexdigest()
            candidates = [
                n
                for n, v in buffers.items()
                if (v.get("data") or {}).get("sha256") == digest
                and v["dtype"] == "u32"
                and v["shape"] == [1, 512, 4]
            ]
            if not candidates:
                raise ValueError("The exact E8P sign table is missing from model weights")
            tables[name] = dict(kind="buffer", name=candidates[0])
        return tables[name]

    parity_strategies = {}

    def selector_table(original):
        name = original["name"]
        if name in selector_tables:
            return selector_tables[name]
        b = buffers[name]
        if b["dtype"] != "u32" or b["shape"] not in ([1, 256, 2], [1, 512, 4], [1, 256, 1]):
            raise ValueError("Invalid E8P decoder table shape")
        tensor = b["data"]["tensor"]
        path = source_path((model / "cache/weights").resolve(), index["weight_map"][tensor])
        with safe_open(path, framework="np") as weights:
            raw = weights.get_tensor(tensor)
        if hashlib.sha256(raw.tobytes()).hexdigest() != b["data"]["sha256"]:
            raise ValueError("Invalid E8P decoder table identity")
        if b["shape"] == [1, 256, 1]:
            parity_strategies[name] = e8p_selector_supports_parity_xor(raw[0])
            selector_tables[name] = original
            return original
        if b["shape"] == [1, 512, 4]:
            basis = (
                np.ascontiguousarray(raw[0, :256, [0, 2]].T)
                .view(np.int8)
                .reshape(256, 8)
                .astype(np.int16)
                - 1
            )
            if not np.isin(basis, [-10, -6, -2, 0, 2, 6, 10]).all():
                raise ValueError("Source sign table is not a spread29 E8P basis")
            basis = basis.astype(np.int8)
            if not np.array_equal(e8p_sign_table(basis)[None], raw):
                raise ValueError("Source sign table has inconsistent parity/sign entries")
        else:
            basis = raw.view(np.int8).reshape(256, 8)
        encoded = e8p_selector_table(basis)[None]
        digest = hashlib.sha256(encoded.tobytes()).hexdigest()
        existing = next(
            (
                n
                for n, v in buffers.items()
                if (v.get("data") or {}).get("sha256") == digest
                and v["dtype"] == "u32"
                and v["shape"] == [1, 256, 1]
            ),
            None,
        )
        if existing is None:
            existing = f"W_e8p_selector_{digest}"
            filename = f"e8p-selector-{digest}.safetensors"
            save_file(
                {existing: encoded},
                destination / "cache/weights" / filename,
                metadata={f"orin.layout.{existing}": "native_contiguous"},
            )
            index["weight_map"][existing] = filename
            index.setdefault("metadata", {})["total_size"] = (
                index.get("metadata", {}).get("total_size", 0) + encoded.nbytes
            )
            contract = dict(
                name=existing,
                dtype="u32",
                shape=[1, 256, 1],
                layout="native_contiguous",
                alignment=256,
                access="read",
            )
            buffer = dict(contract, data=dict(tensor=existing, sha256=digest))
            data["metadata"]["buffers"].append(buffer)
            data["buffer_scopes"][existing] = "weights"
            buffers[existing] = buffer
            package["buffer_contracts"].append(contract)
            data["metadata"]["weight_bytes"] += encoded.nbytes
        parity_strategies[existing] = e8p_selector_supports_parity_xor(encoded[0])
        selector_tables[name] = dict(kind="buffer", name=existing)
        return selector_tables[name]

    if not components or set(components) - set(COMPONENTS):
        raise ValueError("Unknown or empty decode optimization components")
    updated = dict.fromkeys(components, 0)
    q4_heads_repaired = 0
    single_head = next((k for k in package["kernels"] if k["name"] == "head_m1/body/k5"), None)
    for i, kernel in enumerate(package["kernels"]):
        program = kernel["name"].split("/")[0]
        match = re.fullmatch(r"(?:verify|flash_batch|mtp_warm|mtp_head|head)_m([1-8])", program)
        prefill = re.fullmatch(r"prefill_m(512|2048|4096)", program)
        rows = (
            int(prefill[1])
            if prefill
            else int(match[1])
            if match
            else 1
            if program == "decode"
            else None
        )
        if rows is None:
            continue
        if prefill and not set(updated) & {"rotations", "prefill_experts", "gdn_prefill"}:
            continue
        if prefill and "gdn_prefill" in updated:
            roles = {
                a["value"].removesuffix(".data_ptr()") for a in host(kernel)["ordered_arguments"]
            }
            if roles == {"Q", "K", "V", "G", "Beta", "State", "Out"}:
                pointers = pointer_bindings(kernel, host(kernel))
                state = buffers[pointers["State"]["name"]]
                validate_gdn_prefill_state(state, pointers["State"])
                rebuilt, assets = compile_kernel(
                    f"gdn-prefill-m{rows}",
                    lambda rows=rows: gdn_sequence(
                        rows, threads=32, value_tile=16, in_place=True, prefetch_qk=True
                    ),
                )
                package["kernels"][i] = replace_binding(kernel, rebuilt, assets, pointers)
                updated["gdn_prefill"] += 1
                continue
        if "rotations" in updated and {
            a["value"].removesuffix(".data_ptr()") for a in host(kernel)["ordered_arguments"]
        } == {"X", "Signs", "Q", "S"}:
            pointers = pointer_bindings(kernel, host(kernel))
            live_rows, width, swiglu = rotation_geometry(program, kernel, pointers, buffers)
            if swiglu != ("expf(" in reader.read(kernel["source"]).decode()):
                raise ValueError("Rotation activation boundary does not match its weight role")
            rebuilt, assets = compile_kernel(
                f"rotation-m{live_rows}-k{width}-swiglu{int(swiglu)}",
                lambda live_rows=live_rows, width=width, swiglu=swiglu: rotate_activation_a8(
                    live_rows, width, swiglu=swiglu
                ),
            )
            package["kernels"][i] = replace_binding(kernel, rebuilt, assets, pointers)
            updated["rotations"] += 1
            continue
        if (
            "rotations" in updated
            and program.startswith("mtp_head_m")
            and any(a.get("name") == "Q4_output_packed" for a in kernel["args"])
        ):
            pointers = pointer_bindings(kernel, host(kernel))
            if single_head is None or set(pointers) != {"A", "W", "S", "AS", "Out"}:
                raise ValueError("Missing single-token Q4 head")
            template_pointers = pointer_bindings(single_head, host(single_head))
            for role in ("W", "S"):
                if pointers[role] != template_pointers[role]:
                    raise ValueError("MTP head differs from the shared Q4 output weights")
            out = buffers[pointers["Out"]["name"]]
            if out["dtype"] != "f32":
                raise ValueError("Expected FP32 draft logits")
            assets = {field: single_head[field] for field in ("module", "source", "host_abi")}
            package["kernels"][i] = replace_binding(kernel, host(single_head), assets, pointers)
            q4_heads_repaired += 1
            continue
        weights = [
            buffers[a["name"]]
            for a in kernel["args"]
            if a.get("name") in buffers and buffers[a["name"]].get("data")
        ]
        packed = next((w for w in weights if w["name"].endswith("_exps.weight_packed")), None)
        router = next((w for w in weights if w["name"].endswith("ffn_gate_inp.weight_value")), None)
        shared_gate = next(
            (w for w in weights if w["name"].endswith("ffn_gate_inp_shexp.weight_value")), None
        )
        hc_down = next(
            (w for w in weights if "hc_" in w["name"] and w["name"].endswith("_down.weight_value")),
            None,
        )
        hc_up = next(
            (w for w in weights if "hc_" in w["name"] and w["name"].endswith("_up.weight_value")),
            None,
        )
        small = next(
            (w for w in weights if w["dtype"] == "i8" and w["shape"] == [48, hidden]), None
        )
        if prefill and "prefill_experts" not in updated:
            continue
        if "routing" in updated and not weights:
            roles = {
                a["value"].removesuffix(".data_ptr()") for a in host(kernel)["ordered_arguments"]
            }
            if roles == {"Logits", "IDs", "Prob"}:
                pointers = pointer_bindings(kernel, host(kernel))
                if kernel["grid"] != [rows, 1, 1]:
                    raise ValueError("Unexpected router row geometry")
                rebuilt, assets = compile_kernel(
                    f"routing-m{rows}", lambda rows=rows: router_topk(rows, experts, 10)
                )
                package["kernels"][i] = replace_binding(kernel, rebuilt, assets, pointers)
                updated["routing"] += 1
                continue
        if prefill:
            roles = {
                a["value"].removesuffix(".data_ptr()") for a in host(kernel)["ordered_arguments"]
            }
            plan = expert_tile_config(rows)
            if roles == {"Counts", "RowOffsets", "TileOffsets", "TileCount"}:
                pointers = pointer_bindings(kernel, host(kernel))
                rebuilt, assets = compile_kernel(
                    f"expert-offsets-bm{plan['block_m']}",
                    lambda plan=plan: expert_offsets(experts, plan["block_m"]),
                )
                package["kernels"][i] = replace_binding(kernel, rebuilt, assets, pointers)
                continue
            if roles == {"Counts", "TileOffsets", "TileExpert", "TileRow"}:
                pointers = pointer_bindings(kernel, host(kernel))
                needed = (rows * 10 + plan["block_m"] - 1) // plan["block_m"] + min(
                    experts, rows * 10
                )
                if any(
                    math.prod(buffers[pointers[a]["name"]]["shape"]) < needed
                    for a in ("TileExpert", "TileRow")
                ):
                    raise ValueError("Expert routing tile arena is too small")
                rebuilt, assets = compile_kernel(
                    f"expert-tiles-m{rows}-bm{plan['block_m']}",
                    lambda rows=rows, plan=plan: expert_tiles(rows, experts, 10, plan["block_m"]),
                )
                package["kernels"][i] = replace_binding(kernel, rebuilt, assets, pointers)
                continue
        conv = next((w for w in weights if w["name"].endswith("ssm_conv1d.weight_value")), None)
        if not prefill and rows == 1 and conv is not None and "gdn_conv" in updated:
            pointers = pointer_bindings(kernel, host(kernel))
            if (
                conv["dtype"] != "f32"
                or conv["shape"] != [10240, 4]
                or set(pointers)
                != {
                    "X",
                    "W",
                    "HI",
                    "lengths",
                    "positions",
                    "Q",
                    "K",
                    "V",
                    "HO",
                    "positions_out",
                }
            ):
                raise ValueError("Expected native single-token GDN convolution ABI")
            if any(buffers[pointers[role]["name"]]["dtype"] != "f16" for role in ("Q", "K", "V")):
                raise ValueError("Expected native GDN output rounding")
            rebuilt, assets = compile_kernel(
                "gdn-conv-compact",
                lambda: gdn_conv_prep(
                    B=1,
                    tokens=1,
                    compact_decode=True,
                    normalize_round_fp16=False,
                    conv_product_round_fp16=False,
                    weight_dtype="float32",
                ),
            )
            package["kernels"][i] = replace_binding(kernel, rebuilt, assets, pointers)
            updated["gdn_conv"] += 1
            continue
        expert_component = "prefill_experts" if prefill else "experts"
        if packed is not None and expert_component in updated:
            e, groups, n, words = packed["shape"]
            if e != experts or words != 16 or n not in (1280, hidden):
                raise ValueError("Unexpected expert packing")
            k = groups * 128
            pointers = pointer_bindings(kernel, host(kernel))
            tile_capacity = buffers[pointers["TileExpert"]["name"]]["shape"][0]
            plan = expert_projection_config(rows, k)
            tiles = (
                (rows * 10 + plan["block_m"] - 1) // plan["block_m"] + min(experts, rows * 10)
                if prefill
                else kernel["grid"][0]
            )
            if tiles > tile_capacity or tiles < min(experts, rows * 10):
                raise ValueError("Unexpected static routed tile capacity")
            if plan["selectorbook"]:
                pointers["Book"] = selector_table(pointers["Book"])
                if plan.get("warp_decode") and parity_strategies[pointers["Book"]["name"]]:
                    plan["parity_xor"] = True
            elif plan["shortbook"] and not pointers["Book"]["name"].endswith("short_table"):
                pointers["Book"] = short_table(pointers["Book"])
            rebuilt, assets = compile_kernel(
                f"expert-m{rows}-n{n}-k{k}-xor{int(plan.get('parity_xor', False))}",
                lambda rows=rows, tiles=tiles, n=n, k=k, plan=plan: integer_vq_grouped(
                    rows * 10, experts, tiles, n, k, kind="e8p", shared_table=True, **plan
                ),
            )
            package["kernels"][i] = replace_binding(kernel, rebuilt, assets, pointers)
            updated[expert_component] += 1
        elif prefill:
            continue
        elif router is not None and "routers" in updated:
            if router["dtype"] != "bf16" or router["shape"] != [experts, hidden]:
                raise ValueError("Unexpected router weight")
            pointers = pointer_bindings(kernel, host(kernel))
            bn = router_tile(rows, experts, hidden, experts, hidden)
            rebuilt, assets = compile_kernel(
                f"router-m{rows}",
                lambda rows=rows, bn=bn: dense_projection(
                    rows, experts, hidden, "bfloat16", "float32", block_n=bn
                ),
            )
            package["kernels"][i] = replace_binding(kernel, rebuilt, assets, pointers)
            updated["routers"] += 1
        elif shared_gate is not None and "shared_gates" in updated:
            if shared_gate["dtype"] != "bf16" or shared_gate["shape"] != [1, hidden]:
                raise ValueError("Unexpected shared gate weight")
            pointers = pointer_bindings(kernel, host(kernel))
            # Preserve Tensor Core K traversal and FP32 accumulation/FP16
            # rounding while loading only the live activation and weight rows.
            rebuilt, assets = compile_kernel(
                f"shared-gate-m{rows}",
                lambda rows=rows: scalar_bf16_projection(rows, hidden),
            )
            package["kernels"][i] = replace_binding(kernel, rebuilt, assets, pointers)
            updated["shared_gates"] += 1
        elif hc_down is not None and "hc_down" in updated:
            if hc_down["dtype"] != "bf16" or hc_down["shape"] != [320, 10240]:
                raise ValueError("Unexpected HC down weight")
            pointers = pointer_bindings(kernel, host(kernel))
            if set(pointers) != {"A", "W", "Partials"}:
                raise ValueError("Expected split-K HC down ABI")
            if kernel["grid"] not in ([1, 10, 4], [1, 5, 4]):
                raise ValueError("Expected a small HC split-K launch")
            # Shared plan arenas can be larger than this program's live rows.
            for argument, needed, item_bytes in (
                ("A", rows * 10240, 2),
                ("Partials", 4 * rows * 320, 4),
            ):
                buffer = buffers[pointers[argument]["name"]]
                offset = pointers[argument].get("offset", 0)
                if offset + needed * item_bytes > int(np.prod(buffer["shape"])) * item_bytes:
                    raise ValueError("HC down binding exceeds its workspace")
            rebuilt, assets = compile_kernel(
                f"hc-down-m{rows}",
                lambda rows=rows: hc_down_partial(rows, 320, 10240, 4),
            )
            package["kernels"][i] = replace_binding(kernel, rebuilt, assets, pointers)
            updated["hc_down"] += 1
        elif hc_up is not None and "hc_up" in updated:
            if hc_up["dtype"] != "bf16" or hc_up["shape"] != [10240, 320]:
                raise ValueError("Unexpected HC up weight")
            pointers = pointer_bindings(kernel, host(kernel))
            if set(pointers) != {"A", "W", "Normed", "Out"}:
                raise ValueError("Expected fused HC up/mix ABI")
            rebuilt, assets = compile_kernel(
                f"hc-up-m{rows}", lambda rows=rows: hc_up_mix(rows, hidden, 320)
            )
            package["kernels"][i] = replace_binding(kernel, rebuilt, assets, pointers)
            updated["hc_up"] += 1
        elif small is not None and "small_projections" in updated:
            pointers = pointer_bindings(kernel, host(kernel))
            if set(pointers) != {"A", "Weight", "WeightScale", "TokenScale", "Output"}:
                raise ValueError("Expected row-scaled INT8 projection ABI")
            rebuilt, assets = compile_kernel(
                f"small-m{rows}", lambda rows=rows: int8_small_projection(rows, 48, hidden)
            )
            package["kernels"][i] = replace_binding(kernel, rebuilt, assets, pointers)
            updated["small_projections"] += 1
        elif (
            "gdn_sequences" in updated
            and not weights
            and any(a.get("name", "").endswith("_gdn") for a in kernel["args"])
        ):
            pointers = pointer_bindings(kernel, host(kernel))
            compact = program.startswith("verify_")
            required = {"Q", "K", "V", "G", "Beta", "State", "Out"}
            if compact:
                required.add("Prefix")
            if set(pointers) != required:
                continue
            # Legacy static batch plans share projections but invoke each
            # request's recurrent state for one token in their private phase.
            gdn_rows = 1 if "/private/" in kernel["name"] else rows
            rebuilt, assets = compile_kernel(
                f"gdn-sequence-m{gdn_rows}-{'compact' if compact else 'inplace'}",
                lambda rows=gdn_rows, compact=compact: gdn_sequence(
                    rows, compact=compact, in_place=not compact
                ),
            )
            package["kernels"][i] = replace_binding(kernel, rebuilt, assets, pointers)
            updated["gdn_sequences"] += 1
    if not all(updated.values()):
        raise ValueError("Missing requested decode optimization bindings")
    write_json(destination / "cache/weights/model.safetensors.index.json", index)
    digest = compact_package(operator, package, destination / "cache/packages")
    shutil.rmtree(operator)
    data["execution_package"] = digest
    write_json(destination / "cache/model.json", data)
    write_json(
        report / "upgrade.json",
        dict(
            execution_package=digest,
            replaced=updated,
            compiled_variants=len(exports),
            weight_bytes=data["metadata"]["weight_bytes"],
            weight_payloads_unchanged=True,
            q4_last_token_heads_rebuilt=q4_heads_repaired,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--components", nargs="+", choices=COMPONENTS, default=COMPONENTS)
    args = parser.parse_args()
    with atomic_model(args.model_output, args.engine) as staging:
        upgrade(args.model.resolve(strict=True), staging, args.output, args.components)
    print("DECODE PACKAGE READY", args.model_output, flush=True)


if __name__ == "__main__":
    main()
