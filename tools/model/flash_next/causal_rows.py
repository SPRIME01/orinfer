"""Publish length-parametrized causal state kernels on shared token-row frames."""

import argparse
import math
import re
from pathlib import Path

from tools.model.publication import (
    atomic_model,
    clone_model,
    commit_package,
    load_model,
    write_json,
)
from tools.model.flash_next.compiler import Compiler


def history_geometry(buffer, channels=10240):
    """A flattened state still stores chronological rows of raw channels."""
    elements = math.prod(buffer["shape"])
    if elements % channels or not 0 < elements // channels <= 9:
        raise ValueError("Invalid causal history geometry")
    return elements // channels, channels


def ring_geometry(buffer):
    if len(buffer["shape"]) != 2 or buffer["shape"][1] != 10240:
        raise ValueError("Invalid Flash hidden ring")
    return tuple(buffer["shape"])


def factory(roles, rows, context, buffers, args, shift=24):
    from kernels.model import qsa, qsa_attention, flash_next, flash_mtp, flash_control
    from kernels.model import speculation, speculative_factors, causal_rows, control, ple
    from kernels.model.gdn_sequence import _gdn_sequence, gdn_commit
    from kernels.operators.op08_gdn_conv_prep import gdn_conv_prep
    from kernels.vision.features import overlay

    opts = dict(dynamic_rows=True)
    if {"HI", "HO", "lengths", "positions_out"} <= roles:
        return "gdn-conv", lambda: gdn_conv_prep.__wrapped__(
            B=1,
            tokens=rows,
            normalize_round_fp16=False,
            conv_product_round_fp16=False,
            weight_dtype="float32",
            dynamic_tokens=True,
        )
    if {"Q", "K", "V", "G", "Beta", "State", "Out"} <= roles:
        compact = "Prefix" in roles
        return f"gdn-{int(compact)}", lambda: _gdn_sequence(
            rows,
            compact=compact,
            in_place=not compact,
            threads=128 if compact else 32,
            value_tile=32 if compact else 16,
            prefetch_qk=not compact,
            token_unroll=1 if compact else 4,
            **opts,
        )
    if roles == {"G", "K", "SavedG", "SavedK"}:
        return "save-gdn", lambda: speculative_factors.save_gdn_factors.__wrapped__(rows, **opts)
    if roles == {"X", "State", "Prefix"}:
        b = buffers[args["State"]["name"]]
        history, width = history_geometry(b)
        return f"history-{history}-{width}", lambda: flash_mtp.history_prefix.__wrapped__(
            rows, width, history, **opts
        )
    if roles == {"Position", "Prefix", "QK", "State"}:
        return "pending-prefix", lambda: flash_mtp.pending_prefix.__wrapped__(rows, **opts)
    if roles == {"Count", "Prefix", "State"}:
        b = buffers[args["State"]["name"]]
        width = math.prod(b["shape"])
        dtype = {"f16": "float16", "f32": "float32"}[b["dtype"]]
        return f"select-{width}-{dtype}", lambda: speculation.select_prefix.__wrapped__(
            rows, width, dtype, **opts
        )
    if roles == {"Accepted", "G", "K", "State", "Update"}:
        return "gdn-commit", lambda: gdn_commit.__wrapped__(rows, **opts)
    if roles == {"Gated", "Normed", "Output", "State", "Weight"}:
        return "ple-conv", lambda: ple.ple_conv.__wrapped__(rows, 10240, **opts)
    if roles in ({"NextState", "Normed", "State"}, {"NextState", "Normed"}):
        args.setdefault("State", args["NextState"])
        return "ple-history", lambda: ple.ple_history.__wrapped__(rows, 10240, **opts)
    if {"QGate", "KCache", "VCache"} <= roles:
        return "qsa-prepare", lambda: flash_next.qsa_prepare.__wrapped__(
            rows, context, staged=True, is_neox_style=True, mrope=True, **opts
        )
    if roles == {"K", "KS", "Key", "Position", "V", "VS", "Value"}:
        return "qsa-store", lambda: qsa.kv_store.__wrapped__(rows, context, **opts)
    if roles == {"Coordinates", "Position", "QK", "Query", "Weight"}:
        return "index-query", lambda: qsa.index_query.__wrapped__(rows, context, **opts)
    if roles == {"Cache", "Coordinates", "Pending", "Position", "QK", "Weight"}:
        return "index-compress", lambda: qsa.index_compress.__wrapped__(
            rows, context, mrope=True, **opts
        )
    if roles == {"Pending", "Position", "QK"}:
        return "index-pending", lambda: qsa.index_pending.__wrapped__(rows, **opts)
    if roles == {"Cache", "Position", "Query", "Scores"}:
        return "index-scores", lambda: qsa.index_scores.__wrapped__(rows, context, **opts)
    if "Hist" in roles:
        if "Scores" in roles:
            return f"radix-hist-{shift}", lambda: qsa.radix_histogram.__wrapped__(
                rows, context, shift, **opts
            )
        return f"radix-choose-{shift}", lambda: qsa.radix_choose.__wrapped__(
            rows, context, shift, **opts
        )
    if roles == {"Counts", "Position", "Prefix", "Scores"}:
        return "selection-counts", lambda: qsa.selection_counts.__wrapped__(rows, context, **opts)
    if roles == {"Counts", "Greater", "Offsets", "Position", "Selected"}:
        return "selection-offsets", lambda: qsa.selection_offsets.__wrapped__(rows, context, **opts)
    if roles == {"Greater", "Offsets", "Position", "Prefix", "Scores", "Selected"}:
        return "selection-scatter", lambda: qsa.selection_scatter.__wrapped__(rows, context, **opts)
    if {"Query", "Max", "Den", "Selected"} <= roles:
        return "qsa-attention", lambda: qsa_attention.sparse_attention.__wrapped__(
            rows, context, packed=True, **opts
        )
    if roles == {"Den", "Gate", "Max", "Out", "Output"}:
        return "qsa-merge", lambda: qsa.sparse_merge.__wrapped__(rows, **opts)
    if roles == {"Hidden", "Position", "Ring"}:
        ring, width = ring_geometry(buffers[args["Ring"]["name"]])
        return f"capture-{ring}-{width}", lambda: flash_control.capture.__wrapped__(
            rows, width, ring, **opts
        )
    if roles == {"Condition", "Hidden"}:
        return "last-hidden", lambda: flash_control.last_hidden.__wrapped__(rows, 10240, **opts)
    if roles == {"Out", "Step", "Target"}:
        ring, _ = ring_geometry(buffers[args["Target"]["name"]])
        return f"gather-hidden-{ring}", lambda: speculation.gather_target_hidden.__wrapped__(
            rows, 10240, ring, ring=True, **opts
        )
    if roles == {"Length"}:
        return "length", lambda: causal_rows.length.__wrapped__(rows)
    if roles == {"Step"}:
        return "advance", lambda: control.advance.__wrapped__(rows, **opts)
    if roles == {"Embedding", "Features", "Index", "Position"}:
        count = math.prod(buffers[args["Features"]["name"]]["shape"]) // 2560
        return f"overlay-{count}", lambda: overlay.__wrapped__(rows, 2560, context, count, **opts)
    if roles == {"Pair", "Tokens", "Status"}:
        return "selections", lambda: flash_control.selections.__wrapped__(rows, **opts)
    raise ValueError(f"Unsupported causal state ABI: {sorted(roles)}")


def publish(data, package, compiler):
    """Keep one binding per semantic slot; wrappers contain only row/copy extents."""
    original = {k["name"]: k for k in package["kernels"]}
    added, launches, removed = [], [], set()
    context = data["metadata"]["max_context"]

    def emit(old, name, rows, aliases=None, shift=24):
        args = compiler.pointers(old)
        args = {n: v for n, v in args.items() if n not in ("m", "rows")}
        for v in args.values():
            if aliases and v.get("name") in aliases:
                v["name"] = aliases[v["name"]]
        key, build = factory(set(args), rows, context, compiler.buffers, args, shift)
        k, launch = compiler.emit(old, name, args, rows, build=build, key=f"causal-{key}-{rows}")
        added.append(k)
        launches.append(launch)

    families = [
        (r"flash_segment_m(\d+)/(.*)", "flash_segment_m512/", "causal_rows/target/", 512),
        (
            r"(?:verify_rows_m|flash_verify_segment_m)(\d+)/(.*)",
            "verify_rows_m8/",
            "causal_rows/verify/",
            8,
        ),
        (r"restore_m(\d+)/(.*)", "restore_m8/", "causal_rows/restore/", 8),
        (r"draft_rows/causal_m(\d+)/(.*)", "draft_rows/causal_m512/", "causal_rows/draft/", 512),
    ]
    for pattern, source, target, rows in families:
        shifts = {}
        removed.update(n for n in original if re.fullmatch(pattern, n))
        selected = sorted(
            (k for k in original.values() if k["name"].startswith(source)),
            key=lambda k: (k["name"].rsplit("/k", 1)[0], int(k["name"].rsplit("/k", 1)[1])),
        )
        for old in selected:
            section = old["name"].rsplit("/", 1)[0]
            roles = compiler.pointers(old)
            shift = 24
            if "Hist" in roles:
                index = shifts.get(section, 0)
                shift = (24, 16, 8, 0)[index // 2]
                shifts[section] = index + 1
            emit(old, old["name"].replace(source, target, 1), rows, shift=shift)
    # Shared draft chronology is also the private path for independent packets.
    removed.update(
        n for n in original if re.fullmatch(r"draft_rows/(?:private|warm_m\d+/private)/k\d+", n)
    )
    controls = [
        ("mtp_warm_m512/begin/k0", "causal_rows/warm/gather", 512),
        ("mtp_warm_m512/begin/k1", "causal_rows/warm/length", 512),
        ("mtp_warm_m512/end/k0", "causal_rows/warm/advance", 512),
        ("draft_rows/capture_m512", "causal_rows/warm/capture", 512),
        ("verify_m8/end/k0", "causal_rows/verify/advance", 8),
        ("verify_m8/head/k8", "causal_rows/verify/selections", 8),
        ("verify_rows_capture_m8/body/k0", "causal_rows/verify/capture", 8),
        ("flash_mixed_capture_m512/body/k0", "causal_rows/target/capture", 512),
    ]
    if data["metadata"].get("vision"):
        controls += [("mtp_warm_m512/begin/k2", "causal_rows/warm/overlay", 512)]
    for source, target, rows in controls:
        emit(original[source], target, rows)
    for name in original:
        if re.match(
            r"(?:mtp_warm_m\d+/(?:begin|end)/|draft_rows/capture_m|verify_rows_capture_m|flash_(?:mixed|verify)_capture_m)",
            name,
        ):
            removed.add(name)
        if re.match(r"verify_m\d+/(?:end/|head/)", name):
            removed.add(name)
    package["kernels"] = [k for k in package["kernels"] if k["name"] not in removed] + added
    package["dynamic_batch_kernels"] = [
        k for k in package["dynamic_batch_kernels"] if k["name"] not in removed
    ] + launches
    package["plan_kernel_contracts"] = sorted(
        (set(package["plan_kernel_contracts"]) - removed) | {k["name"] for k in added}
    )
    spec = data["metadata"]["mtp"]
    spec["batch_programs"] = [
        n
        for n in spec["batch_programs"]
        if not re.match(
            r"draft_rows/(?:private|warm_m)",
            n,
        )
    ]
    spec["batch_programs"].extend(["causal_rows/draft"])
    for layer in range(48):
        spec["batch_programs"].append(f"causal_rows/verify/layer{layer}/private")
    spec["batch_programs"] += ["causal_rows/verify/layer1/ple", "causal_rows/verify/capture"]
    return len(removed)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "model-output", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--compile-cache", type=Path)
    a = p.parse_args()
    if a.compile_cache:
        from tools.model.publication import seed_compile_cache

        seed_compile_cache(a.compile_cache, a.output / "cache/0.1.15")
    from tools.operators.common import configure

    configure()
    data, origin, package = load_model(a.model)
    with atomic_model(a.model_output) as staging:
        root = clone_model(a.model, staging, origin)
        compiler = Compiler(
            origin, root, a.output, {b["name"]: b for b in data["metadata"]["buffers"]}
        )
        count = publish(data, package, compiler)
        digest = commit_package(staging, root, data, package)
    write_json(
        a.output / "results.json",
        dict(removed=count, compiled=len(compiler.compiled), execution_package=digest),
    )


if __name__ == "__main__":
    main()
