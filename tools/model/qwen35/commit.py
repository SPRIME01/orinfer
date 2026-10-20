"""Batch exact-prefix restoration with the shared independent-owner wrapper."""

import hashlib
import inspect
from kernels.model.independent_rows import independent_rows
from kernels.model.gdn_sequence import gdn_commit_controlled
from kernels.model.speculation import select_prefix
from tools.model.compile import primitive
from tools.model.qwen35.parallel import LANES

TABLE = "QwenCommitPointers"
COLUMNS = (
    "VerifyK",
    "VerifyG",
    "VerifyUpdate",
    "AcceptedInputs",
    "SegmentLength",
    "State",
    "HistoryPrefixes",
    "History",
)


def factory(role, hk, hv, channels, layers, columns, capacity):
    columns = {name: i for i, name in enumerate(columns)}
    if role == "state":
        if capacity:
            from kernels.model.gdn_deferred import commit

            fn = commit(capacity, key_heads=hk, value_heads=hv)
        else:
            fn = inspect.unwrap(gdn_commit_controlled)(8, key_heads=hk, value_heads=hv)
        mapping = dict(
            K="VerifyK",
            G="VerifyG",
            Update="VerifyUpdate",
            Accepted="AcceptedInputs",
            Length="SegmentLength",
            State="State",
        )
        if capacity:
            mapping.update(
                LogK="GdnLogKey", LogD="GdnLogDelta", LogG="GdnLogGate", Count="GdnLogCount"
            )
    elif role == "history":
        fn = inspect.unwrap(select_prefix)(8, 3 * channels, "float16")
        mapping = dict(Prefix="HistoryPrefixes", Count="AcceptedInputs", State="History")
    else:
        raise ValueError("Unknown commit role")
    roles = {name: ("private", identity) for name, identity in mapping.items()}
    wrapped, _ = independent_rows(
        fn, LANES, roles, columns, row_stride=layers * len(columns), dynamic=True
    )
    return wrapped


def build(b):
    layers = len(b.text["layer_types"])
    table = b.mtp.get("verification_table")
    columns = table["columns"] if table else COLUMNS
    target = table["buffer"] if table else TABLE
    capacity = 0
    if table and table.get("deferred"):
        capacity = next(
            v["shape"][0] for n, v in b.weights.buffers.items() if n.endswith("_GdnLogGate")
        )
    if not table:
        b.weights.fixed(target, "u64", [LANES * layers * len(columns)])
    for layer, kind in enumerate(b.text["layer_types"]):
        if kind != "linear_attention":
            continue
        for role in ("state", "history"):
            fn = factory(role, b.hk, b.hv, b.qkv, layers, columns, capacity)
            b.emit(
                f"commit/layer{layer}/{role}",
                "commit-rows-" + hashlib.sha256(str(fn).encode()).hexdigest(),
                lambda fn=fn: primitive(lambda: fn),
                dict(
                    Pointers=dict(kind="buffer_slice", name=target, offset=layer * len(columns) * 8)
                ),
                LANES,
            )
