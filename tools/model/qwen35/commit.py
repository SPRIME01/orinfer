"""Batch exact-prefix restoration with the shared independent-owner wrapper."""

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


def factory(role, hk, hv, channels, layers, layer):
    columns = {name: i for i, name in enumerate(COLUMNS)}
    if role == "state":
        fn = inspect.unwrap(gdn_commit_controlled)(8, key_heads=hk, value_heads=hv)
        mapping = dict(
            K="VerifyK",
            G="VerifyG",
            Update="VerifyUpdate",
            Accepted="AcceptedInputs",
            Length="SegmentLength",
            State="State",
        )
    elif role == "history":
        fn = inspect.unwrap(select_prefix)(8, 3 * channels, "float16")
        mapping = dict(Prefix="HistoryPrefixes", Count="AcceptedInputs", State="History")
    else:
        raise ValueError("Unknown commit role")
    roles = {name: ("private", identity) for name, identity in mapping.items()}
    wrapped, _ = independent_rows(
        fn, LANES, roles, columns, row_stride=layers * len(COLUMNS), dynamic=True
    )
    return wrapped


def build(b):
    layers = len(b.text["layer_types"])
    b.weights.fixed(TABLE, "u64", [LANES * layers * len(COLUMNS)])
    for layer, kind in enumerate(b.text["layer_types"]):
        if kind != "linear_attention":
            continue
        for role in ("state", "history"):
            b.emit(
                f"commit/layer{layer}/{role}",
                f"commit-rows-{role}-{b.hk}-{b.hv}-{b.qkv}-{layers}",
                lambda role=role: primitive(factory, role, b.hk, b.hv, b.qkv, layers, layer),
                dict(
                    Pointers=dict(kind="buffer_slice", name=TABLE, offset=layer * len(COLUMNS) * 8)
                ),
                LANES,
            )
