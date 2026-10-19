"""Qwen MoE FFN: shared routing, grouped integer projection and row transforms."""

import math
import numpy as np
from tools.model.compile import primitive


def workspace(b):
    w, c, e, top, f, h = (
        b.weights,
        4096,
        b.text["num_experts"],
        b.text["num_experts_per_tok"],
        b.f,
        b.h,
    )
    if "MoeLogits" in w.buffers:
        return
    tiles = math.ceil(c * top / 16) + min(e, c * top)
    for name, dtype, shape in [
        ("Logits", "f32", [c, e]),
        ("Gate", "f16", [c]),
        ("IDs", "i32", [c, top]),
        ("Prob", "f32", [c, top]),
        ("Partial", "i32", [e, math.ceil(c * top / 1024)]),
        ("SegmentOffsets", "i32", [e, math.ceil(c * top / 1024)]),
        ("Relative", "i32", [c, top]),
        ("Counts", "i32", [e]),
        ("Offsets", "i32", [e]),
        ("TileOffsets", "i32", [e]),
        ("TileExpert", "i32", [tiles]),
        ("TileRow", "i32", [tiles]),
        ("TileCount", "i32", [1]),
        ("Slots", "i32", [c, top]),
        ("A8", "i8", [c * top, h]),
        ("AS", "f16", [c * top]),
        ("GateUp", "f16", [c * top, 2 * f]),
        ("DownA8", "i8", [c * top, f]),
        ("DownAS", "f16", [c * top]),
        ("Output", "f16", [c * top, h]),
        ("Shared", "f16", [c, h]),
    ]:
        w.fixed("Moe" + name, dtype, shape)


def banks(b, prefix, draft):
    w, e = b.weights, b.text["num_experts"]
    if not draft:
        return [
            (
                w.immutable(prefix + ".experts." + name + ".packed"),
                w.immutable(prefix + ".experts." + name + ".scale"),
            )
            for name in ("gate_up_proj", "down_proj")
        ]
    result = []
    for role in ("gate_up", "down"):
        operands = []
        scales = []
        for i in range(e):
            names = [
                prefix + f".experts.{i}." + x + "_proj.weight"
                for x in (("gate", "up") if role == "gate_up" else ("down",))
            ]
            operands.append(np.concatenate([w.array(n + ".int8") for n in names]))
            scales.append(np.concatenate([w.array(n + ".scale") for n in names]))
        name = "Runtime." + prefix + "." + role
        result.append(
            (
                w.constant(name + ".int8", np.stack(operands)),
                w.constant(name + ".scale", np.stack(scales)),
            )
        )
    return result


def build(b, section, prefix, *, rows=4096, draft=False):
    from kernels.model import moe, rotation_a8
    from kernels.operators.op30_activation_quantization import _compile as a8_compile

    workspace(b)
    w, h, f = b.weights, b.h, b.f
    e, top = b.text["num_experts"], b.text["num_experts_per_tok"]
    norm = "MtpNorm" if draft else "Norm"
    output = "MtpHidden" if draft else "Hidden"
    gate_up = "MtpGateUp" if draft else "GateUp"
    codec = "w8" if draft else "e8p_q4"

    def emit(role, fn, args, *dimensions, multiplier=1, **options):
        b.emit(
            section + "/" + role,
            "moe-" + fn.__name__ + str((rows, dimensions, options)),
            lambda: primitive(
                fn, *dimensions, **({} if fn is a8_compile else dict(dynamic_rows=True)), **options
            ),
            args,
            rows * multiplier,
            row_multiplier=multiplier,
        )

    router = w.combined(
        prefix + ".router", [prefix + ".gate.weight", prefix + ".shared_expert_gate.weight"]
    )
    emit(
        "router",
        moe.router_projection,
        dict(X=norm, W=router, Logits="MoeLogits", SharedGate="MoeGate"),
        rows,
        h,
        e,
    )
    emit(
        "topk",
        moe.router_topk,
        dict(Logits="MoeLogits", IDs="MoeIDs", Prob="MoeProb"),
        rows,
        e,
        top,
    )
    emit(
        "histogram_local",
        moe.expert_histogram_local,
        dict(IDs="MoeIDs", Partial="MoePartial", Slots="MoeRelative"),
        rows,
        e,
        top,
    )
    emit(
        "histogram_offsets",
        moe.expert_histogram_offsets,
        dict(Partial="MoePartial", Offsets="MoeSegmentOffsets", Counts="MoeCounts"),
        rows,
        e,
        top,
    )
    emit(
        "histogram_finish",
        moe.expert_histogram_finish,
        dict(IDs="MoeIDs", Offsets="MoeSegmentOffsets", Slots="MoeRelative"),
        rows,
        e,
        top,
    )
    b.emit(
        section + "/offsets",
        f"moe-offsets-{e}-16",
        lambda: primitive(moe.expert_offsets, e, 16),
        dict(
            Counts="MoeCounts",
            RowOffsets="MoeOffsets",
            TileOffsets="MoeTileOffsets",
            TileCount="MoeTileCount",
        ),
        rows,
        dynamic=False,
    )
    emit(
        "tiles",
        moe.expert_tiles,
        dict(
            Counts="MoeCounts",
            TileOffsets="MoeTileOffsets",
            TileExpert="MoeTileExpert",
            TileRow="MoeTileRow",
        ),
        rows,
        e,
        top,
        16,
    )
    emit(
        "dispatch",
        moe.expert_dispatch,
        dict(
            A="A8",
            Scale="AS",
            IDs="MoeIDs",
            RelativeSlot="MoeRelative",
            RowOffsets="MoeOffsets",
            D="MoeA8",
            DS="MoeAS",
            SlotMap="MoeSlots",
        ),
        rows,
        h,
        e,
        top,
        h,
    )
    for role, offset in (("gate", 0), ("up", f if draft else 4096 * f)):
        b.projection(
            section + "/" + role,
            prefix + ".shared_expert." + role + "_proj.weight",
            f,
            h,
            gate_up,
            stride=2 * f if draft else f,
            offset=offset,
            rows=rows,
            kind=codec,
        )
    if draft:
        emit(
            "down_quant",
            a8_compile,
            dict(X=gate_up, Mask="UnusedA8Mask", Q="A8", S="AS"),
            f,
            f,
            False,
            True,
            256,
        )
    else:
        b.rotation(section + "/swiglu_rotate", f, gate_up, swiglu=True, plane_rows=4096)
    b.projection(
        section + "/down",
        prefix + ".shared_expert.down_proj.weight",
        h,
        f,
        "MoeShared",
        rows=rows,
        kind=codec,
    )
    bank_up, bank_down = banks(b, prefix, draft)
    common = dict(
        Book=b.book,
        Counts="MoeCounts",
        Offsets="MoeOffsets",
        TileExpert="MoeTileExpert",
        TileRow="MoeTileRow",
        TileCount="MoeTileCount",
    )
    emit(
        "expert_up",
        moe.integer_experts,
        dict(common, A="MoeA8", W=bank_up[0], WS=bank_up[1], AS="MoeAS", Out="MoeGateUp"),
        rows,
        2 * f,
        h,
        e,
        top,
        codec=codec,
    )
    if draft:
        emit(
            "expert_rotate",
            a8_compile,
            dict(X="MoeGateUp", Mask="UnusedA8Mask", Q="MoeDownA8", S="MoeDownAS"),
            f,
            f,
            False,
            True,
            256,
            multiplier=top,
        )
    else:
        emit(
            "expert_rotate",
            rotation_a8.rotate_activation_a8,
            dict(X="MoeGateUp", Signs=b.signs[f], Q="MoeDownA8", S="MoeDownAS"),
            rows * top,
            f,
            True,
            multiplier=top,
        )
    emit(
        "expert_down",
        moe.integer_experts,
        dict(
            common, A="MoeDownA8", W=bank_down[0], WS=bank_down[1], AS="MoeDownAS", Out="MoeOutput"
        ),
        rows,
        h,
        f,
        e,
        top,
        codec=codec,
    )
    emit(
        "combine",
        moe.moe_combine,
        dict(
            Expert="MoeOutput",
            SlotMap="MoeSlots",
            Prob="MoeProb",
            Shared="MoeShared",
            SharedGate="MoeGate",
            Output=output,
        ),
        rows,
        h,
        rows * top,
        top,
    )
