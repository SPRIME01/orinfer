"""Validated static tiles shared by initial AOT builds and package upgrades."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectionPolicy:
    """Storage determines precision; row geometry only chooses the implementation."""

    packed: str
    scale: str
    signs: str
    outputs: int
    inputs: int

    def implementation(self, rows, dynamic=False):
        if type(rows) is not int or rows < 1 or type(dynamic) is not bool:
            raise ValueError("Invalid projection rows")
        return "gemv" if rows <= 8 and not dynamic else "tensor_core"


def projection_policies(buffers):
    """Restore selected Q4 roles, including QKV's shared Z rotation.

    Keep W8 sources available for offline reference, but their presence must
    never override the selected Q4 representation in a generated body.
    """
    policies = {}
    for name, b in buffers.items():
        if not name.startswith("Q4_") or not name.endswith("_packed"):
            continue
        base = name.removesuffix("_packed")
        signs = base + "_signs"
        if base.endswith(".attn_qkv"):
            signs = base.removesuffix(".attn_qkv") + ".attn_gate_signs"
        scale = base + "_scale"
        if b["dtype"] != "u16" or len(b["shape"]) != 2:
            raise ValueError(f"Invalid Q4 matrix: {name}")
        n, packed_k = b["shape"]
        k = packed_k * 4
        for role, dtype, shape in ((scale, "f16", [n]), (signs, "i8", [k])):
            operand = buffers.get(role)
            if operand is None or operand["dtype"] != dtype or operand["shape"] != shape:
                raise ValueError(f"Invalid Q4 operand: {role}")
        policies["W_" + base.removeprefix("Q4_") + ".weight_weight"] = ProjectionPolicy(
            name, scale, signs, n, k
        )
    return policies


def expert_implementation(rows, packed_k, *, nibble_book):
    expert_tile_config(rows)
    if packed_k not in (64, 128) or type(nibble_book) is not bool:
        raise ValueError("Invalid expert representation")
    # Verified exact A8 decomposition applies to all small row shapes, not
    # one named program. Larger causal chunks keep the same compressed bank.
    return "exact_a8" if rows <= 8 and packed_k == 64 and nibble_book else "int8"


def expert_tile_config(rows):
    if type(rows) is not int or not 1 <= rows <= 4096:
        raise ValueError("Unsupported expert row count")
    block_m = (
        64 if rows >= 1024 else 16 if rows == 512 else 32 if rows >= 256 else 8 if rows <= 8 else 16
    )
    shortbook = rows <= 8 or rows >= 256
    return dict(
        block_m=block_m,
        block_n=128 if shortbook else 64,
        shortbook=shortbook,
        num_stages=2 if rows <= 4 else 1 if shortbook and block_m <= 32 else 2,
        byte_permute=rows <= 8 or block_m == 64,
        selectorbook=shortbook,
        group_n=rows == 512,
    )


def router_tile(rows, outputs, inputs, experts, hidden):
    return 32 if rows <= 8 and outputs == experts and inputs == hidden else 64


def expert_projection_config(rows, inputs, *, parity_xor=False):
    if type(inputs) is not int or inputs <= 0 or inputs % 128:
        raise ValueError("Unsupported expert input width")
    if type(parity_xor) is not bool:
        raise ValueError("Invalid E8P parity strategy")
    config = expert_tile_config(rows)
    if rows <= 8:
        config.update(
            warp_decode=True,
            block_n=32 if inputs >= 2048 else 64,
            split_k=2 if inputs >= 2048 else 1,
            num_stages=2,
        )
    if rows <= 8 and parity_xor:
        config["parity_xor"] = True
    return config
