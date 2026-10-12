"""Validated static tiles shared by initial AOT builds and package upgrades."""


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
