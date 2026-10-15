"""Build-time decode shapes; serving consumes only compiled row selections."""

from dataclasses import dataclass

CAPACITY = 128
SPECIALIZATIONS = (1, 2, 3, 4, 8, 16, 32)


@dataclass(frozen=True)
class DecodeShape:
    rows: int
    dynamic: bool = False

    @property
    def key(self):
        return "dynamic" if self.dynamic else f"m{self.rows}"

    def ranges(self):
        if not self.dynamic:
            return [(self.rows, self.rows)]
        result, first = [], 1
        for row in (*SPECIALIZATIONS, CAPACITY + 1):
            if first < row:
                result.append((first, row - 1))
            first = row + 1
        return result


def decode_shapes():
    return [DecodeShape(row) for row in SPECIALIZATIONS] + [DecodeShape(CAPACITY, True)]


def selection_name(slot, minimum, maximum, *, skip=False):
    if not 1 <= minimum <= maximum <= CAPACITY:
        raise ValueError("Invalid decode selection range")
    return f"{slot}/{'skip' if skip else 'rows'}/{minimum}-{maximum}"


def q4_configs(rows):
    """Small, legal search space, shared by exact and runtime-row compilation."""
    bm = min(rows, 8)
    return (
        [dict(algorithm="dp4a", block_m=bm, block_n=n, block_k=128, stages=2) for n in (8, 16)]
        + [
            dict(algorithm="tensor_core", block_m=m, block_n=n, block_k=128, stages=2)
            for m in (8, 16, 32, 64)
            for n in (64, 128)
        ]
        + [
            dict(algorithm="tensor_core", block_m=128, block_n=n, block_k=k, stages=s)
            for n in (64, 128)
            for k, s in ((128, 1), (128, 2), (256, 1))
        ]
    )


def w8_configs(rows, outputs):
    if outputs <= 0:
        raise ValueError("Invalid W8 output width")
    return [
        dict(block_m=m, block_n=n, block_k=128) for m in (16, 32, 64, 128) for n in (64, 128)
    ] + [dict(algorithm="dp4a", block_m=min(rows, 4), block_n=n, block_k=128) for n in (4, 8, 16)]


def swiglu_configs(rows):
    return [
        dict(algorithm="tensor_core", block_m=m, block_n=n, block_k=128)
        for m in (16, 32)
        for n in (64, 128)
    ] + [dict(algorithm="dp4a", block_m=min(rows, 4), block_n=n, block_k=128) for n in (4, 8, 16)]
