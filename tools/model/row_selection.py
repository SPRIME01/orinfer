"""Measured dynamic-row selections shared by model package builders."""

import math
from collections import Counter


def project(selection, keys):
    """Coalesce a pipeline choice after restricting it to one component's knobs."""
    configs, indices, names = [], {}, {}
    for name in [selection["default"], *(r["implementation"] for r in selection["selections"])]:
        config = {key: selection["configs"][int(name)][key] for key in keys}
        signature = tuple(config.items())
        if signature not in indices:
            indices[signature] = str(len(configs))
            configs.append(config)
        names[name] = indices[signature]
    default = names[selection["default"]]
    intervals = []
    for row in selection["selections"]:
        name = names[row["implementation"]]
        if name == default:
            continue
        item = dict(row, implementation=name)
        if (
            intervals
            and intervals[-1]["implementation"] == name
            and intervals[-1]["max_rows"] + 1 == item["min_rows"]
            and intervals[-1].get("multiple_of", 1) == item.get("multiple_of", 1)
        ):
            intervals[-1]["max_rows"] = item["max_rows"]
        else:
            intervals.append(item)
    return dict(configs=configs, default=default, selections=intervals)


def prefill_fallback(selection, default, capacity=128):
    """Keep bounded decode choices around one measured prefill configuration."""
    live = [selection["default"]] * capacity
    for choice in selection["selections"]:
        for row in range(choice["min_rows"], min(choice["max_rows"], capacity) + 1):
            if row % choice.get("multiple_of", 1) == 0:
                live[row - 1] = choice["implementation"]
    intervals = []
    for row, name in enumerate(live, 1):
        if name == default:
            continue
        if (
            intervals
            and intervals[-1]["implementation"] == name
            and intervals[-1]["max_rows"] + 1 == row
        ):
            intervals[-1]["max_rows"] = row
        else:
            intervals.append(dict(min_rows=row, max_rows=row, multiple_of=1, implementation=name))
    return dict(
        selection, default=default, selections=intervals, used=sorted(set(live) | {default})
    )


def ranges(records, capacity=4096):
    """Nearest measured length selects a legal implementation, then coalesce."""
    names = sorted({name for row in records for name in row["times"]})
    measured = {
        name: [(r["rows"], r["times"][name]) for r in records if name in r["times"]]
        for name in names
    }

    def cost(name, rows):
        if name == "aligned" and rows % 128:
            return float("inf")
        points = measured[name]
        n, latency = min(points, key=lambda p: abs(math.log(p[0] / rows)))
        return latency * rows / n

    chosen = [min(names, key=lambda name: cost(name, rows)) for rows in range(1, capacity + 1)]
    default = "general" if "general" in names else max(sorted(set(chosen)), key=chosen.count)
    selections = []
    for name in sorted(set(chosen) - {default}):
        multiple = 128 if name == "aligned" else 1
        live = list(range(multiple, capacity + 1, multiple))
        start = end = None
        for row in live:
            if chosen[row - 1] == name:
                if start is None:
                    start = row
                end = row
            elif start is not None:
                selections.append(
                    dict(min_rows=start, max_rows=end, multiple_of=multiple, implementation=name)
                )
                start = end = None
        if start is not None:
            selections.append(
                dict(min_rows=start, max_rows=end, multiple_of=multiple, implementation=name)
            )
    return dict(
        default=default,
        selections=sorted(selections, key=lambda r: r["min_rows"]),
        used=sorted(set(chosen)),
    )


def split_fixed(first, last, fixed):
    """Cut exact AOT points out of a dynamic interval."""
    result = []
    for row in sorted({int(r) for r in fixed if first <= int(r) <= last}):
        if first < row:
            result.append((first, row - 1))
        first = row + 1
    if first <= last:
        result.append((first, last))
    return result


def samples(capacity):
    """Cover geometric extents and tile tails without imposing dispatch tiers."""
    base = min(18, capacity)
    return sorted(
        set(range(1, min(17, capacity) + 1))
        | {round(math.exp(math.log(base) + math.log(capacity / base) * i / 24)) for i in range(25)}
        | {
            n + d
            for n in (32, 64, 128, 256, 512, 1024, 2048, 4096)
            for d in (-1, 0, 1)
            if 1 <= n + d <= capacity
        }
    )


def choices(selection, capacity, fixed=(), *, bounded=False):
    """Publish only used dynamic configs, excluding fixed AOT row extents."""
    live = [selection["default"]] * capacity
    for choice in selection["selections"]:
        step = choice.get("multiple_of", 1)
        for row in range(choice["min_rows"], min(choice["max_rows"], capacity) + 1):
            if row % step == 0:
                live[row - 1] = choice["implementation"]
    remaining = set(range(1, capacity + 1)) - set(fixed)
    usage = Counter(live[r - 1] for r in remaining)
    if not usage:
        return
    default = max(usage, key=lambda key: (usage[key], key))
    if not bounded:
        yield "", selection["configs"][int(default)]
    first = 1
    for row in range(2, capacity + 2):
        if row == capacity + 1 or live[row - 1] != live[first - 1]:
            if bounded or live[first - 1] != default:
                for low, high in split_fixed(first, row - 1, fixed):
                    yield f"/rows/{low}-{high}", selection["configs"][int(live[first - 1])]
            first = row
