"""GPU sampling contract, tails, deterministic radix and changed-input replay."""

import argparse
import json
from pathlib import Path
import torch
from tools.operators.common import benchmark, configure, export_kernel, environment
from kernels.model import sampling as s


class Pipeline:
    def __init__(self, vocab, context=1024):
        self.vocab, self.context = vocab, context
        blocks = (vocab + s.TILE - 1) // s.TILE
        self.kernels = {
            name: getattr(s, name)(vocab, *([context] if name == "history" else []))
            for name in (
                "initialize",
                "history",
                "scores",
                "masses",
                "radix_histogram",
                "radix_choose",
            )
        }

        def empty(shape, dtype):
            return torch.empty(shape, dtype=dtype, device="cuda")

        self.counts = empty((vocab,), torch.int32)
        self.bias = empty((vocab,), torch.float64)
        self.history = empty((context,), torch.int32)
        self.bias_ids = empty((vocab,), torch.int32)
        self.bias_values = empty((vocab,), torch.float64)
        self.control = empty((4,), torch.int32)
        self.parameters = empty((6,), torch.float64)
        self.mask = empty(((vocab + 31) // 32,), torch.uint32)
        self.scores = empty((vocab,), torch.float64)
        self.keys = empty((vocab,), torch.uint64)
        self.maximum = empty((blocks,), torch.float64)
        self.invalid = empty((blocks,), torch.int32)
        self.mass = empty((vocab,), torch.uint64)
        self.state = empty((7,), torch.uint64)
        self.hist = empty((blocks, 256), torch.uint64)
        self.output = empty((2,), torch.int32)
        self.logits = empty((vocab,), torch.float32)

    def run(self):
        k = self.kernels
        k["initialize"](self.counts, self.bias)
        k["history"](
            self.history, self.bias_ids, self.bias_values, self.control, self.counts, self.bias
        )
        k["scores"](
            self.logits,
            self.counts,
            self.bias,
            self.mask,
            self.control,
            self.parameters,
            self.scores,
            self.keys,
            self.maximum,
            self.invalid,
        )
        k["masses"](
            self.scores,
            self.keys,
            self.maximum,
            self.invalid,
            self.parameters,
            self.mass,
            self.output,
        )
        for mode in range(3):
            for digit in range(12):
                k["radix_histogram"](
                    self.keys,
                    self.mass,
                    self.state,
                    self.control,
                    self.parameters,
                    self.hist,
                    digit,
                    mode,
                )
                k["radix_choose"](
                    self.hist, self.state, self.control, self.parameters, self.output, digit, mode
                )

    def inputs(
        self,
        logits,
        top_k=0,
        top_p=1.0,
        uniform=0.37,
        temperature=0.8,
        penalties=(0.0, 0.0, 1.0),
        history=(),
        bias=None,
        mask=None,
    ):
        self.logits.copy_(logits)
        self.control.copy_(
            torch.tensor(
                [len(history), len(bias or {}), int(mask is not None), top_k], dtype=torch.int32
            )
        )
        self.parameters.copy_(
            torch.tensor([temperature, top_p, *penalties, uniform], dtype=torch.float64)
        )
        if history:
            self.history[: len(history)].copy_(torch.tensor(history, dtype=torch.int32))
        if bias:
            self.bias_ids[: len(bias)].copy_(torch.tensor(list(bias), dtype=torch.int32))
            self.bias_values[: len(bias)].copy_(
                torch.tensor(list(bias.values()), dtype=torch.float64)
            )
        if mask is not None:
            words = [
                sum(int(mask[i]) << (i % 32) for i in range(j, min(j + 32, self.vocab)))
                for j in range(0, self.vocab, 32)
            ]
            self.mask.copy_(torch.tensor(words, dtype=torch.uint32))


def reference(logits, k, p, u, temp, penalties, history, bias, mask):
    values = logits.double().cpu()
    counts = torch.bincount(torch.tensor(history, dtype=torch.int64), minlength=len(values))
    presence, frequency, repetition = penalties
    values = torch.where(
        counts > 0,
        torch.where(values < 0, values * repetition, values / repetition) - presence,
        values,
    )
    values -= counts * frequency
    for token, value in bias.items():
        values[token] += value
    ids = [i for i in range(len(values)) if mask is None or mask[i]]
    ids.sort(key=lambda i: (-values[i].item(), i))
    if k:
        ids = ids[:k]
    mass = torch.exp((values[ids] - values[ids[0]]) / temp)
    cutoff = int(torch.searchsorted(mass.cumsum(0), mass.sum() * p))
    ids, mass = ids[: cutoff + 1], mass[: cutoff + 1]
    index = min(int(torch.searchsorted((mass / mass.sum()).cumsum(0), u, right=True)), len(ids) - 1)
    return ids[index]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sanitizer", action="store_true", help="Skip instrumented timing repetitions")
    a = p.parse_args()
    configure()
    # Establish the primary context before NVRTC/cuda-python adapter discovery.
    torch.empty((), device="cuda")
    reports = []
    for vocab in (35, 1057, 248320):
        pipe = Pipeline(vocab)
        cases = 0
        gen = torch.Generator().manual_seed(20261002)
        settings = (
            ((0, 1.0), (7, 0.8))
            if a.sanitizer
            else ((0, 1.0), (0, 0.9), (1, 0.4), (7, 0.8), (vocab, 0.95))
        )
        for k, nucleus in settings:
            for uniform in (0.17,) if a.sanitizer else (0.0, 0.17, 0.73, 0.999999):
                logits = torch.randn(vocab, generator=gen) * 3
                logits[:4] = 0  # ties, including when k cuts a tied group
                penalties = (0.2, 0.13, 1.1)
                history = [0, 0, 2, vocab - 1]
                bias = {3: 0.7, vocab - 1: -0.2}
                mask = [i % 3 != 1 for i in range(vocab)]
                pipe.inputs(
                    logits,
                    k,
                    nucleus,
                    uniform,
                    penalties=penalties,
                    history=history,
                    bias=bias,
                    mask=mask,
                )
                pipe.run()
                actual = pipe.output.cpu().tolist()
                expected = reference(
                    logits, k, nucleus, uniform, 0.8, penalties, history, bias, mask
                )
                assert actual == [expected, 0], (vocab, k, nucleus, uniform, actual, expected)
                cases += 1
        # A tied vocabulary checks exact token-ID ordering at both cutoffs.
        logits = torch.zeros(vocab)
        pipe.inputs(logits, 7, 0.5, 0.73)
        pipe.run()
        assert pipe.output.cpu().tolist() == [2, 0]
        if a.sanitizer:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                pipe.run()
            latency = None
        else:
            latency, graph = benchmark(pipe.run, repetitions=10)
        pipe.inputs(torch.arange(vocab, dtype=torch.float32), 1)
        graph.replay()
        assert pipe.output.cpu().tolist() == [vocab - 1, 0]
        bad = torch.zeros(vocab)
        bad[-1] = float("nan")
        pipe.inputs(bad)
        graph.replay()
        assert pipe.output.cpu().tolist()[1] == 1
        pipe.inputs(torch.zeros(vocab), mask=[False] * vocab)
        graph.replay()
        assert pipe.output.cpu().tolist()[1] == 1
        for name, kernel in pipe.kernels.items():
            export_kernel(kernel, a.output / str(vocab) / name)
        reports.append(dict(vocab=vocab, cases=cases, changed_input_replay=True, timing=latency))
        print(json.dumps(reports[-1]), flush=True)
    (a.output / "results.json").write_text(
        json.dumps(dict(environment=environment(), cases=reports), indent=2)
    )


if __name__ == "__main__":
    main()
