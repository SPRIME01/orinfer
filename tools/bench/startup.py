"""Measure release CLI startup through the first complete streaming Chat request.

The parent holds the shared GPU lock until its child exits. File caches are left
unchanged. First output is a nonempty text/reasoning/tool delta, not a token count.
"""

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import time
import urllib.request

from tools.bench.concurrency import events

ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def gpu_owner(path, timeout):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as exc:
                if time.monotonic() >= deadline:
                    raise RuntimeError("GPU lock is busy") from exc
                time.sleep(0.01)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def stream(opener, base, body, started, timeout):
    request = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(dict(body, stream=True, stream_options={"include_usage": True})).encode(),
        headers={"Content-Type": "application/json"},
    )
    if os.getenv("ORINFER_API_KEY"):
        request.add_header("Authorization", "Bearer " + os.environ["ORINFER_API_KEY"])
    first, usage, finish, done = None, None, None, False
    content, reasoning = [], []
    with opener.open(request, timeout=timeout) as response:
        for event in events(response):
            if time.monotonic() - started > timeout:
                raise TimeoutError("First request exceeded the startup timeout")
            if event == "[DONE]":
                done = True
                break
            value = json.loads(event)
            if value.get("error"):
                raise RuntimeError(value["error"])
            usage = value.get("usage") or usage
            for choice in value.get("choices", []):
                delta = choice.get("delta", {})
                if first is None and any(
                    delta.get(key) for key in ["content", "reasoning_content", "tool_calls"]
                ):
                    first = time.monotonic() - started
                content.append(delta.get("content") or "")
                reasoning.append(delta.get("reasoning_content") or "")
                finish = choice.get("finish_reason") or finish
    if not done or first is None or not finish or not usage:
        raise ValueError("Incomplete first response or missing usage")
    return dict(
        process_to_first_output_s=first,
        process_to_complete_s=time.monotonic() - started,
        usage=usage,
        finish_reason=finish,
        content="".join(content),
        reasoning="".join(reasoning),
    )


def trial(engine, model, body, output, index, server_args, timeout):
    # The caller owns the global lock; the child's private lock remains held for
    # its model lifetime without conflicting with that enclosing GPU ownership.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    log_path = output / f"trial-{index}.log"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with log_path.open("w") as log:
        started = time.monotonic()
        process = subprocess.Popen(
            [
                str(engine),
                "serve",
                str(model),
                "--model",
                body["model"],
                "--listen",
                f"127.0.0.1:{port}",
                "--gpu-lock",
                str(output / f"trial-{index}.lock"),
                *server_args,
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            base = f"http://127.0.0.1:{port}"
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited during startup: {log_path}")
                if time.monotonic() - started >= timeout:
                    raise TimeoutError(f"Server readiness timeout: {log_path}")
                try:
                    with opener.open(base + "/health", timeout=0.2) as response:
                        health = json.load(response)
                except (OSError, ValueError):
                    time.sleep(0.01)
                    continue
                if health.get("model") != body["model"]:
                    raise ValueError("Unexpected service model on benchmark port")
                if health.get("status") == "ready":
                    break
                time.sleep(0.01)
            ready = time.monotonic() - started
            result = stream(opener, base, body, started, timeout)
            result.update(trial=index, process_to_ready_s=ready, log=str(log_path))
            return result
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, default=ROOT / "target/release/orinfer")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--gpu-lock", type=Path, default=ROOT / "artifacts/gpu-experiment.lock")
    parser.add_argument("--gpu-lock-timeout", type=float, default=0)
    parser.add_argument("--server-args", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()
    if not 1 <= args.runs <= 100 or args.timeout <= 0 or args.gpu_lock_timeout < 0:
        parser.error("Require 1..100 runs, positive timeout and nonnegative GPU lock timeout")
    engine, model, output = (
        args.engine.resolve(strict=True),
        args.model.resolve(strict=True),
        args.output.absolute(),
    )
    body = json.loads(args.request.read_text())
    body.setdefault("model", "startup-bench")
    body.setdefault("seed", 20261002)
    output.mkdir(parents=True, exist_ok=False)
    identity = hashlib.sha256((model / "cache/model.json").read_bytes()).hexdigest()
    rows = []
    for index in range(args.runs):
        with gpu_owner(args.gpu_lock, args.gpu_lock_timeout):
            rows.append(trial(engine, model, body, output, index, args.server_args, args.timeout))
        report = dict(
            engine=str(engine),
            model=str(model),
            model_sha256=identity,
            request=body,
            server_args=args.server_args,
            measurements=rows,
            median={
                key: statistics.median(row[key] for row in rows)
                for key in [
                    "process_to_ready_s",
                    "process_to_first_output_s",
                    "process_to_complete_s",
                ]
            },
            scope="CLI process to HTTP readiness, first nonempty SSE output and complete first response; existing file caches preserved; GPU lock waiting and shutdown excluded.",
        )
        (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["median"], indent=2))


if __name__ == "__main__":
    main()
