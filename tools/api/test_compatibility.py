"""Exercise the real HTTP/SSE acceptance checker with scripted protocol faults."""

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import unittest

from tools.api.compatibility import check_ollama_vectors


@contextmanager
def endpoint(fault=None):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(request)
            tools = bool(request.get("tools"))
            message = {"role": "assistant", "content": None if tools else "sunny"}
            if tools:
                message["tool_calls"] = [
                    {
                        "id": f"call_{len(requests)}",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps(
                                {"location": "Paris" if fault == "wrong_location" else "Boston"}
                            ),
                        },
                    }
                ]
            if fault == "different_history" and not isinstance(
                request["messages"][-1]["content"], list
            ):
                message["content"] = "different history"
            choice = {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tools else "stop",
            }
            usage = {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20}
            self.send_response(200)
            streaming = request.get("stream", False)
            self.send_header(
                "Content-Type", "text/event-stream" if streaming else "application/json"
            )
            self.end_headers()
            if not streaming:
                self.wfile.write(json.dumps({"choices": [choice], "usage": usage}).encode())
                return
            metadata = {
                "id": f"response_{len(requests)}",
                "created": 123,
                "model": request["model"],
                "object": "chat.completion.chunk",
            }

            def event(delta=None, finish=None, metrics=None):
                if fault == "repeated_role" and delta and "content" in delta:
                    delta["role"] = "assistant"
                chunk = dict(
                    metadata, choices=[{"index": 0, "delta": delta, "finish_reason": finish}]
                )
                if metrics is not None:
                    chunk.update(choices=[], usage=metrics)
                if fault == "changed_timestamp" and delta and "content" in delta:
                    chunk["created"] += 1
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())

            event({"role": "assistant"})
            if tools:
                call = message["tool_calls"][0]
                event(
                    {
                        "tool_calls": [
                            dict(call, index=0, function=dict(call["function"], arguments=""))
                        ]
                    }
                )
                arguments = call["function"]["arguments"]
                if fault == "invalid_stream_arguments":
                    arguments = "{"
                for part in (arguments[:4], arguments[4:]):
                    event({"tool_calls": [{"index": 0, "function": {"arguments": part}}]})
            else:
                event({"content": message["content"]})
            event({}, choice["finish_reason"])
            if fault == "wrong_usage":
                usage["completion_tokens"] += 1
            event(metrics=usage)
            if fault != "missing_done":
                self.wfile.write(b"data: [DONE]\n\n")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01))
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)


class OllamaAcceptanceTests(unittest.TestCase):
    def check(self, fault=None):
        evidence = {}
        with endpoint(fault) as (base, requests):
            check_ollama_vectors(base, {"model": "local-alias", "seed": -1}, evidence.__setitem__)
        return evidence, requests

    def test_histories_use_fixed_seed_and_local_model_and_accept_distinct_call_ids(self):
        evidence, requests = self.check()
        self.assertEqual(len(evidence), 3)
        self.assertEqual(len(requests), 9)
        self.assertTrue(all(request["model"] == "local-alias" for request in requests))
        self.assertTrue(all(request["seed"] == 20261002 for request in requests))
        self.assertTrue(all(request["temperature"] == 0 for request in requests))
        self.assertTrue(all(len(case["upstream"]) == 40 for case in evidence.values()))

    def test_bad_history_normalization_is_detected(self):
        with self.assertRaises(AssertionError):
            self.check("different_history")

    def test_wrong_tool_arguments_and_malformed_fragments_are_detected(self):
        for fault in ("wrong_location", "invalid_stream_arguments"):
            with self.subTest(fault=fault), self.assertRaises((AssertionError, ValueError)):
                self.check(fault)

    def test_incorrect_usage_and_missing_sse_terminator_are_detected(self):
        for fault in ("wrong_usage", "missing_done"):
            with self.subTest(fault=fault), self.assertRaises(AssertionError):
                self.check(fault)

    def test_repeated_role_and_changed_timestamp_are_detected(self):
        for fault in ("repeated_role", "changed_timestamp"):
            with self.subTest(fault=fault), self.assertRaises(AssertionError):
                self.check(fault)
