#!/usr/bin/python3
"""CPU subprocess fixture for real Unix-socket gateway and lifecycle tests."""

import http.server
import json
import os
from pathlib import Path
import signal
import socketserver
import sys
import threading
import time

root = Path(sys.argv[2])
name = sys.argv[sys.argv.index("--model") + 1]
socket = sys.argv[sys.argv.index("--listen") + 1].removeprefix("unix:")
with (root / "starts").open("a") as output:
    output.write(str(os.getpid()) + "\n")
while (root / "hold_start").exists():
    time.sleep(0.005)
if (root / "fail_start").exists():
    sys.exit(2)
write_lock = threading.Lock()


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps(dict(status="ready", model=name)).encode())

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with write_lock, (root / "requests.jsonl").open("a") as output:
            output.write(json.dumps(request) + "\n")
        spec = root / "response.json"
        spec = json.loads(spec.read_text()) if spec.exists() else {}
        streaming = request.get("stream", False)
        self.send_response(spec.get("status", 200))
        self.send_header("Content-Type", "text/event-stream" if streaming else "application/json")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Request-Id", self.headers.get("X-Request-Id", "fixture"))
        self.end_headers()
        try:
            if not streaming:
                self.wfile.write(json.dumps(spec.get("body", request)).encode())
                return
            for event in spec.get("events", [{"choices": [{"delta": {"content": "hello"}}]}]):
                self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
                self.wfile.flush()
            while (root / "hold_response").exists():
                self.wfile.write(b": keepalive\n\n")
                self.wfile.flush()
                time.sleep(0.02)
            if not spec.get("abrupt_eof", False):
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            (root / "disconnected").touch()


class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


signal.signal(signal.SIGTERM, lambda *_args: sys.exit(0))
Server(socket, Handler).serve_forever()
