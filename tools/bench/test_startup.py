"""Exercise real local HTTP/SSE timing and subprocess cleanup without CUDA."""

import os
from pathlib import Path
import sys
import tempfile
import unittest

from tools.bench.startup import gpu_owner, trial


class StartupTests(unittest.TestCase):
    def fixture(self, root):
        lock = root / "gpu.lock"
        engine = root / "engine.py"
        engine.write_text(
            f"#!{sys.executable}\n"
            + f"LOCK = {str(lock)!r}\n"
            + """
import fcntl, json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
if '--fail' in sys.argv:
    sys.exit(2)
address = sys.argv[sys.argv.index('--listen') + 1]
model = sys.argv[sys.argv.index('--model') + 1]
Path(sys.argv[2], 'pid').write_text(str(os.getpid()))
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        payload = json.dumps({'status':'ready', 'model':model}).encode()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(payload)
    def do_POST(self):
        self.rfile.read(int(self.headers['Content-Length']))
        with open(LOCK, 'a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                raise RuntimeError('Parent must own the shared GPU lock')
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.end_headers()
        values = [
            {'choices':[{'delta':{'role':'assistant'}}]},
            {'choices':[{'delta':{'content':'hello'}}]},
            {'choices':[{'delta':{},'finish_reason':'stop'}]},
            {'choices':[], 'usage':{'prompt_tokens':3,'completion_tokens':2}},
        ]
        for value in values:
            self.wfile.write(('data: '+json.dumps(value)+'\\n\\n').encode())
        self.wfile.write(b'data: [DONE]\\n\\n')
host, port = address.rsplit(':', 1)
try:
    HTTPServer((host, int(port)), Handler).serve_forever()
except KeyboardInterrupt:
    pass
"""
        )
        engine.chmod(0o700)
        return engine, lock

    def test_real_process_timing_waits_for_output_and_reaps_child(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine, lock = self.fixture(root)
            with gpu_owner(lock, 0):
                result = trial(engine, root, {"model": "fixture"}, root, 0, [], 5)
            self.assertEqual(result["content"], "hello")
            self.assertEqual(result["usage"]["completion_tokens"], 2)
            self.assertEqual(result["finish_reason"], "stop")
            self.assertLessEqual(result["process_to_ready_s"], result["process_to_first_output_s"])
            self.assertLessEqual(
                result["process_to_first_output_s"], result["process_to_complete_s"]
            )
            with self.assertRaises(ProcessLookupError):
                os.kill(int((root / "pid").read_text()), 0)

    def test_early_server_failure_releases_the_shared_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine, lock = self.fixture(root)
            with self.assertRaisesRegex(RuntimeError, "Server exited"):
                with gpu_owner(lock, 0):
                    trial(engine, root, {"model": "fixture"}, root, 0, ["--fail"], 5)
            with gpu_owner(lock, 0):
                pass

    def test_busy_lock_fails_without_starting_a_child(self):
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / "gpu.lock"
            with gpu_owner(lock, 0):
                with self.assertRaisesRegex(RuntimeError, "GPU lock is busy"):
                    with gpu_owner(lock, 0):
                        self.fail("Foreign ownership must be preserved")


if __name__ == "__main__":
    unittest.main()
