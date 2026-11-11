"""DeepSeekBackend against a real local HTTP server that mimics the API (no network, no key)."""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backends import BackendError, BackendTimeout, DeepSeekBackend, TurnRequest  # noqa: E402


class FakeDeepSeek(BaseHTTPRequestHandler):
    mode = "ok"
    seen: list = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeDeepSeek.seen.append((self.headers.get("Authorization"), body))
        if self.mode == "slow":
            time.sleep(1.5)
        if self.mode == "401":
            out, code = {"error": {"message": "Authentication Fails", "type": "authentication_error"}}, 401
        elif self.mode == "nochoices":
            out, code = {"choices": []}, 200
        else:
            out, code = {"choices": [{"message": {"role": "assistant", "content": "  réponse  "}}]}, 200
        data = json.dumps(out).encode()
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):  # client already gave up (timeout test)
            pass


class DeepSeekBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeDeepSeek)
        cls.server.daemon_threads = True
        cls.url = f"http://127.0.0.1:{cls.server.server_address[1]}/chat/completions"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        FakeDeepSeek.mode, FakeDeepSeek.seen = "ok", []

    def backend(self, **kw):
        return DeepSeekBackend(api_key="sk-test", url=self.url, **kw)

    def test_sends_exact_request_and_returns_content(self):
        out = self.backend(model="deepseek-reasoner").complete(TurnRequest("critic", 1, "PROMPT"))
        self.assertEqual(out, "réponse")
        self.assertEqual(FakeDeepSeek.seen, [("Bearer sk-test", {
            "model": "deepseek-reasoner", "stream": False, "messages": [{"role": "user", "content": "PROMPT"}]})])

    def test_http_error_surfaces_api_message(self):
        FakeDeepSeek.mode = "401"
        with self.assertRaisesRegex(BackendError, r"^deepseek HTTP 401: Authentication Fails$"):
            self.backend().complete(TurnRequest("critic", 1, "p"))

    def test_malformed_response(self):
        FakeDeepSeek.mode = "nochoices"
        with self.assertRaisesRegex(BackendError, "no choices"):
            self.backend().complete(TurnRequest("critic", 1, "p"))

    def test_timeout(self):
        FakeDeepSeek.mode = "slow"
        with self.assertRaises(BackendTimeout):
            self.backend(timeout=0.3).complete(TurnRequest("critic", 1, "p"))

    def test_unavailable_without_key(self):
        backend = DeepSeekBackend(api_key="")
        self.assertFalse(backend.available())
        with self.assertRaisesRegex(BackendError, "DEEPSEEK_API_KEY"):
            backend.complete(TurnRequest("critic", 1, "p"))


if __name__ == "__main__":
    unittest.main()
