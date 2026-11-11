"""HTTP-level end-to-end test: real server on an ephemeral port, real SSE parsing, mock agents."""
from __future__ import annotations

import http.client
import json
import os
import sys
import tempfile
import threading
import unittest
from typing import Dict, List
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backends import MockBackend  # noqa: E402
from server import App, make_server  # noqa: E402


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.runs = tempfile.TemporaryDirectory()
        self.app = App({"mock": MockBackend()}, self.runs.name, max_active=1)
        self.server = make_server(self.app, "127.0.0.1", 0)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.runs.cleanup()

    def request(self, method: str, path: str, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        data = json.dumps(body).encode() if body is not None else None
        conn.request(method, path, body=data, headers={"Content-Type": "application/json", **(headers or {})})
        resp = conn.getresponse()
        return resp, resp.read()

    def start(self, **kw) -> str:
        body = dict(question="Is P true?", proposer="mock", critic="mock", judge="mock", max_rounds=3)
        body.update(kw)
        resp, raw = self.request("POST", "/api/debates", body)
        self.assertEqual(resp.status, 202, raw)
        return json.loads(raw)["id"]

    def read_sse(self, path: str, headers=None) -> List[Dict]:
        resp, raw = self.request("GET", path, headers=headers)
        self.assertEqual(resp.status, 200)
        self.assertEqual(resp.getheader("Content-Type"), "text/event-stream")
        events = []
        for chunk in raw.decode().split("\n\n"):
            data = [line[len("data: "):] for line in chunk.splitlines() if line.startswith("data: ")]
            if data:
                events.append(json.loads(data[0]))
        return events

    def test_index_and_backends(self):
        resp, raw = self.request("GET", "/")
        self.assertEqual(resp.status, 200)
        self.assertIn(b"<title>Adversarial Reasoning</title>", raw)
        resp, raw = self.request("GET", "/api/backends")
        self.assertEqual(json.loads(raw), {"mock": {"available": True}})

    def test_debate_streams_to_completion_twice(self):
        for _ in range(2):  # anti-flake: two full runs in a row
            debate_id = self.start()
            events = self.read_sse(f"/api/debates/{debate_id}/events")
            self.assertEqual([e["type"] for e in events], [
                "debate_started",
                "turn_started", "turn_completed",  # proposer opening
                "turn_started", "turn_completed",  # critic r1 CONTEST
                "turn_started", "turn_completed",  # proposer r1
                "turn_started", "turn_completed",  # critic r2 ACCEPT
                "converged",
                "turn_started", "turn_completed",  # judge
                "debate_completed",
            ])
            resp, raw = self.request("GET", f"/api/debates/{debate_id}")
            snap = json.loads(raw)
            self.assertEqual(snap["state"], "completed")
            self.assertEqual(len(snap["turns"]), 5)
            self.assertTrue(os.path.exists(os.path.join(self.runs.name, f"{debate_id}.json")))

    def test_replay_from_last_event_id(self):
        debate_id = self.start()
        full = self.read_sse(f"/api/debates/{debate_id}/events")
        tail = self.read_sse(f"/api/debates/{debate_id}/events", headers={"Last-Event-ID": "9"})
        self.assertEqual(tail, full[10:])
        self.assertEqual(self.read_sse(f"/api/debates/{debate_id}/events?from=12"), full[12:])

    def test_bad_requests(self):
        cases = [
            ({"question": "q", "proposer": "mock", "critic": "nope", "judge": "mock"}, "unknown backend"),
            ({"question": "", "proposer": "mock", "critic": "mock", "judge": "mock"}, "empty"),
            ({"question": "q", "proposer": "mock"}, "invalid request"),
            ({"question": "q", "proposer": "mock", "critic": "mock", "judge": "mock", "max_rounds": 9}, "max_rounds"),
        ]
        for body, msg in cases:
            with self.subTest(body=body):
                resp, raw = self.request("POST", "/api/debates", body)
                self.assertEqual(resp.status, 400)
                self.assertIn(msg, json.loads(raw)["error"])

    def test_post_rejects_non_json_content_type(self):
        body = {"question": "q", "proposer": "mock", "critic": "mock", "judge": "mock"}
        for ctype in ("text/plain", "application/x-www-form-urlencoded", "multipart/form-data"):
            with self.subTest(ctype=ctype):
                resp, raw = self.request("POST", "/api/debates", body, headers={"Content-Type": ctype})
                self.assertEqual(resp.status, 415)
        self.assertEqual(self.app.debates, {})
        resp, _ = self.request("POST", "/api/debates", body,
                               headers={"Content-Type": "application/json; charset=utf-8"})
        self.assertEqual(resp.status, 202)

    def test_foreign_host_header_rejected(self):
        for path in ("/", "/api/backends"):
            with self.subTest(path=path):
                resp, raw = self.request("GET", path, headers={"Host": "evil.example:8765"})
                self.assertEqual(resp.status, 403)
        body = {"question": "q", "proposer": "mock", "critic": "mock", "judge": "mock"}
        resp, _ = self.request("POST", "/api/debates", body, headers={"Host": "evil.example"})
        self.assertEqual(resp.status, 403)
        self.assertEqual(self.app.debates, {})
        for ok in ("localhost:8765", "127.0.0.1", "[::1]:8765"):
            with self.subTest(host=ok):
                self.assertEqual(self.request("GET", "/api/backends", headers={"Host": ok})[0].status, 200)

    def test_unknown_routes_404(self):
        self.assertEqual(self.request("GET", "/api/debates/0123456789ab/events")[0].status, 404)
        self.assertEqual(self.request("GET", "/nope")[0].status, 404)

    def test_concurrency_cap_returns_429(self):
        gate = threading.Event()

        class Blocking(MockBackend):
            def complete(self, request):
                gate.wait(5)
                return super().complete(request)

        self.app.registry["mock"] = Blocking()
        self.app.runner.registry = self.app.registry
        first = self.start()
        resp, raw = self.request("POST", "/api/debates",
                                 {"question": "q", "proposer": "mock", "critic": "mock", "judge": "mock"})
        self.assertEqual(resp.status, 429)
        gate.set()
        self.assertEqual(self.read_sse(f"/api/debates/{first}/events")[-1]["type"], "debate_completed")



class AuthServerTests(unittest.TestCase):
    """Same server, with HTTP Basic auth enabled."""

    USER, PASSWORD = "alice", "correct-horse-battery"

    def setUp(self):
        from server import BasicAuth
        self.runs = tempfile.TemporaryDirectory()
        self.auth = BasicAuth(self.USER, self.PASSWORD, max_failures=3, lockout_seconds=60)
        self.app = App({"mock": MockBackend()}, self.runs.name, auth=self.auth)
        self.server = make_server(self.app, "127.0.0.1", 0)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    tearDown = ServerTests.tearDown
    request = ServerTests.request

    @staticmethod
    def basic(user, password):
        import base64
        return {"Authorization": "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()}

    def test_no_credentials_gets_401_challenge(self):
        resp, _ = self.request("GET", "/")
        self.assertEqual(resp.status, 401)
        self.assertEqual(resp.getheader("WWW-Authenticate"), 'Basic realm="Adversarial Reasoning", charset="UTF-8"')

    def test_wrong_password_and_wrong_user_rejected(self):
        for headers in (self.basic(self.USER, "nope"), self.basic("bob", self.PASSWORD),
                        {"Authorization": "Basic !!!notbase64"}, {"Authorization": "Bearer x"}):
            with self.subTest(headers=headers):
                self.assertEqual(self.request("GET", "/api/backends", headers=headers)[0].status, 401)
                self.auth._failures.clear()

    def test_valid_credentials_full_debate_with_any_host(self):
        h = self.basic(self.USER, self.PASSWORD)
        self.assertEqual(self.request("GET", "/", headers={**h, "Host": "203.0.113.7:8765"})[0].status, 200)
        resp, raw = self.request("POST", "/api/debates",
                                 {"question": "q", "proposer": "mock", "critic": "mock", "judge": "mock"}, headers=h)
        self.assertEqual(resp.status, 202)
        resp, raw = self.request("GET", f"/api/debates/{json.loads(raw)['id']}/events", headers=h)
        self.assertEqual(resp.status, 200)
        self.assertIn(b'"type": "debate_completed"', raw)

    def test_csrf_check_still_applies_when_authenticated(self):
        h = {**self.basic(self.USER, self.PASSWORD), "Content-Type": "text/plain"}
        resp, _ = self.request("POST", "/api/debates",
                               {"question": "q", "proposer": "mock", "critic": "mock", "judge": "mock"}, headers=h)
        self.assertEqual(resp.status, 415)
        self.assertEqual(self.app.debates, {})

    def test_lockout_after_repeated_failures_blocks_even_good_password(self):
        bad = self.basic(self.USER, "wrong")
        for _ in range(3):
            self.assertEqual(self.request("GET", "/", headers=bad)[0].status, 401)
        self.assertEqual(self.request("GET", "/", headers=self.basic(self.USER, self.PASSWORD))[0].status, 429)

    def test_probe_without_header_does_not_count_as_failure(self):
        for _ in range(5):
            self.request("GET", "/")
        self.assertEqual(self.request("GET", "/", headers=self.basic(self.USER, self.PASSWORD))[0].status, 200)


class BasicAuthConfigTests(unittest.TestCase):
    def test_rejects_short_password_and_empty_user(self):
        from server import BasicAuth
        for user, pw in (("u", "short"), ("", "long-enough-password")):
            with self.subTest(user=user), self.assertRaises(ValueError):
                BasicAuth(user, pw)

    def test_refuses_public_bind_without_auth(self):
        from server import main
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ADVR_USER", None)
            os.environ.pop("ADVR_PASSWORD", None)
            with self.assertRaises(SystemExit) as ctx, mock.patch("sys.stderr"):
                main(["--host", "0.0.0.0", "--port", "0"])
        self.assertEqual(ctx.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
