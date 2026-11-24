"""HTTP-level end-to-end test: real server on an ephemeral port, real SSE parsing, mock agents."""
from __future__ import annotations

import http.client
import json
import re
import os
import sys
import tempfile
import threading
import time
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
        self.assertIn(b"<title>llmarena</title>", raw)
        self.assertEqual(resp.getheader("Cache-Control"), "no-cache")
        self.assertIn(b'data-mode="chat"', raw)
        resp, raw = self.request("GET", "/api/backends")
        self.assertEqual(json.loads(raw), {"mock": {"available": True, "models": [], "efforts": []}})

    def test_favicon_is_the_logo_svg(self):
        resp, raw = self.request("GET", "/favicon.svg")
        self.assertEqual(resp.status, 200)
        self.assertEqual(resp.getheader("Content-Type"), "image/svg+xml")
        self.assertTrue(raw.startswith(b"<svg"))
        self.assertIn(b"#059669", raw)  # judge check mark of logo E

    def test_index_links_favicon_relatively(self):
        _, raw = self.request("GET", "/")
        self.assertIn(b'<link rel="icon" type="image/svg+xml" href="favicon.svg">', raw)

    def test_index_uses_relative_api_urls_so_it_works_under_a_path_prefix(self):
        # served as https://host/llmarena/ behind Caddy handle_path: "/api/x" would leave the prefix
        _, raw = self.request("GET", "/")
        self.assertIsNone(re.search(rb"""["'`(]/api\b""", raw))
        self.assertIn(b'fetch("api/backends")', raw)

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
            ({"question": "q", "proposer": "mock", "critic": "mock", "judge": "mock",
              "options": {"judge": {"effort": "high"}}}, "options.judge.effort: 'high' not offered by mock"),
            ({"question": "q", "proposer": "mock", "critic": "mock", "judge": "mock", "options": "opus"},
             "options must be an object"),
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



    def finish(self, debate_id: str) -> None:
        """Read the stream to its end, then wait (bounded) for the runner thread to give its slot back:
        the slot is released just after the terminal event."""
        self.read_sse(f"/api/debates/{debate_id}/events")
        deadline = time.monotonic() + 5
        while not self.app.slots.acquire(blocking=False):
            self.assertLess(time.monotonic(), deadline, "slot never released")
            threading.Event().wait(0.01)
        self.app.slots.release()

    def test_user_message_after_verdict_reopens_for_bonus_round(self):
        debate_id = self.start(max_rounds=1)
        self.finish(debate_id)
        resp, raw = self.request("POST", f"/api/debates/{debate_id}/messages", {"text": "Et le coût ?"})
        self.assertEqual(resp.status, 202, raw)
        self.assertEqual(json.loads(raw), {"reopened": True})
        events = self.read_sse(f"/api/debates/{debate_id}/events")
        types = [e["type"] for e in events]
        self.assertEqual(types.count("debate_completed"), 2)
        self.assertEqual(events[-1]["bonus"], 1)
        self.assertIn({"type": "user_message", "text": "Et le coût ?"},
                      [{"type": e["type"], "text": e.get("text")} for e in events if e["type"] == "user_message"])
        self.finish(debate_id)  # the slot taken by the bonus round comes back

    def test_user_message_errors(self):
        self.assertEqual(self.request("POST", "/api/debates/0123456789ab/messages", {"text": "x"})[0].status, 404)
        debate_id = self.start(max_rounds=1)
        self.finish(debate_id)
        self.assertEqual(self.request("POST", f"/api/debates/{debate_id}/messages", {"text": "  "})[0].status, 400)
        self.assertEqual(self.request("POST", f"/api/debates/{debate_id}/messages", {"text": 3})[0].status, 400)
        self.app.get_debate(debate_id).state = "failed"
        self.assertEqual(self.request("POST", f"/api/debates/{debate_id}/messages", {"text": "x"})[0].status, 409)

    def test_user_message_on_finished_debate_needs_a_free_slot(self):
        debate_id = self.start(max_rounds=1)
        self.finish(debate_id)
        self.assertTrue(self.app.slots.acquire(blocking=False))  # max_active=1: now full
        try:
            resp, _ = self.request("POST", f"/api/debates/{debate_id}/messages", {"text": "x"})
            self.assertEqual(resp.status, 429)
            debate = self.app.get_debate(debate_id)
            self.assertEqual((debate.state, debate.pending_user, debate.bonus_rounds), ("completed", [], 0))
        finally:
            self.app.slots.release()


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

    def test_favicon_also_requires_credentials(self):
        self.assertEqual(self.request("GET", "/favicon.svg")[0].status, 401)

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


class ClientIpTests(unittest.TestCase):
    def setUp(self):
        from server import client_ip, parse_trusted_proxies
        self.client_ip = client_ip
        self.trusted = parse_trusted_proxies(["10.203.65.0/28"])

    def test_untrusted_peer_cannot_spoof_forwarded_for(self):
        self.assertEqual(self.client_ip("203.0.113.9", "198.51.100.1", self.trusted), "203.0.113.9")
        self.assertEqual(self.client_ip("203.0.113.9", "198.51.100.1", ()), "203.0.113.9")

    def test_trusted_proxy_uses_rightmost_entry(self):
        # left part is whatever the client sent; the proxy appends the real peer last
        self.assertEqual(self.client_ip("10.203.65.1", "6.6.6.6, 198.51.100.7", self.trusted), "198.51.100.7")

    def test_malformed_or_missing_header_falls_back_to_peer(self):
        for header in (None, "", "not-an-ip", "1.2.3.4, ", "1.2.3.4,::zz"):
            with self.subTest(header=header):
                self.assertEqual(self.client_ip("10.203.65.1", header, self.trusted), "10.203.65.1")

    def test_parse_rejects_garbage(self):
        from server import parse_trusted_proxies
        with self.assertRaises(ValueError):
            parse_trusted_proxies(["10.0.0.0/33"])


class ProxiedAuthServerTests(unittest.TestCase):
    """Behind a trusted proxy, one client's failures must not lock out another client."""

    USER, PASSWORD = AuthServerTests.USER, AuthServerTests.PASSWORD

    def setUp(self):
        from server import BasicAuth, parse_trusted_proxies
        self.runs = tempfile.TemporaryDirectory()
        self.app = App({"mock": MockBackend()}, self.runs.name,
                       auth=BasicAuth(self.USER, self.PASSWORD, max_failures=3, lockout_seconds=60),
                       trusted_proxies=parse_trusted_proxies(["127.0.0.1/32"]))
        self.server = make_server(self.app, "127.0.0.1", 0)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    tearDown = ServerTests.tearDown
    request = ServerTests.request
    basic = staticmethod(AuthServerTests.basic)

    def test_scanner_lockout_is_per_forwarded_client(self):
        scanner = {**self.basic(self.USER, "wrong"), "X-Forwarded-For": "203.0.113.66"}
        for _ in range(3):
            self.assertEqual(self.request("GET", "/", headers=scanner)[0].status, 401)
        self.assertEqual(self.request("GET", "/", headers=scanner)[0].status, 429)
        me = {**self.basic(self.USER, self.PASSWORD), "X-Forwarded-For": "198.51.100.7"}
        self.assertEqual(self.request("GET", "/", headers=me)[0].status, 200)


class LlmAuthRouteTests(unittest.TestCase):
    """/api/auth routes: HTTPS gate, exact AuthManager calls, typed errors mapped to status codes."""

    def setUp(self):
        from auth import AuthManager
        from server import parse_trusted_proxies
        self.runs = tempfile.TemporaryDirectory()
        self.llm = mock.create_autospec(AuthManager, instance=True)
        self.app = App({"mock": MockBackend()}, self.runs.name, llm_auth=self.llm,
                       trusted_proxies=parse_trusted_proxies(["127.0.0.1/32"]))
        self.server = make_server(self.app, "127.0.0.1", 0)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    tearDown = ServerTests.tearDown
    request = ServerTests.request
    HTTPS = {"X-Forwarded-Proto": "https", "X-Forwarded-For": "198.51.100.7"}

    def test_debate_refused_while_a_seat_is_signed_out(self):
        from auth import AuthStatus, UnknownProvider

        def status(agent):
            if agent == "mock":
                raise UnknownProvider(agent)
            return AuthStatus(logged_in=(agent == "claude"))
        self.llm.status.side_effect = status
        resp, raw = self.request("POST", "/api/debates", {"question": "q?", "proposer": "claude", "critic": "codex",
                                                          "judge": "codex", "max_rounds": 1})
        self.assertEqual(resp.status, 409, raw)
        self.assertTrue(json.loads(raw)["error"].startswith("codex non connecté"), raw)
        self.assertEqual([c.args for c in self.llm.status.call_args_list], [("claude",), ("codex",)])  # deduplicated

    def test_debate_with_non_cli_agents_skips_the_login_check(self):
        from auth import UnknownProvider
        self.llm.status.side_effect = UnknownProvider("mock")
        resp, raw = self.request("POST", "/api/debates", {"question": "q?", "proposer": "mock", "critic": "mock",
                                                          "judge": "mock", "max_rounds": 1})
        self.assertEqual(resp.status, 202, raw)
        self.llm.status.assert_called_once_with("mock")

    def test_actions_refused_without_https(self):
        for headers in ({}, {"X-Forwarded-Proto": "http"}, {"X-Forwarded-For": "198.51.100.7"}):
            resp, raw = self.request("POST", "/api/auth/claude/login", {}, headers=headers)
            self.assertEqual(resp.status, 403, (headers, raw))
        self.llm.start_login.assert_not_called()

    def test_login_over_https_calls_manager_once(self):
        self.llm.start_login.return_value = {"state": "awaiting_user", "url": "https://claude.com/x",
                                             "user_code": None, "error": None}
        resp, raw = self.request("POST", "/api/auth/claude/login", {}, headers=self.HTTPS)
        self.assertEqual(resp.status, 200, raw)
        self.assertEqual(json.loads(raw)["url"], "https://claude.com/x")
        self.llm.start_login.assert_called_once_with("claude")

    def test_code_is_forwarded_exactly(self):
        self.llm.submit_code.return_value = {"state": "verifying", "url": None, "user_code": None, "error": None}
        resp, _ = self.request("POST", "/api/auth/claude/code", {"code": "abc#def"}, headers=self.HTTPS)
        self.assertEqual(resp.status, 200)
        self.llm.submit_code.assert_called_once_with("claude", "abc#def")

    def test_typed_errors_map_to_status(self):
        from auth import InvalidAuthCode, NoLoginInProgress, UnknownProvider
        cases = [(UnknownProvider("x"), 404), (NoLoginInProgress("claude"), 409),
                 (InvalidAuthCode("code looks wrong"), 400)]
        for exc, status in cases:
            self.llm.submit_code.side_effect = exc
            resp, raw = self.request("POST", "/api/auth/claude/code", {"code": "c"}, headers=self.HTTPS)
            self.assertEqual(resp.status, status, raw)

    def test_logout_and_cancel(self):
        from auth import AuthStatus
        self.llm.logout.return_value = AuthStatus(False)
        resp, raw = self.request("POST", "/api/auth/codex/logout", {}, headers=self.HTTPS)
        self.assertEqual((resp.status, json.loads(raw)), (200, {"logged_in": False}))
        self.llm.logout.assert_called_once_with("codex")
        resp, _ = self.request("POST", "/api/auth/codex/cancel", {}, headers=self.HTTPS)
        self.assertEqual(resp.status, 200)
        self.llm.cancel.assert_called_once_with("codex")

    def test_overview_is_readable_without_https_gate(self):
        self.llm.overview.return_value = {"claude": {"logged_in": True}}
        resp, raw = self.request("GET", "/api/auth")
        self.assertEqual((resp.status, json.loads(raw)), (200, {"claude": {"logged_in": True}}))

    def test_unknown_action_is_404(self):
        resp, _ = self.request("POST", "/api/auth/claude/hack", {}, headers=self.HTTPS)
        self.assertEqual(resp.status, 404)


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


def deepseek_key_store_for_test(directory):
    from backends import deepseek_key_store
    return deepseek_key_store(directory)


class ApiKeyRouteTests(unittest.TestCase):
    """/api/keys: HTTPS gate, real 0600 store, the key never comes back in a body or a log line."""

    KEY = "sk-0123456789abcdefABCDEFwxyz"

    def setUp(self):
        from backends import DeepSeekBackend, deepseek_key_store
        from server import parse_trusted_proxies
        self.runs = tempfile.TemporaryDirectory()
        self.keys_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.keys_dir.cleanup)
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("DEEPSEEK_API_KEY", None)
        self.store = deepseek_key_store(self.keys_dir.name)
        self.app = App({"mock": MockBackend(), "deepseek": DeepSeekBackend(self.store)}, self.runs.name,
                       trusted_proxies=parse_trusted_proxies(["127.0.0.1/32"]), api_keys={"deepseek": self.store})
        self.server = make_server(self.app, "127.0.0.1", 0)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    tearDown = ServerTests.tearDown
    HTTPS = LlmAuthRouteTests.HTTPS

    def request(self, method, path, body=None, headers=None):
        resp, raw = ServerTests.request(self, method, path, body, headers)
        return resp, raw.decode()

    def backends(self):
        return json.loads(self.request("GET", "/api/backends")[1])

    def test_save_status_clear_round_trip_without_leaking_the_key(self):
        with self.assertLogs("advr.server", "INFO") as server_logs, self.assertLogs("api_keys", "INFO") as key_logs:
            resp, raw = self.request("POST", "/api/keys/deepseek/save", {"key": " " + self.KEY + "\n"},
                                     headers=self.HTTPS)
            self.assertEqual((resp.status, json.loads(raw)), (200, {"configured": True, "source": "file",
                                                                    "hint": "…wxyz"}))
            self.assertTrue(self.backends()["deepseek"]["available"])
            resp, status_raw = self.request("GET", "/api/keys")
            self.assertEqual(json.loads(status_raw), {"deepseek": {"configured": True, "source": "file",
                                                                   "hint": "…wxyz"}})
            resp, raw = self.request("POST", "/api/keys/deepseek/clear", {}, headers=self.HTTPS)
            self.assertEqual((resp.status, json.loads(raw)["configured"]), (200, False))
        self.assertFalse(self.backends()["deepseek"]["available"])
        for line in server_logs.output + key_logs.output:
            self.assertNotIn(self.KEY, line)
        self.assertNotIn(self.KEY, status_raw)

    def test_refused_without_https(self):
        for headers in ({}, {"X-Forwarded-Proto": "http"}):
            for action, body in (("save", {"key": self.KEY}), ("clear", {})):
                resp, raw = self.request("POST", f"/api/keys/deepseek/{action}", body, headers=headers)
                self.assertEqual(resp.status, 403, (headers, action, raw))
                self.assertNotIn(self.KEY, raw)
        self.assertEqual(self.store.get(), "")

    def test_invalid_key_is_400_and_not_echoed(self):
        bad = "sk-not a key; X-Header: 1"
        resp, raw = self.request("POST", "/api/keys/deepseek/save", {"key": bad}, headers=self.HTTPS)
        self.assertEqual(resp.status, 400, raw)
        self.assertNotIn("not a key", raw)
        resp, raw = self.request("POST", "/api/keys/deepseek/save", {"key": 12345}, headers=self.HTTPS)
        self.assertEqual((resp.status, json.loads(raw)), (400, {"error": "key must be a string"}))
        self.assertEqual(self.store.get(), "")

    def test_unwritable_store_is_a_500_json_error_without_the_key(self):
        blocker = os.path.join(self.keys_dir.name, "not-a-dir")
        open(blocker, "w").close()
        self.app.api_keys["deepseek"] = deepseek_key_store_for_test(os.path.join(blocker, "keys"))
        with self.assertLogs("advr.server", "ERROR") as logs:
            resp, raw = self.request("POST", "/api/keys/deepseek/save", {"key": self.KEY}, headers=self.HTTPS)
        self.assertEqual((resp.status, json.loads(raw)), (500, {"error": "could not store the key on the server"}))
        self.assertNotIn(self.KEY, "\n".join(logs.output))

    def test_unknown_provider_and_form_post(self):
        resp, _ = self.request("POST", "/api/keys/claude/save", {"key": self.KEY}, headers=self.HTTPS)
        self.assertEqual(resp.status, 404)
        resp, _ = self.request("POST", "/api/keys/deepseek/save", {"key": self.KEY},
                               headers={**self.HTTPS, "Content-Type": "application/x-www-form-urlencoded"})
        self.assertEqual(resp.status, 415)
        self.assertEqual(self.store.get(), "")



class DisableAgentTests(unittest.TestCase):
    """--disable-agent: a provider that refuses this client disappears from seats AND the account panel."""

    def test_removes_the_agent_from_seats_and_accounts(self):
        from auth import DEFAULT_PROVIDERS
        from server import without_agents
        registry = {"claude": MockBackend(), "gemini": MockBackend(), "mock": MockBackend()}
        seats, providers = without_agents(registry, DEFAULT_PROVIDERS, ["gemini"])
        self.assertEqual(sorted(seats), ["claude", "mock"])
        self.assertNotIn("gemini", providers)
        self.assertIn("claude", providers)
        self.assertIn("gemini", DEFAULT_PROVIDERS)  # the shared default is not mutated

    def test_unknown_or_mock_is_refused(self):
        from server import without_agents
        registry = {"claude": MockBackend(), "mock": MockBackend()}
        for bad in (["gemnii"], ["mock"]):
            with self.assertRaises(ValueError):
                without_agents(registry, {}, bad)

    def test_disabled_agent_cannot_take_a_seat(self):
        from server import without_agents
        seats, _ = without_agents({"gemini": MockBackend(), "mock": MockBackend()}, {}, ["gemini"])
        app = App(seats, tempfile.mkdtemp())
        server = make_server(app, "127.0.0.1", 0)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        conn = __import__("http.client").client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
        conn.request("POST", "/api/debates", json.dumps({"question": "q", "proposer": "gemini", "critic": "mock",
                                                          "judge": "mock", "max_rounds": 1}),
                     {"Content-Type": "application/json"})
        resp = conn.getresponse()
        self.assertEqual(resp.status, 400, resp.read())
        conn.close()


if __name__ == "__main__":
    unittest.main()


class RecordingMock(MockBackend):
    """MockBackend that records prompts and can hold a reply until ``gate`` is set."""

    def __init__(self) -> None:
        super().__init__()
        self.prompts: List[str] = []
        self.gate = threading.Event()
        self.gate.set()

    def complete(self, request):
        self.prompts.append(request.prompt)
        self.gate.wait(5)
        return super().complete(request)


class ChatServerTests(unittest.TestCase):
    """Chat routes over real HTTP + SSE, saved in a real project folder."""

    def setUp(self):
        from projects import ProjectStore
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ProjectStore(self.tmp.name)
        self.store.create("default")
        self.backend = RecordingMock()
        self.app = App({"mock": self.backend}, None, max_active=1, projects=self.store)
        self.server = make_server(self.app, "127.0.0.1", 0)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    request = ServerTests.request
    read_sse = ServerTests.read_sse

    def wait_idle(self, chat_id: str) -> List[Dict]:
        """SSE until the reply lands, then wait (bounded) for the reply thread to free its slot."""
        events = self.read_sse(f"/api/chats/{chat_id}/events")
        deadline = time.monotonic() + 5
        while not self.app.slots.acquire(blocking=False):
            self.assertLess(time.monotonic(), deadline, "slot never released")
            threading.Event().wait(0.01)
        self.app.slots.release()
        return events

    def new_chat(self, text="Retiens 42.") -> str:
        resp, raw = self.request("POST", "/api/chats", {"agent": "mock", "text": text})
        self.assertEqual(resp.status, 202, raw)
        return json.loads(raw)["id"]

    def test_rename_survives_restart_and_empty_title_restores_first_message(self):
        chat_id = self.new_chat("Retiens 42.")
        self.wait_idle(chat_id)
        resp, raw = self.request("PATCH", f"/api/chats/{chat_id}", {"title": "  Mon   nombre  "})
        self.assertEqual(resp.status, 200, raw)
        self.assertEqual(json.loads(raw)["title"], "Mon nombre")
        self.app.chats.clear()  # restart: the custom title must come back from disk
        self.assertEqual([c["title"] for c in self.store.chat_summaries("default")], ["Mon nombre"])
        resp, raw = self.request("GET", f"/api/chats/{chat_id}")
        self.assertEqual(json.loads(raw)["title"], "Mon nombre")
        resp, raw = self.request("PATCH", f"/api/chats/{chat_id}", {"title": ""})
        self.assertEqual((resp.status, json.loads(raw)["title"]), (200, "Retiens 42."))

    def test_rename_errors(self):
        chat_id = self.new_chat()
        self.wait_idle(chat_id)
        for path, body, status in [
            (f"/api/chats/{chat_id}", {"title": 3}, 400),
            (f"/api/chats/{chat_id}", {}, 400),
            (f"/api/chats/{chat_id}", {"title": "x" * 121}, 400),
            ("/api/chats/0123456789ab", {"title": "x"}, 404),
        ]:
            resp, raw = self.request("PATCH", path, body)
            self.assertEqual(resp.status, status, (body, raw))

    def test_delete_removes_chat_from_disk_and_list(self):
        chat_id = self.new_chat()
        self.wait_idle(chat_id)
        resp, raw = self.request("DELETE", f"/api/chats/{chat_id}")
        self.assertEqual((resp.status, raw), (204, b""))
        self.assertEqual(self.store.chat_summaries("default"), [])
        self.assertIsNone(self.store.load_chat(chat_id))
        resp, _ = self.request("GET", f"/api/chats/{chat_id}")
        self.assertEqual(resp.status, 404)
        resp, _ = self.request("DELETE", f"/api/chats/{chat_id}")
        self.assertEqual(resp.status, 404)

    def test_delete_refused_while_agent_answers(self):
        self.backend.gate.clear()
        chat_id = self.new_chat()
        resp, raw = self.request("DELETE", f"/api/chats/{chat_id}")
        self.assertEqual(resp.status, 409, raw)
        self.backend.gate.set()
        self.wait_idle(chat_id)
        self.assertEqual(len(self.store.chat_summaries("default")), 1)

    def test_message_racing_a_delete_is_refused_and_not_written_back(self):
        chat_id = self.new_chat()
        self.wait_idle(chat_id)
        chat = self.app.get_chat(chat_id)
        self.app.delete_chat(chat)  # the handler already holds this object when the delete lands
        from chat import ChatDeleted
        with self.assertRaises(ChatDeleted):
            self.app.post_chat_message(chat, "encore")
        self.assertIsNone(self.store.load_chat(chat_id))
        self.assertTrue(self.app.slots.acquire(blocking=False), "slot leaked")
        self.app.slots.release()

    def test_chat_remembers_and_is_saved_in_project(self):
        chat_id = self.new_chat()
        events = self.wait_idle(chat_id)
        self.assertEqual([(e["type"], (e.get("turn") or {}).get("role")) for e in events],
                         [("chat_started", None), ("turn_completed", "user"), ("turn_completed", "assistant")])
        resp, raw = self.request("POST", f"/api/chats/{chat_id}/messages", {"text": "Quel nombre ?"})
        self.assertEqual((resp.status, json.loads(raw)), (202, {"id": chat_id}))
        self.wait_idle(chat_id)
        self.assertIn("### USER\nRetiens 42.", self.backend.prompts[1])
        self.assertIn("### USER\nQuel nombre ?", self.backend.prompts[1])
        # saved: a fresh App on the same folder (server restart) keeps the memory
        self.app.chats.clear()
        snap = json.loads(self.request("GET", f"/api/chats/{chat_id}")[1])
        self.assertEqual((snap["state"], len(snap["turns"]), snap["title"]), ("idle", 4, "Retiens 42."))
        self.assertEqual(self.request("POST", f"/api/chats/{chat_id}/messages", {"text": "Et alors ?"})[0].status, 202)
        self.wait_idle(chat_id)
        self.assertIn("Quel nombre ?", self.backend.prompts[2])
        listing = json.loads(self.request("GET", "/api/projects/default/chats")[1])
        self.assertEqual([(c["id"], c["agent"], c["messages"]) for c in listing], [(chat_id, "mock", 6)])
        debates = json.loads(self.request("GET", "/api/projects/default/debates")[1])
        self.assertEqual(debates, [])  # chats stay out of the debate list

    def test_message_while_busy_is_409_and_no_slot_is_429(self):
        self.backend.gate.clear()
        chat_id = self.new_chat()
        resp, raw = self.request("POST", f"/api/chats/{chat_id}/messages", {"text": "vite"})
        self.assertEqual(resp.status, 409, raw)
        resp, raw = self.request("POST", "/api/chats", {"agent": "mock", "text": "other"})
        self.assertEqual(resp.status, 429, raw)  # max_active=1, taken by the running reply
        self.backend.gate.set()
        self.wait_idle(chat_id)
        self.assertEqual(len(self.app.get_chat(chat_id).turns), 2)
        self.assertEqual(len(self.app.chats), 1)  # the refused chat was not created

    def test_errors(self):
        cases = [("/api/chats", {"agent": "mock"}, 400),
                 ("/api/chats", {"agent": "mock", "text": "  "}, 400),
                 ("/api/chats", {"agent": "nope", "text": "hi"}, 400),
                 ("/api/chats", {"agent": "mock", "model": "opus", "text": "hi"}, 400),
                 ("/api/chats", {"agent": "mock", "project": "missing", "text": "hi"}, 400),
                 ("/api/chats", {"text": "hi"}, 400),
                 ("/api/chats/0123456789ab/messages", {"text": "hi"}, 404)]
        for path, body, status in cases:
            resp, raw = self.request("POST", path, body)
            self.assertEqual(resp.status, status, (body, raw))
        self.assertEqual(self.request("GET", "/api/chats/0123456789ab")[0].status, 404)
        self.assertEqual(self.request("GET", "/api/chats/0123456789ab/events")[0].status, 404)
        self.assertEqual(self.app.chats, {})
        self.assertTrue(self.app.slots.acquire(blocking=False))  # no slot leaked by the refusals
        self.app.slots.release()

    def test_signed_out_agent_is_refused(self):
        from auth import AuthManager, AuthStatus
        self.app.llm_auth = mock.create_autospec(AuthManager, instance=True)
        self.app.llm_auth.status.return_value = AuthStatus(logged_in=False)
        resp, raw = self.request("POST", "/api/chats", {"agent": "mock", "text": "hi"})
        self.assertEqual(resp.status, 409, raw)
        self.assertTrue(json.loads(raw)["error"].startswith("mock non connecté"), raw)
        self.app.llm_auth.status.assert_called_once_with("mock")
        self.assertEqual(self.backend.prompts, [])

    def test_restart_mid_reply_keeps_message_and_reports_interruption(self):
        from chat import Chat
        self.backend.gate.clear()
        chat_id = self.new_chat("Retiens 42.")
        snap = self.store.load_chat(chat_id)  # on disk while the agent is still answering
        self.assertEqual((snap["state"], [t["content"] for t in snap["turns"]]), ("busy", ["Retiens 42."]))
        # what a restarted server does: reload from disk
        self.app.chats.clear()
        reloaded = self.app.get_chat(chat_id)
        self.assertEqual((reloaded.state, reloaded.turns), ("idle", []))
        last = reloaded.events[-1]
        self.assertEqual((last["type"], last["kind"]), ("reply_failed", "interrupted"))
        self.assertEqual(self.store.load_chat(chat_id)["events"][-1]["seq"], last["seq"])  # persisted, seqs stable
        # the open page reconnects after the user turn and gets that failure, then the stream closes
        events = self.read_sse(f"/api/chats/{chat_id}/events?from={last['seq']}")
        self.assertEqual([e["type"] for e in events], ["reply_failed"])
        self.assertEqual(Chat.from_snapshot(self.store.load_chat(chat_id)).events[-1]["seq"], last["seq"])
        self.assertEqual(json.loads(self.request("GET", "/api/projects/default/chats")[1]), [])  # empty chat not listed
        self.backend.gate.set()
