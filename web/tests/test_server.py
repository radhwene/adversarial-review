"""HTTP-level end-to-end test: real server on an ephemeral port, real SSE parsing, mock agents."""
from __future__ import annotations

import http.client
import json
import re
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
        self.assertEqual(json.loads(raw), {"mock": {"available": True, "models": [], "efforts": []}})

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


if __name__ == "__main__":
    unittest.main()
