"""agy-relay against a fake ``agy`` (real subprocess, real HTTP server on loopback)."""
from __future__ import annotations

import http.client
import json
import os
import stat
import sys
import tempfile
import textwrap
import threading
import time
import unittest
import unittest.mock
from http.server import ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "agy_relay"))

import relay  # noqa: E402

TOKEN = "t" * 40
EMAIL = "someone.secret@example.com"

# Records argv/stdin/cwd, writes the account e-mail to stderr like the real agy, then
# sleeps FAKE_SLEEP seconds (optionally spawning a child that would outlive it) and replays FAKE_STDOUT.
FAKE_AGY = """\
    #!{python}
    import json, os, subprocess, sys, time
    record = {{"argv": sys.argv[1:], "stdin": sys.stdin.read(), "cwd": os.getcwd()}}
    json.dump(record, open(os.environ["FAKE_RECORD"], "w"))
    sys.stderr.write("signed in as {email}\\n")
    sleep = float(os.environ.get("FAKE_SLEEP", "0"))
    if os.environ.get("FAKE_SPAWN_CHILD"):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        open(os.environ["FAKE_CHILD"], "w").write(str(child.pid))
    time.sleep(sleep)
    sys.stdout.write(os.environ.get("FAKE_STDOUT", ""))
    sys.exit(int(os.environ.get("FAKE_RC", "0")))
"""


def result_line(**result) -> str:
    return json.dumps({"event": "result", "result": result}) + "\n"


INIT = json.dumps({"event": "init", "init": {"model": "gemini-3.8-flash-low"}}) + "\n"
OK_STREAM = INIT + result_line(status="SUCCESS", response="Bonjour", usage={"total_tokens": 12})


class RelayTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.fake = os.path.join(tmp.name, "agy")
        with open(self.fake, "w") as fh:
            fh.write(textwrap.dedent(FAKE_AGY.format(python=sys.executable, email=EMAIL)))
        os.chmod(self.fake, stat.S_IRWXU)
        self.record = os.path.join(tmp.name, "record.json")
        self.child_pid_file = os.path.join(tmp.name, "child.pid")
        saved = dict(os.environ)
        self.addCleanup(lambda: (os.environ.clear(), os.environ.update(saved)))
        os.environ.update(FAKE_RECORD=self.record, FAKE_CHILD=self.child_pid_file,
                          FAKE_STDOUT=OK_STREAM)

    def recorded(self) -> dict:
        with open(self.record) as fh:
            return json.load(fh)


class RunnerTests(RelayTestCase):
    def runner(self, **kw) -> relay.AgyRunner:
        return relay.AgyRunner(self.fake, **kw)

    def test_success_returns_text_and_usage(self):
        answer = self.runner().complete("Salut", "gemini-3.8-flash-low")
        self.assertEqual(answer, relay.AgyAnswer(text="Bonjour", usage={"total_tokens": 12}))

    def test_argv_is_strict_headless_and_denies_every_tool(self):
        self.runner(timeout_s=90).complete("Salut", "gemini-3.8-flash-medium")
        argv = self.recorded()["argv"]
        self.assertEqual(argv[argv.index("--model") + 1], "gemini-3.8-flash-medium")
        settings = json.loads(argv[argv.index("--settings") + 1])
        self.assertEqual(settings["toolPermission"], "strict")
        deny = settings["permissions"]["deny"]
        for tool in ("*", "run_command", "read_file", "view_file", "write_file", "read_url_content"):
            self.assertIn(tool, deny)
        self.assertNotIn("allow", settings["permissions"])
        for flag in ("--disable-slash-commands", "--print=", "--print-timeout=90s"):
            self.assertIn(flag, argv)
        self.assertNotIn("--dangerously-skip-permissions", argv)

    def test_prompt_goes_through_stdin_as_one_ndjson_user_event(self):
        self.runner().complete('ligne 1\nligne "2"', "gemini-3.8-flash-low")
        stdin = self.recorded()["stdin"]
        self.assertEqual(stdin.count("\n"), 1)
        self.assertEqual(json.loads(stdin), {"event": "user", "message": {"content": 'ligne 1\nligne "2"'}})

    def test_each_turn_runs_in_a_fresh_empty_dir_that_is_removed(self):
        self.runner().complete("a", "gemini-3.8-flash-low")
        first = self.recorded()["cwd"]
        self.runner().complete("b", "gemini-3.8-flash-low")
        second = self.recorded()["cwd"]
        self.assertNotEqual(first, second)
        self.assertFalse(os.path.exists(first))

    def test_error_result_is_agent_error_with_email_redacted(self):
        os.environ["FAKE_STDOUT"] = result_line(status="ERROR", error=f"quota for {EMAIL} exceeded")
        os.environ["FAKE_RC"] = "1"
        with self.assertRaises(relay.AgyFailure) as ctx:
            self.runner().complete("x", "gemini-3.8-flash-low")
        self.assertEqual(ctx.exception.kind, "agent_error")
        self.assertNotIn(EMAIL, ctx.exception.message)
        self.assertIn("<redacted>", ctx.exception.message)

    def test_denied_tool_calls_are_counted(self):
        os.environ["FAKE_STDOUT"] = result_line(status="SUCCESS", response="ok",
                                                denied_actions=[{"action": "read_file"}, {"action": "command"}])
        self.assertEqual(self.runner().complete("x", "gemini-3.8-flash-low").denied, 2)

    def test_empty_response_after_denied_tool(self):
        os.environ["FAKE_STDOUT"] = result_line(status="SUCCESS", response="",
                                                denied_actions=[{"action": "command"}])
        with self.assertRaises(relay.AgyFailure) as ctx:
            self.runner().complete("x", "gemini-3.8-flash-low")
        self.assertEqual(ctx.exception.kind, "empty_response")
        self.assertIn("forbidden tool", ctx.exception.message)
        self.assertEqual(ctx.exception.denied, 1)

    def test_missing_result_event_is_protocol_error(self):
        os.environ["FAKE_STDOUT"] = INIT + "not json\n"
        with self.assertRaises(relay.AgyFailure) as ctx:
            self.runner().complete("x", "gemini-3.8-flash-low")
        self.assertEqual(ctx.exception.kind, "protocol")

    def test_nonzero_exit_without_result_is_protocol_error(self):
        os.environ["FAKE_STDOUT"] = ""
        os.environ["FAKE_RC"] = "3"
        with self.assertRaises(relay.AgyFailure) as ctx:
            self.runner().complete("x", "gemini-3.8-flash-low")
        self.assertEqual(ctx.exception.kind, "protocol")

    def test_hung_agy_is_killed_with_its_children(self):
        os.environ.update(FAKE_SLEEP="30", FAKE_SPAWN_CHILD="1")
        runner = self.runner(timeout_s=0)
        started = time.monotonic()
        with self.assertRaises(relay.AgyFailure) as ctx:
            with unittest.mock.patch.object(relay, "KILL_GRACE_S", 1):
                runner.complete("x", "gemini-3.8-flash-low")
        self.assertEqual(ctx.exception.kind, "timeout")
        self.assertLess(time.monotonic() - started, 10)
        with open(self.child_pid_file) as fh:
            child = int(fh.read())
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and _alive(child):
            time.sleep(0.05)
        self.assertFalse(_alive(child), "agy's child process survived the kill")

    def test_second_concurrent_turn_is_busy(self):
        os.environ["FAKE_SLEEP"] = "1.5"
        runner = self.runner(queue_wait_s=0.1)
        first = threading.Thread(target=runner.complete, args=("a", "gemini-3.8-flash-low"))
        first.start()
        self.addCleanup(first.join)
        time.sleep(0.5)
        with self.assertRaises(relay.AgyFailure) as ctx:
            runner.complete("b", "gemini-3.8-flash-low")
        self.assertEqual(ctx.exception.kind, "busy")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie still answers kill(0); count it as dead.
    try:
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().split()[2] != "Z"
    except FileNotFoundError:
        return False


class HttpTests(RelayTestCase):
    def setUp(self):
        super().setUp()
        handler = relay.make_handler(relay.AgyRunner(self.fake), TOKEN)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def call(self, method, path, body=None, token=TOKEN, raw=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_address[1], timeout=10)
        headers = {"Content-Type": "application/json"}
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        payload = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        conn.request(method, path, body=payload, headers=headers)
        resp = conn.getresponse()
        data = json.loads(resp.read() or b"{}")
        conn.close()
        return resp.status, data

    def test_health_needs_no_token(self):
        self.assertEqual(self.call("GET", "/health", token=None), (200, {"ok": True}))

    def test_complete_ok(self):
        status, data = self.call("POST", "/complete", {"prompt": "Salut", "model": "gemini-3.8-flash-low"})
        self.assertEqual((status, data), (200, {"text": "Bonjour", "usage": {"total_tokens": 12}}))

    def test_missing_or_wrong_token_is_401_and_agy_never_runs(self):
        for token in (None, "wrong", TOKEN + "x"):
            status, data = self.call("POST", "/complete", {"prompt": "x", "model": "gemini-3.8-flash-low"},
                                     token=token)
            self.assertEqual(status, 401)
            self.assertEqual(data["error"]["kind"], "unauthorized")
        self.assertFalse(os.path.exists(self.record))

    def test_unknown_model_is_400(self):
        status, data = self.call("POST", "/complete", {"prompt": "x", "model": "claude-opus-4-6-thinking"})
        self.assertEqual((status, data["error"]["kind"]), (400, "bad_request"))
        self.assertFalse(os.path.exists(self.record))

    def test_bad_bodies_are_400(self):
        for raw in (b"not json", b"[1]", json.dumps({"prompt": "", "model": "gemini-3.8-flash-low"}).encode(),
                    json.dumps({"prompt": "x" * (relay.MAX_PROMPT_CHARS + 1),
                                "model": "gemini-3.8-flash-low"}).encode()):
            status, data = self.call("POST", "/complete", raw=raw)
            self.assertEqual((status, data["error"]["kind"]), (400, "bad_request"), raw[:20])

    def test_stderr_email_never_reaches_the_client(self):
        os.environ["FAKE_STDOUT"] = result_line(status="ERROR", error="authentication failed")
        os.environ["FAKE_RC"] = "1"
        status, data = self.call("POST", "/complete", {"prompt": "x", "model": "gemini-3.8-flash-low"})
        self.assertEqual((status, data["error"]["kind"]), (502, "agent_error"))
        self.assertNotIn(EMAIL, json.dumps(data))

    def test_access_log_counts_denied_tools_on_failure(self):
        os.environ["FAKE_STDOUT"] = result_line(status="SUCCESS", response="",
                                                denied_actions=[{"action": "read_file"}])
        with self.assertLogs("agy-relay", level="INFO") as logs:
            status, _ = self.call("POST", "/complete", {"prompt": "x", "model": "gemini-3.8-flash-low"})
        self.assertEqual(status, 502)
        self.assertIn("outcome=empty_response denied=1", logs.output[0])

    def test_access_log_has_no_prompt(self):
        with self.assertLogs("agy-relay", level="INFO") as logs:
            self.call("POST", "/complete", {"prompt": "SECRET-PROMPT", "model": "gemini-3.8-flash-low"})
        self.assertEqual(len(logs.output), 1)
        self.assertIn("outcome=ok denied=0", logs.output[0])
        self.assertNotIn("SECRET-PROMPT", logs.output[0])


class MainTests(unittest.TestCase):
    def test_short_token_refuses_to_start(self):
        with unittest.mock.patch.dict(os.environ, {"AGY_RELAY_TOKEN": "short"}):
            with self.assertLogs("agy-relay", level="ERROR"):
                self.assertEqual(relay.main(["--port", "0"]), 2)


if __name__ == "__main__":
    unittest.main()
