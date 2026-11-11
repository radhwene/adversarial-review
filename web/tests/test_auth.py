"""AuthManager against scripted fake CLIs (real subprocesses, real pipes)."""
import os
import sys
import tempfile
import textwrap
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import auth  # noqa: E402
from auth import (AWAITING_USER, COMPLETED, EXPIRED, FAILED, SUBSCRIPTION, API_BILLING, AuthManager,  # noqa: E402
                  InvalidAuthCode, NoLoginInProgress, ProviderSpec, UnknownProvider,
                  parse_claude_status, parse_codex_status)

GOOD_CODE = "AbCdEf0123456789_-xyz#StAtE-0123456789"

FAKE_CLAUDE = r'''
import sys
print("Opening browser to sign in…", flush=True)
print("\x1b[2mIf the browser didn't open, visit:\x1b[0m https://evil.example/phish", flush=True)
print("If the browser didn't open, visit: https://claude.com/cai/oauth/authorize?code=true&state=s1", flush=True)
sys.stdout.write("Paste code here if prompted > "); sys.stdout.flush()
code = sys.stdin.readline().strip()
open(sys.argv[1], "w").write(code)
if code != sys.argv[2]:
    print("OAuth error: invalid code " + code, flush=True)
    sys.exit(1)
print("Login successful.", flush=True)
'''

FAKE_CODEX = r'''
import sys, time
print("\x1b[1mWelcome to Codex\x1b[0m", flush=True)
print("   \x1b[94mhttps://auth.openai.com/codex/device\x1b[0m", flush=True)
print("   \x1b[94mBTI3-CQUN6\x1b[0m", flush=True)
time.sleep(float(sys.argv[1]))
sys.exit(int(sys.argv[2]))
'''


def wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


class AuthManagerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        d = self.tmp.name
        self.claude_py = os.path.join(d, "fake_claude.py")
        self.codex_py = os.path.join(d, "fake_codex.py")
        self.received = os.path.join(d, "received_code")
        with open(self.claude_py, "w") as fh:
            fh.write(textwrap.dedent(FAKE_CLAUDE))
        with open(self.codex_py, "w") as fh:
            fh.write(textwrap.dedent(FAKE_CODEX))

    def manager(self, codex_sleep="5", codex_rc="0", timeout=30.0):
        exe = sys.executable
        providers = {
            "claude": ProviderSpec(
                login_argv=(exe, self.claude_py, self.received, GOOD_CODE),
                status_argv=(exe, "-c", 'print(\'{"loggedIn": false}\'); raise SystemExit(1)'),
                logout_argv=(exe, "-c", "pass"),
                parse_status=parse_claude_status,
                url_hosts=frozenset({"claude.com", "claude.ai"}), takes_code=True),
            "codex": ProviderSpec(
                login_argv=(exe, self.codex_py, codex_sleep, codex_rc),
                status_argv=(exe, "-c", 'import sys; print("Not logged in", file=sys.stderr)'),
                logout_argv=(exe, "-c", "pass"),
                parse_status=parse_codex_status,
                url_hosts=frozenset({"auth.openai.com"}), takes_code=False),
        }
        mgr = AuthManager(providers, login_timeout=timeout)
        self.addCleanup(lambda: [mgr.cancel(p) for p in providers])
        return mgr

    def session(self, mgr, provider):
        return mgr._sessions[provider]

    def test_claude_flow_shows_allowed_url_and_pipes_exact_code(self):
        mgr = self.manager()
        mgr.start_login("claude")
        s = self.session(mgr, "claude")
        self.assertTrue(wait_for(lambda: s.state == AWAITING_USER))
        self.assertEqual(s.url, "https://claude.com/cai/oauth/authorize?code=true&state=s1")  # evil.example skipped
        mgr.submit_code("claude", "  " + GOOD_CODE + "\n")
        self.assertTrue(wait_for(lambda: s.state == COMPLETED))
        with open(self.received) as fh:
            self.assertEqual(fh.read(), GOOD_CODE)
        self.assertIsNone(s.error)

    def test_claude_wrong_code_fails_and_never_echoes_the_code(self):
        mgr = self.manager()
        mgr.start_login("claude")
        s = self.session(mgr, "claude")
        self.assertTrue(wait_for(lambda: s.state == AWAITING_USER))
        wrong = "WrongCode0123456789#state-abcdef"
        mgr.submit_code("claude", wrong)
        self.assertTrue(wait_for(lambda: s.state == FAILED))
        self.assertTrue(s.error.startswith("exit 1: "), s.error)
        self.assertIn("[code]", s.error)
        self.assertNotIn(wrong, s.error)

    def test_invalid_code_is_rejected_before_reaching_the_cli(self):
        mgr = self.manager()
        mgr.start_login("claude")
        s = self.session(mgr, "claude")
        self.assertTrue(wait_for(lambda: s.state == AWAITING_USER))
        for bad in ("short", "has space in it 0123456789", "x" * 2000, "semi;colon0123456789abc"):
            with self.assertRaises(InvalidAuthCode):
                mgr.submit_code("claude", bad)
        self.assertEqual(s.state, AWAITING_USER)
        self.assertFalse(os.path.exists(self.received))

    def test_code_without_login_raises(self):
        with self.assertRaises(NoLoginInProgress):
            self.manager().submit_code("claude", GOOD_CODE)

    def test_codex_takes_no_code(self):
        with self.assertRaises(InvalidAuthCode):
            self.manager().submit_code("codex", GOOD_CODE)

    def test_unknown_provider(self):
        mgr = self.manager()
        for call in (lambda: mgr.start_login("gemini"), lambda: mgr.status("gemini"),
                     lambda: mgr.logout("gemini"), lambda: mgr.submit_code("gemini", GOOD_CODE)):
            with self.assertRaises(UnknownProvider):
                call()

    def test_codex_device_code_parsed_through_ansi_and_completes(self):
        mgr = self.manager(codex_sleep="0.3", codex_rc="0")
        snap = mgr.start_login("codex")
        s = self.session(mgr, "codex")
        self.assertTrue(wait_for(lambda: s.state == AWAITING_USER))
        self.assertEqual((s.url, s.user_code), ("https://auth.openai.com/codex/device", "BTI3-CQUN6"))
        self.assertTrue(wait_for(lambda: s.state == COMPLETED))
        self.assertIn(snap["state"], ("starting", AWAITING_USER))

    def test_second_start_returns_running_session_not_a_new_process(self):
        mgr = self.manager()
        mgr.start_login("codex")
        first = self.session(mgr, "codex")
        self.assertTrue(wait_for(lambda: first.state == AWAITING_USER))
        again = mgr.start_login("codex")
        self.assertIs(self.session(mgr, "codex"), first)
        self.assertEqual(again["user_code"], "BTI3-CQUN6")

    def test_restart_after_failure_spawns_new_process(self):
        mgr = self.manager(codex_sleep="0", codex_rc="1")
        mgr.start_login("codex")
        first = self.session(mgr, "codex")
        self.assertTrue(wait_for(lambda: first.state == FAILED))
        mgr.start_login("codex")
        self.assertIsNot(self.session(mgr, "codex"), first)

    def test_login_expires_and_kills_the_cli(self):
        mgr = self.manager(timeout=0.4)
        mgr.start_login("codex")
        s = self.session(mgr, "codex")
        self.assertTrue(wait_for(lambda: s.state == EXPIRED))
        self.assertTrue(wait_for(lambda: s.proc.poll() is not None))
        self.assertIn("start again", s.error)

    def test_cancel_kills_and_forgets(self):
        mgr = self.manager()
        mgr.start_login("codex")
        s = self.session(mgr, "codex")
        mgr.cancel("codex")
        self.assertTrue(wait_for(lambda: s.proc.poll() is not None))
        self.assertNotIn("codex", mgr._sessions)
        self.assertEqual(s.error, "cancelled")

    def test_overview_shape(self):
        mgr = self.manager()
        self.assertEqual(mgr.overview(), {
            "claude": {"logged_in": False, "account": None, "detail": "", "takes_code": True, "login": None},
            "codex": {"logged_in": False, "account": None, "detail": "", "takes_code": False, "login": None},
        })

    def test_status_reads_codex_verdict_from_stderr_of_a_real_process(self):
        exe = sys.executable
        spec = ProviderSpec((exe,), (exe, "-c", 'import sys; print("Logged in using ChatGPT", file=sys.stderr)'), (exe,),
                            parse_codex_status, frozenset(), False)
        st = AuthManager({"codex": spec}).status("codex")
        self.assertEqual((st.logged_in, st.account), (True, SUBSCRIPTION))

    def test_status_started_before_login_end_is_not_cached(self):
        exe = sys.executable
        flag = os.path.join(self.tmp.name, "logged_in")
        script = ("import os, sys, time; was = os.path.exists(sys.argv[1]); time.sleep(0.5);"
                  "print('Logged in using ChatGPT' if was else 'Not logged in', file=sys.stderr)")
        spec = ProviderSpec((exe,), (exe, "-c", script, flag), (exe,), parse_codex_status, frozenset(), False)
        mgr = AuthManager({"codex": spec})
        results = []
        t = threading.Thread(target=lambda: results.append(mgr.status("codex")))
        t.start()
        time.sleep(0.2)              # status check is in flight, saw "signed out"
        open(flag, "w").close()      # login finishes meanwhile
        with mgr._lock:
            mgr._invalidate("codex")
        t.join()
        self.assertFalse(results[0].logged_in)          # the in-flight answer is stale...
        self.assertTrue(mgr.status("codex").logged_in)  # ...and was not cached

    def test_missing_cli_reports_not_installed(self):
        spec = ProviderSpec(("nope-cli-xyz",), ("nope-cli-xyz",), ("nope-cli-xyz",), parse_codex_status,
                            frozenset(), False)
        st = AuthManager({"x": spec}).status("x")
        self.assertEqual((st.logged_in, st.detail), (False, "CLI not installed"))


class StatusParserTests(unittest.TestCase):
    def test_claude_subscription(self):
        st = parse_claude_status(0, '{"loggedIn": true, "authMethod": "claude.ai"}', "")
        self.assertEqual((st.logged_in, st.account), (True, SUBSCRIPTION))

    def test_claude_console_is_flagged_as_api_billing(self):
        st = parse_claude_status(0, '{"loggedIn": true, "authMethod": "console"}', "")
        self.assertEqual((st.logged_in, st.account), (True, API_BILLING))

    def test_claude_logged_out_and_garbage(self):
        self.assertFalse(parse_claude_status(1, '{"loggedIn": false, "authMethod": "none"}', "").logged_in)
        self.assertFalse(parse_claude_status(0, "not json", "").logged_in)
        self.assertFalse(parse_claude_status(0, "[]", "").logged_in)

    def test_codex_texts(self):
        self.assertEqual(parse_codex_status(0, "Logged in using ChatGPT\n", "").account, SUBSCRIPTION)
        self.assertEqual(parse_codex_status(0, "Logged in using an API key - sk-***\n", "").account, API_BILLING)
        self.assertFalse(parse_codex_status(0, "Not logged in\n", "").logged_in)
        self.assertFalse(parse_codex_status(0, "Logged in somehow new\n", "").logged_in)  # unknown -> not in

    def test_codex_verdict_on_stderr_is_read(self):  # what codex 0.142.5 actually does
        self.assertEqual(parse_codex_status(0, "", "Logged in using ChatGPT\n").account, SUBSCRIPTION)
        self.assertFalse(parse_codex_status(0, "", "Not logged in\n").logged_in)

    def test_claude_ignores_stderr_noise(self):
        st = parse_claude_status(0, '{"loggedIn": true, "authMethod": "claude.ai"}', "warning: something\n")
        self.assertEqual(st.account, SUBSCRIPTION)

    def test_url_allowlist_rejects_lookalikes(self):
        hosts = frozenset({"claude.com"})
        self.assertIsNone(auth.allowed_url("https://claude.com.evil.io/x", hosts))
        self.assertIsNone(auth.allowed_url("https://evilclaude.com/x", hosts))
        self.assertIsNone(auth.allowed_url("http://claude.com/x", hosts))
        self.assertEqual(auth.allowed_url("go https://www.claude.com/a?b=1 now", hosts), "https://www.claude.com/a?b=1")


if __name__ == "__main__":
    unittest.main()
