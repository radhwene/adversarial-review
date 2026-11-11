"""gemini_auth.py against a fake ``gemini`` on PATH, run under a real pseudo-terminal."""
from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import gemini_auth  # noqa: E402
from auth import (API_BILLING, AWAITING_USER, COMPLETED, SUBSCRIPTION, AuthManager,  # noqa: E402
                  DEFAULT_PROVIDERS, InvalidAuthCode, parse_gemini_status)

HELPER = os.path.join(os.path.dirname(HERE), "gemini_auth.py")
GOOD_CODE = "4/0AVGzR1Bexample-code_value"
URL = ("https://accounts.google.com/o/oauth2/v2/auth?redirect_uri=https%3A%2F%2Fcodeassist.google.com"
       "%2Fauthcode&response_type=code&client_id=x")

# Mimics gemini 0.66.0: refuses without a TTY, prints the URL, prompts without a newline, then
# saves oauth_creds.json and keeps running (the real CLI goes on to its interactive UI).
FAKE_GEMINI = """\
    #!{python}
    import json, os, sys, time
    home = os.path.join(os.environ["GEMINI_CLI_HOME"], ".gemini")
    open(os.path.join(home, "pid"), "w").write(str(os.getpid()))
    if not sys.stdin.isatty():
        print("Manual authorization is required but the current session is non-interactive"); sys.exit(41)
    if os.environ.get("NO_BROWSER") != "true":
        print("would open a browser"); sys.exit(3)
    if os.environ.get("FAKE_MODE") == "die":
        print("boom"); sys.exit(7)
    sys.stdout.write("\\x1b[?1049h\\x1b[2J\\x1b[HPlease visit the following URL to authorize the application:\\r\\n\\r\\n"
                     + {url!r} + "\\r\\n\\r\\nEnter the authorization code: ")
    sys.stdout.flush()
    code = sys.stdin.readline().strip()
    open(os.path.join(home, "received"), "w").write(code)
    json.dump({{"refresh_token": "r", "access_token": "a"}}, open(os.path.join(home, "oauth_creds.json"), "w"))
    json.dump({{"active": "me@example.com", "old": []}}, open(os.path.join(home, "google_accounts.json"), "w"))
    time.sleep(60)
"""


class GeminiHelperTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = self.tmp.name
        self.dir = os.path.join(self.home, ".gemini")
        os.makedirs(self.dir)
        bindir = os.path.join(self.home, "bin")
        os.makedirs(bindir)
        fake = os.path.join(bindir, "gemini")
        with open(fake, "w") as fh:
            fh.write(textwrap.dedent(FAKE_GEMINI.format(python=sys.executable, url=URL)))
        os.chmod(fake, stat.S_IRWXU)
        self.env = {**os.environ, "GEMINI_CLI_HOME": self.home, "PATH": bindir + os.pathsep + os.environ["PATH"]}
        for name in gemini_auth.API_KEY_ENVS:
            self.env.pop(name, None)

    def write(self, name, data):
        with open(os.path.join(self.dir, name), "w") as fh:
            json.dump(data, fh)

    def read(self, name):
        with open(os.path.join(self.dir, name)) as fh:
            return fh.read()

    def spawn_login(self, **extra_env):
        proc = subprocess.Popen([sys.executable, HELPER, "login"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, env={**self.env, **extra_env})
        def cleanup():
            if proc.poll() is None:
                proc.kill(); proc.wait()
            for fh in (proc.stdin, proc.stdout):
                if not fh.closed:
                    fh.close()
        self.addCleanup(cleanup)
        return proc

    # -- status / logout / settings ------------------------------------------------
    def test_status_logged_out_without_creds(self):
        self.assertEqual(gemini_auth.status(self.dir), {"logged_in": False, "account": None})

    def test_status_needs_a_refresh_token(self):
        self.write("oauth_creds.json", {"access_token": "a"})
        self.assertEqual(gemini_auth.status(self.dir), {"logged_in": False, "account": None})

    def test_status_reports_active_account(self):
        self.write("oauth_creds.json", {"refresh_token": "r"})
        self.write("google_accounts.json", {"active": "me@example.com"})
        self.assertEqual(gemini_auth.status(self.dir), {"logged_in": True, "account": "me@example.com"})

    def test_status_flags_api_key_env(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "k"}):
            self.assertEqual(gemini_auth.status(self.dir), {"logged_in": True, "account": "api_key"})

    def test_logout_removes_credentials_only(self):
        self.write("oauth_creds.json", {"refresh_token": "r"})
        self.write("google_accounts.json", {"active": "x"})
        self.write("settings.json", {"ui": {"theme": "x"}})
        gemini_auth.logout(self.dir)
        self.assertEqual(sorted(os.listdir(self.dir)), ["settings.json"])
        gemini_auth.logout(self.dir)  # idempotent

    def test_ensure_settings_merges_and_is_private(self):
        self.write("settings.json", {"ui": {"theme": "dark"}, "security": {"folderTrust": {"enabled": False}}})
        gemini_auth.ensure_settings(self.dir)
        self.assertEqual(json.loads(self.read("settings.json")), {
            "ui": {"theme": "dark"},
            "security": {"folderTrust": {"enabled": False}, "auth": {"selectedType": "oauth-personal"}},
            "privacy": {"usageStatisticsEnabled": False}})
        self.assertEqual(stat.S_IMODE(os.stat(os.path.join(self.dir, "settings.json")).st_mode), 0o600)

    def test_status_command_prints_json(self):
        out = subprocess.run([sys.executable, HELPER, "status"], capture_output=True, text=True, env=self.env)
        self.assertEqual((out.returncode, json.loads(out.stdout)), (0, {"logged_in": False, "account": None}))

    def test_unknown_command_is_usage_error(self):
        out = subprocess.run([sys.executable, HELPER, "login", "--api-key"], capture_output=True, text=True, env=self.env)
        self.assertEqual(out.returncode, 2)

    # -- login under a pty ------------------------------------------------------
    def test_login_relays_url_pipes_exact_code_and_stops_the_cli(self):
        proc = self.spawn_login()
        lines = []
        while not any("authorization code" in l for l in lines):
            line = proc.stdout.readline()
            self.assertTrue(line, f"helper ended early: {lines}")
            lines.append(line.strip())
        self.assertIn(URL, lines)
        self.assertFalse(any("\x1b" in l for l in lines), lines)
        proc.stdin.write(GOOD_CODE + "\n"); proc.stdin.flush()
        self.assertEqual(proc.wait(timeout=20), 0)
        self.assertEqual(self.read("received"), GOOD_CODE)
        self.assertEqual(json.loads(self.read("settings.json"))["security"]["auth"]["selectedType"], "oauth-personal")
        fake_pid = int(self.read("pid"))
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                os.kill(fake_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            self.fail("fake gemini still running after sign-in")

    def test_login_fails_when_cli_exits_first(self):
        proc = self.spawn_login(FAKE_MODE="die")
        out = proc.stdout.read()  # stdin stays open: only the CLI's exit can end the helper
        self.assertEqual(proc.wait(timeout=20), 1)
        proc.stdin.close(); proc.stdout.close()
        self.assertIn("gemini exited before sign-in completed", out)

    def test_auth_manager_drives_the_real_gemini_spec(self):
        with mock.patch.dict(os.environ, {"GEMINI_CLI_HOME": self.home, "PATH": self.env["PATH"]}):
            os.environ.pop("GEMINI_API_KEY", None); os.environ.pop("GOOGLE_API_KEY", None)
            mgr = AuthManager({"gemini": DEFAULT_PROVIDERS["gemini"]})
            self.addCleanup(mgr.cancel, "gemini")
            self.assertFalse(mgr.status("gemini").logged_in)
            mgr.start_login("gemini")
            deadline = time.monotonic() + 20
            while mgr._sessions["gemini"].state != AWAITING_USER and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertEqual(mgr._sessions["gemini"].url, URL)
            with self.assertRaises(InvalidAuthCode):
                mgr.submit_code("gemini", "short")
            mgr.submit_code("gemini", GOOD_CODE)
            while mgr._sessions["gemini"].state != COMPLETED and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertEqual(mgr._sessions["gemini"].state, COMPLETED)
            st = mgr.status("gemini")
            self.assertEqual((st.logged_in, st.account), (True, SUBSCRIPTION))
            mgr.logout("gemini")
            self.assertFalse(mgr.status("gemini").logged_in)


class ParseGeminiStatusTests(unittest.TestCase):
    def test_parse(self):
        cases = [
            ((0, '{"logged_in": true, "account": "me@example.com"}'), (True, SUBSCRIPTION)),
            ((0, '{"logged_in": true, "account": "api_key"}'), (True, API_BILLING)),
            ((0, '{"logged_in": false, "account": null}'), (False, None)),
            ((0, '{"logged_in": "yes"}'), (False, None)),
            ((1, '{"logged_in": true}'), (False, None)),
            ((0, "Traceback"), (False, None)),
        ]
        for (rc, out), want in cases:
            with self.subTest(out=out):
                st = parse_gemini_status(rc, out, "")
                self.assertEqual((st.logged_in, st.account), want)


if __name__ == "__main__":
    unittest.main()
