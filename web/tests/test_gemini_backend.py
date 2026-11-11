"""GeminiBackend against a fake ``gemini`` on PATH (real subprocess, real stdin/argv)."""
from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
import textwrap
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from backends import (GEMINI_ISOLATION_FLAGS, BackendError, GeminiBackend, TurnRequest,  # noqa: E402
                      build_registry, gemini_error_message, parse_gemini_json)

# Records argv, stdin and cwd, then prints what FAKE_REPLY / FAKE_RC say.
FAKE_GEMINI = """\
    #!{python}
    import json, os, sys
    record = {{"argv": sys.argv[1:], "stdin": sys.stdin.read(), "cwd": os.getcwd()}}
    json.dump(record, open(os.environ["FAKE_RECORD"], "w"))
    sys.stdout.write(os.environ.get("FAKE_REPLY", ""))
    sys.stderr.write(os.environ.get("FAKE_STDERR", ""))
    sys.exit(int(os.environ.get("FAKE_RC", "0")))
"""


class GeminiBackendTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        fake = os.path.join(tmp.name, "gemini")
        with open(fake, "w") as fh:
            fh.write(textwrap.dedent(FAKE_GEMINI.format(python=sys.executable)))
        os.chmod(fake, stat.S_IRWXU)
        self.record = os.path.join(tmp.name, "record.json")
        self.saved = dict(os.environ)
        self.addCleanup(lambda: (os.environ.clear(), os.environ.update(self.saved)))
        os.environ.update(PATH=tmp.name + os.pathsep + os.environ["PATH"], FAKE_RECORD=self.record)

    def run_turn(self, reply, rc=0, stderr="", model=None, backend=None):
        os.environ.update(FAKE_REPLY=reply, FAKE_RC=str(rc), FAKE_STDERR=stderr)
        backend = backend or GeminiBackend()
        return backend.complete(TurnRequest(role="critic", round=1, prompt="Débat: ∑ prompt", model=model))

    def recorded(self):
        with open(self.record) as fh:
            return json.load(fh)

    def test_argv_stdin_and_isolated_cwd(self):
        text = self.run_turn(json.dumps({"session_id": "s", "response": "  Réponse.\n", "stats": {}}), model="flash")
        self.assertEqual(text, "Réponse.")
        rec = self.recorded()
        self.assertEqual(rec["argv"], [*GEMINI_ISOLATION_FLAGS, "-o", "json", "-m", "flash", "-p", ""])
        self.assertEqual(rec["stdin"], "Débat: ∑ prompt")
        self.assertTrue(os.path.basename(rec["cwd"]).startswith("advr-gemini-"))
        self.assertFalse(os.path.exists(rec["cwd"]))  # scratch removed after the turn

    def test_default_argv_has_no_model(self):
        self.assertEqual(GeminiBackend().build_argv(), ["gemini", *GEMINI_ISOLATION_FLAGS, "-o", "json", "-p", ""])
        self.assertEqual(GeminiBackend(model="pro").build_argv()[-4:], ["-m", "pro", "-p", ""])
        self.assertEqual(GeminiBackend(model="pro").build_argv("flash")[-4:], ["-m", "flash", "-p", ""])

    def test_json_error_is_reported(self):
        reply = json.dumps({"error": {"type": "Error", "message": "Quota exceeded", "code": 429}})
        with self.assertRaisesRegex(BackendError, r"^gemini exited 1: Quota exceeded$"):
            self.run_turn(reply, rc=1, stderr="noise")

    def test_error_document_with_exit_zero_still_fails(self):
        with self.assertRaisesRegex(BackendError, "gemini exited 0: bad"):
            self.run_turn(json.dumps({"response": "x", "error": {"message": "bad"}}))

    def test_non_json_failure_uses_stderr(self):
        with self.assertRaisesRegex(BackendError, r"^gemini exited 41: Please set an Auth method$"):
            self.run_turn("", rc=41, stderr="Please set an Auth method")

    def test_non_json_success_is_an_error(self):
        with self.assertRaisesRegex(BackendError, "non-JSON output: plain text"):
            self.run_turn("plain text")

    def test_empty_response(self):
        for reply in (json.dumps({"response": "   "}), json.dumps({"response": None}), json.dumps({})):
            with self.subTest(reply=reply), self.assertRaisesRegex(BackendError, "^gemini returned empty output$"):
                self.run_turn(reply)

    def test_missing_binary(self):
        os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + "/nonexistent"
        self.assertFalse(GeminiBackend().available())
        with self.assertRaisesRegex(BackendError, "gemini not found on PATH"):
            GeminiBackend().complete(TurnRequest(role="judge", round=0, prompt="p"))

    def test_registry_and_allowlist(self):
        reg = build_registry(gemini_model="pro")
        self.assertIsInstance(reg["gemini"], GeminiBackend)
        self.assertEqual(reg["gemini"].model, "pro")
        self.assertEqual(reg["gemini"].models, ("auto", "pro", "flash", "flash-lite"))
        self.assertEqual(reg["gemini"].efforts, ())


class ParseTests(unittest.TestCase):
    def test_parse_and_error(self):
        self.assertIsNone(parse_gemini_json("[1]"))
        self.assertIsNone(parse_gemini_json("nope"))
        self.assertEqual(parse_gemini_json('{"response": "a"}'), {"response": "a"})
        self.assertIsNone(gemini_error_message(None))
        self.assertIsNone(gemini_error_message({"response": "a"}))
        self.assertEqual(gemini_error_message({"error": "flat"}), "flat")
        self.assertEqual(gemini_error_message({"error": {"message": "m", "code": 1}}), "m")


if __name__ == "__main__":
    unittest.main()
