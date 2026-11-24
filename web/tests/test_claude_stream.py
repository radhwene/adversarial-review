"""Streaming replies: the stream-json parser, ClaudeBackend.stream against a fake ``claude``
executable (real subprocess and pipes), and the chat draft that SSE readers see."""
from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
import textwrap
import unittest
from typing import List
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backends import (CLAUDE_STREAM_FLAGS, BackendError, BackendTimeout, ClaudeBackend, MockBackend,  # noqa: E402
                      TurnRequest, parse_claude_stream)
from chat import Chat, ChatConfig, ChatRunner  # noqa: E402
from test_engine import ScriptedBackend  # noqa: E402


def delta(text: str) -> str:
    return json.dumps({"type": "stream_event", "event": {"type": "content_block_delta", "index": 0,
                                                          "delta": {"type": "text_delta", "text": text}}})


def result(text: str, is_error: bool = False, subtype: str = "success") -> str:
    return json.dumps({"type": "result", "subtype": subtype, "is_error": is_error, "result": text})


THINKING = json.dumps({"type": "stream_event", "event": {"type": "content_block_delta",
                                                         "delta": {"type": "thinking_delta", "thinking": "hmm"}}})
RATE = json.dumps({"type": "rate_limit_event", "rate_limit_info": {"status": "allowed"}})


class ParseTests(unittest.TestCase):
    def test_text_deltas_forwarded_in_order_others_skipped(self):
        got: List[str] = []
        out = parse_claude_stream(["not json", THINKING, delta("Bon"), "[1, 2]", RATE, delta("jour"),
                                   result("Bonjour")], got.append)
        self.assertEqual(got, ["Bon", "jour"])
        self.assertEqual((out.streamed, out.result, out.is_error), ("Bonjour", "Bonjour", False))

    def test_error_result(self):
        out = parse_claude_stream([result("Prompt is too long", is_error=True)], [].append)
        self.assertEqual((out.result, out.is_error), ("Prompt is too long", True))
        out = parse_claude_stream([result("", subtype="error_max_turns")], [].append)
        self.assertTrue(out.is_error)

    def test_no_result_event(self):
        out = parse_claude_stream([delta("cut")], [].append)
        self.assertEqual((out.streamed, out.result, out.is_error), ("cut", None, False))


FAKE_CLAUDE = textwrap.dedent("""\
    #!/usr/bin/env python3
    import json, os, sys, time
    log = os.environ["FAKE_LOG"]
    prompt = sys.stdin.read()
    json.dump({"argv": sys.argv[1:], "prompt": prompt, "cwd": os.getcwd(), "pid": os.getpid()}, open(log, "w"))
    mode = os.environ["FAKE_MODE"]
    def say(obj):
        print(json.dumps(obj), flush=True)
    def piece(t):
        say({"type": "stream_event", "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": t}}})
    if mode == "hang":
        piece("Je ")
        time.sleep(30)  # a stuck CLI: the backend must kill it
    if mode == "crash":
        sys.stderr.write("boom on stderr")
        sys.exit(3)
    piece("Bon"); piece("jour")
    if mode == "ok":
        say({"type": "result", "subtype": "success", "is_error": False, "result": " Bonjour \\n"})
    elif mode == "error":
        say({"type": "result", "subtype": "success", "is_error": True, "result": "Credit balance is too low"})
    elif mode == "empty":
        say({"type": "result", "subtype": "success", "is_error": False, "result": "   "})
    """)


class ClaudeStreamTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        path = os.path.join(self.tmp.name, "claude")
        with open(path, "w") as fh:
            fh.write(FAKE_CLAUDE)
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
        self.log = os.path.join(self.tmp.name, "call.json")
        self.env = mock.patch.dict(os.environ, {"PATH": self.tmp.name + os.pathsep + os.environ["PATH"],
                                                "FAKE_LOG": self.log})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def run_mode(self, mode: str, timeout: float = 10, on_text=None):
        os.environ["FAKE_MODE"] = mode
        got: List[str] = []
        req = TurnRequest(role="chat", round=1, prompt="### USER\nSalut é", model="haiku", effort="low")
        text = ClaudeBackend(timeout=timeout).stream(req, on_text or got.append)
        return text, got

    def call(self):
        with open(self.log) as fh:
            return json.load(fh)

    def test_streams_pieces_and_returns_the_result_text(self):
        text, got = self.run_mode("ok")
        self.assertEqual((text, got), ("Bonjour", ["Bon", "jour"]))
        call = self.call()
        self.assertEqual(call["argv"], ["--print", "--setting-sources", "project", "--tools", "", "--strict-mcp-config",
                                        "--model", "haiku", "--effort", "low", *CLAUDE_STREAM_FLAGS])
        self.assertEqual(call["prompt"], "### USER\nSalut é")
        self.assertTrue(os.path.basename(call["cwd"]).startswith("advr-claude-"))
        self.assertFalse(os.path.exists(call["cwd"]), "scratch dir must be removed")

    def test_error_result_is_a_backend_error_with_its_message(self):
        with self.assertRaises(BackendError) as ctx:
            self.run_mode("error")
        self.assertNotIsInstance(ctx.exception, BackendTimeout)
        self.assertEqual(str(ctx.exception), "claude exited 0: Credit balance is too low")

    def test_crash_reports_stderr(self):
        with self.assertRaises(BackendError) as ctx:
            self.run_mode("crash")
        self.assertEqual(str(ctx.exception), "claude exited 3: boom on stderr")

    def test_empty_result(self):
        with self.assertRaisesRegex(BackendError, "^claude returned empty output$"):
            self.run_mode("empty")

    def test_hung_cli_is_killed_and_reported_as_timeout(self):
        got: List[str] = []
        with self.assertRaisesRegex(BackendTimeout, "timed out after 1s"):
            self.run_mode("hang", timeout=1, on_text=got.append)
        self.assertEqual(got, ["Je "])
        with self.assertRaises(ProcessLookupError):
            os.kill(self.call()["pid"], 0)

    def test_sink_failure_kills_the_cli(self):
        def explode(_text):
            raise RuntimeError("sink broke")
        with self.assertRaisesRegex(RuntimeError, "sink broke"):
            self.run_mode("hang", on_text=explode)
        with self.assertRaises(ProcessLookupError):
            os.kill(self.call()["pid"], 0)

    def test_missing_binary(self):
        os.environ["PATH"] = self.tmp.name + "-nowhere"
        with self.assertRaisesRegex(BackendError, "^claude not found on PATH$"):
            self.run_mode("ok")


class StepBackend(ScriptedBackend):
    """Streams its reply in two pieces and runs ``check`` between them (the reply is in flight)."""

    def __init__(self, reply: str, check) -> None:
        super().__init__([reply])
        self.check = check

    def stream(self, request, on_text):
        reply = self.complete(request)
        on_text(reply[:3])
        self.check()
        on_text(reply[3:])
        return reply


class ChatDraftTests(unittest.TestCase):
    def new_chat(self) -> Chat:
        return Chat(config=ChatConfig(agent="s"))

    def test_draft_grows_during_the_reply_and_is_never_saved(self):
        chat = self.new_chat()
        seen = {}

        def check():
            seen["draft"] = chat.draft
            seen["snapshot"] = json.dumps(chat.snapshot())
            seen["update"] = chat.wait_update(start=len(chat.events), draft_round=-1, draft_sent=0, timeout=1)
            seen["resume"] = chat.wait_update(start=len(chat.events), draft_round=1, draft_sent=2, timeout=1)
        chat.add_user_message("salut")
        ChatRunner({"s": StepBackend("Bonjour", check)}).reply(chat)
        self.assertEqual(seen["draft"], "Bon")
        self.assertNotIn("Bon", seen["snapshot"])
        self.assertEqual(seen["update"], ([], {"round": 1, "offset": 0, "text": "Bon"}))
        self.assertEqual(seen["resume"], ([], {"round": 1, "offset": 2, "text": "n"}))
        self.assertEqual((chat.draft, chat.turns[-1].content, chat.state), ("", "Bonjour", "idle"))

    def test_reader_with_the_whole_draft_waits(self):
        chat = self.new_chat()
        result = {}

        def check():
            result["got"] = chat.wait_update(start=len(chat.events), draft_round=1, draft_sent=3, timeout=0.2)
        chat.add_user_message("salut")
        ChatRunner({"s": StepBackend("Bonjour", check)}).reply(chat)
        self.assertEqual(result["got"], ([], None))

    def test_draft_of_an_older_reply_is_resent_from_zero(self):
        chat = self.new_chat()
        result = {}
        chat.add_user_message("one")
        ChatRunner({"s": ScriptedBackend(["first"])}).reply(chat)

        def check():
            result["got"] = chat.wait_update(start=len(chat.events), draft_round=1, draft_sent=5, timeout=1)
        chat.add_user_message("two")
        ChatRunner({"s": StepBackend("second", check)}).reply(chat)
        self.assertEqual(result["got"], ([], {"round": 3, "offset": 0, "text": "sec"}))

    def test_failure_clears_the_draft_and_late_pieces_are_ignored(self):
        chat = self.new_chat()
        chat.add_user_message("salut")
        chat.append_draft("Bon")
        chat.fail_reply("backend_error", "boom")
        self.assertEqual(chat.draft, "")
        chat.append_draft("jour")  # idle: ignored
        self.assertEqual(chat.draft, "")
        self.assertEqual(chat.wait_update(len(chat.events), -1, 0, timeout=0.1), ([], None))

    def test_non_streaming_backend_still_answers(self):
        chat = self.new_chat()
        chat.add_user_message("salut")
        ChatRunner({"s": ScriptedBackend(["d'un bloc"])}).reply(chat)
        self.assertEqual((chat.turns[-1].content, chat.draft), ("d'un bloc", ""))


class MockStreamTests(unittest.TestCase):
    def test_words_rebuild_the_reply(self):
        got: List[str] = []
        req = TurnRequest(role="chat", round=1, prompt="x" * 10)
        reply = MockBackend().stream(req, got.append)
        self.assertGreater(len(got), 3)
        self.assertEqual("".join(got), reply)
        self.assertEqual(reply, MockBackend().complete(req))


if __name__ == "__main__":
    unittest.main()
