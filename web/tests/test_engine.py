from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from typing import List, Optional
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backends import (Backend, BackendError, BackendTimeout, ClaudeBackend, CodexBackend,  # noqa: E402
                      MockBackend, TurnRequest)
from engine import (Debate, DebateConfig, DebateRunner, InvalidDebateConfig,  # noqa: E402
                    parse_status_block, strip_status_block)


def proposer_reply(text: str, conf: int = 70) -> str:
    return (f"{text}\n\n---PROPOSER_STATUS---\nCONFIDENCE: {conf}\nCONCEDED: 0\nPOSITION_CHANGED: false\n"
            "---END_PROPOSER_STATUS---")


def critic_reply(text: str, verdict: str, objections: int) -> str:
    return f"{text}\n\n---CRITIC_STATUS---\nOBJECTIONS: {objections}\nVERDICT: {verdict}\n---END_CRITIC_STATUS---"


JUDGE_REPLY = "## Verdict\nOK\n\n---JUDGE_STATUS---\nCONFIDENCE: 60\nWINNER: critic\nUNRESOLVED: 1\n---END_JUDGE_STATUS---"


class ScriptedBackend(Backend):
    """Returns scripted replies in order and records every request; fails if over-called."""

    name = "scripted"

    def __init__(self, replies: List[str], error: Optional[Exception] = None) -> None:
        self.replies = list(replies)
        self.error = error
        self.requests: List[TurnRequest] = []

    def available(self) -> bool:
        return True

    def complete(self, request: TurnRequest) -> str:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        if not self.replies:
            raise AssertionError(f"unexpected extra call: {request.role} round {request.round}")
        return self.replies.pop(0)


def run_debate(backend: Backend, rounds: int = 3, runs_dir: Optional[str] = None) -> Debate:
    debate = Debate(config=DebateConfig(question="Is P true?", proposer="s", critic="s", judge="s", max_rounds=rounds))
    DebateRunner({"s": backend}, runs_dir).run(debate)
    return debate


class StatusBlockTests(unittest.TestCase):
    def test_parses_last_block(self):
        text = "a\n---CRITIC_STATUS---\nVERDICT: CONTEST\n---END_CRITIC_STATUS---\nb\n" \
               "---CRITIC_STATUS---\nOBJECTIONS: 0\nVERDICT: ACCEPT\n---END_CRITIC_STATUS---"
        self.assertEqual(parse_status_block(text, "CRITIC_STATUS"), {"OBJECTIONS": "0", "VERDICT": "ACCEPT"})

    def test_missing_block_returns_none(self):
        self.assertIsNone(parse_status_block("no block here", "CRITIC_STATUS"))

    def test_other_block_name_not_matched(self):
        self.assertIsNone(parse_status_block(proposer_reply("x"), "CRITIC_STATUS"))

    def test_strip_removes_block_only(self):
        self.assertEqual(strip_status_block(proposer_reply("Body text"), "PROPOSER_STATUS"), "Body text")


class EngineFlowTests(unittest.TestCase):
    def test_full_rounds_without_convergence(self):
        backend = ScriptedBackend([
            proposer_reply("OPENING-1"),
            critic_reply("CRIT-1", "CONTEST", 2), proposer_reply("REBUT-1"),
            critic_reply("CRIT-2", "CONTEST", 1), proposer_reply("REBUT-2"),
            JUDGE_REPLY,
        ])
        debate = run_debate(backend, rounds=2)

        self.assertEqual(debate.state, "completed")
        self.assertEqual([(r.role, r.round) for r in backend.requests], [
            ("proposer", 0), ("critic", 1), ("proposer", 1), ("critic", 2), ("proposer", 2), ("judge", 3)])
        self.assertEqual([e["type"] for e in debate.events],
                         ["debate_started"] + ["turn_started", "turn_completed"] * 6 + ["debate_completed"])
        self.assertEqual([e["seq"] for e in debate.events], list(range(len(debate.events))))
        self.assertEqual(debate.events[-1]["converged_round"], None)
        self.assertEqual(debate.events[-1]["verdict"], {"CONFIDENCE": "60", "WINNER": "critic", "UNRESOLVED": "1"})

    def test_prompts_carry_question_and_prior_turns(self):
        backend = ScriptedBackend([
            proposer_reply("OPENING-1"), critic_reply("CRIT-1", "CONTEST", 2), proposer_reply("REBUT-1"),
            critic_reply("CRIT-2", "ACCEPT", 0), JUDGE_REPLY,
        ])
        run_debate(backend, rounds=3)
        prompts = [r.prompt for r in backend.requests]
        self.assertTrue(all("Is P true?" in p for p in prompts))
        self.assertIn("### Opening: PROPOSER (s)\nOPENING-1", prompts[1])
        self.assertIn("round 1 of 3", prompts[1])
        self.assertIn("### Round 1: CRITIC (s)\nCRIT-1", prompts[2])
        self.assertIn("REBUT-1", prompts[3])
        # status blocks are stripped from the transcript fed to the next agent
        self.assertNotIn("CRITIC_STATUS---\nOBJECTIONS: 2", prompts[2])
        self.assertIn("CRIT-2", prompts[4])
        self.assertIn("JUDGE", prompts[4])

    def test_critic_accept_stops_early(self):
        backend = ScriptedBackend([proposer_reply("OPEN"), critic_reply("fine", "ACCEPT", 0), JUDGE_REPLY])
        debate = run_debate(backend, rounds=5)
        self.assertEqual([r.role for r in backend.requests], ["proposer", "critic", "judge"])
        self.assertIn({"type": "converged", "round": 1}, [{k: e[k] for k in ("type", "round")}
                                                          for e in debate.events if e["type"] == "converged"])
        self.assertEqual(debate.events[-1]["converged_round"], 1)

    def test_missing_status_block_is_treated_as_contest(self):
        backend = ScriptedBackend([proposer_reply("OPEN"), "critique without block", proposer_reply("R1"), JUDGE_REPLY])
        debate = run_debate(backend, rounds=1)
        critic_turn = debate.turns[1]
        self.assertIsNone(critic_turn.status)
        self.assertEqual(critic_turn.content, "critique without block")
        self.assertEqual([r.role for r in backend.requests], ["proposer", "critic", "proposer", "judge"])

    def test_backend_error_fails_debate(self):
        debate = run_debate(ScriptedBackend([], error=BackendError("boom")))
        self.assertEqual(debate.state, "failed")
        self.assertEqual(debate.events[-1]["type"], "debate_failed")
        self.assertEqual(debate.events[-1]["kind"], "backend_error")
        self.assertEqual(debate.events[-1]["message"], "boom")

    def test_backend_timeout_fails_with_timeout_kind(self):
        debate = run_debate(ScriptedBackend([], error=BackendTimeout("too slow")))
        self.assertEqual(debate.events[-1]["kind"], "timeout")

    def test_transcript_persisted(self):
        with tempfile.TemporaryDirectory() as d:
            debate = run_debate(MockBackend(), rounds=2, runs_dir=d)
            with open(os.path.join(d, f"{debate.id}.json")) as fh:
                saved = json.load(fh)
        self.assertEqual(saved["state"], "completed")
        self.assertEqual(len(saved["turns"]), len(debate.turns))

    def test_mock_backend_converges_in_round_2(self):
        debate = run_debate(MockBackend(), rounds=4)
        self.assertEqual([(t.role, t.round) for t in debate.turns],
                         [("proposer", 0), ("critic", 1), ("proposer", 1), ("critic", 2), ("judge", 5)])


class ConfigValidationTests(unittest.TestCase):
    registry = {"mock": MockBackend()}

    def cfg(self, **kw):
        base = dict(question="q", proposer="mock", critic="mock", judge="mock", max_rounds=3)
        base.update(kw)
        return DebateConfig(**base)

    def test_valid(self):
        self.cfg().validate(self.registry)

    def test_rejects(self):
        for kw, msg in [({"question": "  "}, "empty"), ({"question": "x" * 4001}, "exceeds"),
                        ({"max_rounds": 0}, "max_rounds"), ({"max_rounds": 6}, "max_rounds"),
                        ({"critic": "gpt"}, "unknown backend")]:
            with self.subTest(kw=kw), self.assertRaisesRegex(InvalidDebateConfig, msg):
                self.cfg(**kw).validate(self.registry)

    def test_rejects_unavailable_backend(self):
        missing = mock.Mock(spec=Backend)
        missing.available.return_value = False
        with self.assertRaisesRegex(InvalidDebateConfig, "not installed"):
            self.cfg(judge="x").validate({"mock": MockBackend(), "x": missing})
        missing.available.assert_called_once_with()


class CliBackendTests(unittest.TestCase):
    def test_claude_argv_and_stdin(self):
        done = mock.Mock(returncode=0, stdout=" answer \n", stderr="")
        with mock.patch("backends.subprocess.run", return_value=done) as run:
            out = ClaudeBackend(model="opus", timeout=42).complete(TurnRequest("critic", 1, "PROMPT"))
        self.assertEqual(out, "answer")
        args, kwargs = run.call_args
        self.assertEqual(args[0], ["claude", "--print", "--setting-sources", "project", "--tools", "",
                                   "--strict-mcp-config", "--model", "opus"])
        self.assertEqual(kwargs["input"], "PROMPT")
        self.assertEqual(kwargs["timeout"], 42)
        self.assertNotIn("shell", kwargs)

    def test_claude_nonzero_exit_raises(self):
        done = mock.Mock(returncode=2, stdout="", stderr="auth failed")
        with mock.patch("backends.subprocess.run", return_value=done), \
                self.assertRaisesRegex(BackendError, "claude exited 2: auth failed"):
            ClaudeBackend().complete(TurnRequest("critic", 1, "p"))

    def test_timeout_maps_to_backend_timeout(self):
        import subprocess
        with mock.patch("backends.subprocess.run", side_effect=subprocess.TimeoutExpired("claude", 5)), \
                self.assertRaises(BackendTimeout):
            ClaudeBackend(timeout=5).complete(TurnRequest("critic", 1, "p"))

    def test_codex_reads_last_message_file(self):
        def fake_run(argv, **kwargs):
            out_file = argv[argv.index("-o") + 1]
            with open(out_file, "w") as fh:
                fh.write("final answer\n")
            return mock.Mock(returncode=0, stdout="noisy progress", stderr="")

        with mock.patch("backends.subprocess.run", side_effect=fake_run) as run:
            out = CodexBackend(model="m1").complete(TurnRequest("proposer", 0, "P"))
        self.assertEqual(out, "final answer")
        argv = run.call_args[0][0]
        self.assertEqual(argv[:5], ["codex", "exec", "--json", "-m", "m1"])
        self.assertEqual(argv[argv.index("--sandbox") + 1], "read-only")
        self.assertEqual(argv[-1], "-")

    def test_codex_failure_surfaces_typed_jsonl_error(self):
        api_err = json.dumps({"type": "error", "error": {"message": "model not supported", "code": "model_not_found"}})
        stdout = "\n".join([
            json.dumps({"type": "thread.started"}),
            "not json",
            json.dumps({"type": "turn.failed", "error": {"message": api_err}}),
        ])
        done = mock.Mock(returncode=1, stdout=stdout, stderr="echoed prompt noise")
        with mock.patch("backends.subprocess.run", return_value=done), \
                self.assertRaisesRegex(BackendError, r"^codex exited 1: model not supported$"):
            CodexBackend().complete(TurnRequest("critic", 1, "p"))

    def test_codex_failure_without_events_falls_back_to_stderr(self):
        done = mock.Mock(returncode=1, stdout="", stderr="boom")
        with mock.patch("backends.subprocess.run", return_value=done), \
                self.assertRaisesRegex(BackendError, r"^codex exited 1: boom$"):
            CodexBackend().complete(TurnRequest("critic", 1, "p"))


if __name__ == "__main__":
    unittest.main()
