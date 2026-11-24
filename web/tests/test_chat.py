from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backends import BackendError, BackendTimeout, ClaudeBackend, MockBackend  # noqa: E402
from chat import (CHAT_ROLE, MAX_CHAT_HISTORY_CHARS, MAX_CHAT_MESSAGES, Chat, ChatBusy, ChatConfig,  # noqa: E402
                  ChatRunner, InvalidChatConfig, format_history)
from engine import MAX_USER_MESSAGE_CHARS, InvalidUserMessage, TooManyUserMessages, Turn  # noqa: E402
from test_engine import HookedBackend, ScriptedBackend  # noqa: E402


def new_chat(**cfg) -> Chat:
    return Chat(config=ChatConfig(agent="s", **cfg))


def say(chat: Chat, backend, text: str) -> None:
    chat.add_user_message(text)
    ChatRunner({"s": backend}).reply(chat)


def turn(role: str, content: str) -> Turn:
    return Turn(role=role, agent=role, round=0, content=content, status=None, duration_s=0.0)


class ChatMemoryTests(unittest.TestCase):
    def test_second_reply_sees_the_first_exchange(self):
        backend = ScriptedBackend(["Noté : 42.", "Le nombre était 42."])
        chat = new_chat(model="haiku", effort="low")
        say(chat, backend, "Retiens le nombre 42.")
        say(chat, backend, "Quel nombre ?")
        first, second = backend.requests
        self.assertEqual((first.role, first.round, first.model, first.effort), (CHAT_ROLE, 1, "haiku", "low"))
        self.assertEqual(second.round, 3)
        self.assertIn("### USER\nRetiens le nombre 42.\n\n### ASSISTANT\nNoté : 42.\n\n### USER\nQuel nombre ?",
                      second.prompt)
        self.assertNotIn("Quel nombre", first.prompt)
        self.assertEqual([(t.role, t.round) for t in chat.turns],
                         [("user", 0), ("assistant", 1), ("user", 2), ("assistant", 3)])
        self.assertEqual(chat.state, "idle")
        self.assertEqual([e["type"] for e in chat.events],
                         ["chat_started"] + ["turn_completed"] * 4)
        self.assertEqual(chat.title, "Retiens le nombre 42.")

    def test_long_title_is_cut_with_an_ellipsis(self):
        from chat import TITLE_CHARS
        chat = new_chat()
        chat.add_user_message("mot " * 60)
        self.assertEqual(len(chat.title), TITLE_CHARS)
        self.assertTrue(chat.title.endswith("mot…"), chat.title)
        exact = new_chat()
        exact.add_user_message("x" * TITLE_CHARS)
        self.assertEqual(exact.title, "x" * TITLE_CHARS)

    def test_reply_is_stripped_and_attributed_to_agent(self):
        chat = new_chat()
        say(chat, ScriptedBackend(["  hello \n"]), "hi")
        self.assertEqual((chat.turns[-1].agent, chat.turns[-1].content), ("s", "hello"))


class HistoryWindowTests(unittest.TestCase):
    def test_everything_fits(self):
        turns = [turn("user", "a"), turn("assistant", "b"), turn("user", "c")]
        self.assertEqual(format_history(turns), "### USER\na\n\n### ASSISTANT\nb\n\n### USER\nc")

    def test_oldest_dropped_whole_with_marker(self):
        block = len("### USER\n") + 10  # each block is exactly this long
        turns = [turn("user", str(i) * 10) for i in range(5)]
        # room for exactly 2 blocks (+ the 2-char separator after the first counted one)
        out = format_history(turns, limit=2 * block + 2)
        self.assertEqual(out, "(3 earlier messages omitted: the conversation is longer than the memory window)"
                              "\n\n### USER\n3333333333\n\n### USER\n4444444444")
        out = format_history(turns, limit=2 * block + 1)
        self.assertTrue(out.startswith("(4 earlier messages omitted"), out)

    def test_last_message_kept_even_over_limit(self):
        out = format_history([turn("user", "x" * 50)], limit=10)
        self.assertEqual(out, "### USER\n" + "x" * 50)

    def test_huge_old_reply_cuts_everything_before_it(self):
        turns = [turn("user", "q1"), turn("assistant", "y" * MAX_CHAT_HISTORY_CHARS), turn("user", "q2")]
        out = format_history(turns)
        self.assertTrue(out.startswith("(2 earlier messages omitted"), out[:80])
        self.assertTrue(out.endswith("### USER\nq2"))
        self.assertNotIn("q1", out)

    def test_runner_prompt_is_bounded(self):
        backend = ScriptedBackend(["r" * 4000] * 30)
        chat = new_chat()
        for i in range(30):
            say(chat, backend, f"m{i} " + "u" * 3990)
        self.assertLess(len(backend.requests[-1].prompt), MAX_CHAT_HISTORY_CHARS + 1000)
        self.assertIn("earlier messages omitted", backend.requests[-1].prompt)
        self.assertIn("m29 ", backend.requests[-1].prompt)


class ChatRulesTests(unittest.TestCase):
    def test_message_while_busy_is_refused(self):
        chat = new_chat()
        refused = []

        def second_message():
            with self.assertRaises(ChatBusy):
                chat.add_user_message("encore")
            refused.append(True)
        say(chat, HookedBackend(["ok"], {0: second_message}), "premier")
        self.assertEqual(refused, [True])
        self.assertEqual([t.content for t in chat.turns], ["premier", "ok"])

    def test_invalid_messages(self):
        chat = new_chat()
        for bad in ("", "   ", "x" * (MAX_USER_MESSAGE_CHARS + 1)):
            with self.assertRaises(InvalidUserMessage):
                chat.add_user_message(bad)
        self.assertEqual((chat.turns, chat.state), ([], "idle"))

    def test_message_cap(self):
        chat = new_chat()
        backend = ScriptedBackend(["r"] * (MAX_CHAT_MESSAGES // 2))
        for _ in range(MAX_CHAT_MESSAGES // 2):
            say(chat, backend, "m")
        with self.assertRaises(TooManyUserMessages):
            chat.add_user_message("one more")
        self.assertEqual(len(chat.turns), MAX_CHAT_MESSAGES)

    def test_failure_drops_unanswered_message_and_unlocks(self):
        for error, kind in ((BackendTimeout("t/o"), "timeout"), (BackendError("boom"), "backend_error")):
            chat = new_chat()
            say(chat, ScriptedBackend(["ok"]), "first")
            say(chat, ScriptedBackend([], error=error), "second")
            self.assertEqual(chat.state, "idle")
            self.assertEqual([t.content for t in chat.turns], ["first", "ok"])
            self.assertEqual({k: chat.events[-1][k] for k in ("type", "kind")}, {"type": "reply_failed", "kind": kind})
            say(chat, ScriptedBackend(["retry ok"]), "second again")
            self.assertEqual(chat.turns[-1].content, "retry ok")


class ChatConfigTests(unittest.TestCase):
    registry = {"claude": ClaudeBackend(), "mock": MockBackend()}

    def test_valid(self):
        ChatConfig(agent="mock").validate(self.registry)
        with mock.patch.object(ClaudeBackend, "available", return_value=True):
            ChatConfig(agent="claude", model="haiku", effort="low").validate(self.registry)

    def test_invalid(self):
        cases = [(ChatConfig(agent="nope"), "unknown agent"),
                 (ChatConfig(agent="mock", model="gpt-9"), "not offered by mock"),
                 (ChatConfig(agent="mock", effort="max"), "not offered by mock")]
        for cfg, msg in cases:
            with self.assertRaises(InvalidChatConfig) as ctx:
                cfg.validate(self.registry)
            self.assertIn(msg, str(ctx.exception))


class ChatSnapshotTests(unittest.TestCase):
    def test_round_trip(self):
        chat = new_chat(model="haiku")
        say(chat, ScriptedBackend(["ok"]), "hello")
        back = Chat.from_snapshot(chat.snapshot())
        self.assertEqual((back.id, back.state, back.config, back.turns, back.events),
                         (chat.id, "idle", chat.config, chat.turns, chat.events))
        say(back, ScriptedBackend(["again"]), "more")
        self.assertEqual(len(back.turns), 4)

    def test_reloaded_busy_chat_is_unlocked(self):
        chat = new_chat()
        say(chat, ScriptedBackend(["ok"]), "first")
        chat.add_user_message("lost in the crash")  # busy, reply never came
        back = Chat.from_snapshot(chat.snapshot())
        self.assertEqual(back.state, "idle")
        self.assertEqual([t.content for t in back.turns], ["first", "ok"])
        self.assertEqual((back.events[-1]["type"], back.events[-1]["kind"]), ("reply_failed", "interrupted"))
        say(back, ScriptedBackend(["fine"]), "next")  # not ChatBusy


if __name__ == "__main__":
    unittest.main()
