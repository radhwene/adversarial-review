"""Direct chat with one agent, with memory.

The agent CLIs are stateless (one prompt in, one answer out), so the memory lives here: every
reply is generated from the conversation so far, newest messages first up to
MAX_CHAT_HISTORY_CHARS. One reply at a time per chat; a second message while the agent is
answering is refused (ChatBusy), not queued.

Events (same EventLog as debates, replayed over SSE):
    chat_started     {config}
    turn_completed   {turn}           a user message (state -> busy) or the agent's reply (-> idle)
    reply_failed     {kind, message}  the agent failed; the unanswered user message is dropped (-> idle)
    chat_renamed     {title}          the human renamed the chat ("" title means: back to the first message)

While the agent answers, the text generated so far is kept in ``draft`` (memory only, never saved
and not an event: one save per token would rewrite the file hundreds of times). SSE readers get it
through ``wait_update`` as {round, offset, text} pieces; the final reply replaces it.
"""

from __future__ import annotations

import logging
import threading
import time
import os
import uuid
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from attachments import media_type_of
from backends import Backend, BackendError, BackendTimeout, ImageFile, TurnRequest
from engine import (USER_ROLE, EventLog, InvalidUserMessage, Turn, TooManyUserMessages, UserMessageRejected,
                    clean_user_message, load_prompt, option_error)

logger = logging.getLogger(__name__)

CHAT_ROLE = "chat"             # TurnRequest.role for a chat reply
ASSISTANT_ROLE = "assistant"
MAX_CHAT_HISTORY_CHARS = 60_000  # sent with each reply: bounds prompt size, cost and latency
MAX_CHAT_MESSAGES = 200          # user + assistant, per chat: bounds the snapshot on disk
OMITTED_MARKER = "({n} earlier messages omitted: the conversation is longer than the memory window)"
TITLE_CHARS = 120
IMAGE_ONLY_TITLE = "Image"       # title of a chat whose first message is a photo without text
# Images sent with one reply. Every reply resends the history, so without this cap one early photo
# would be billed again on every turn; older images stay in the text as "[image N]" only.
MAX_IMAGES_PER_REPLY = 4
IMAGE_LABEL = "[image {n}]"

__all__ = ["Chat", "ChatBusy", "ChatConfig", "ChatDeleted", "ChatRunner", "InvalidChatConfig", "InvalidChatTitle",
           "InvalidUserMessage", "TooManyUserMessages", "format_history"]


class InvalidChatConfig(ValueError):
    pass


class ChatBusy(UserMessageRejected):
    """The agent is still answering the previous message."""


class ChatDeleted(UserMessageRejected):
    """The chat was deleted while this request was on its way: nothing may write it back to disk."""


class InvalidChatTitle(ValueError):
    pass


def clean_title(title: str) -> Optional[str]:
    """One line, at most TITLE_CHARS; empty means "no custom title" (None)."""
    title = " ".join(title.split())
    if len(title) > TITLE_CHARS:
        raise InvalidChatTitle(f"title must be at most {TITLE_CHARS} characters")
    return title or None


@dataclass
class ChatConfig:
    agent: str
    project: str = "default"
    model: Optional[str] = None   # None: the backend's default
    effort: Optional[str] = None

    def validate(self, registry: Dict[str, Backend]) -> None:
        backend = registry.get(self.agent)
        if backend is None:
            raise InvalidChatConfig(f"unknown agent: {self.agent!r}")
        if not backend.available():
            raise InvalidChatConfig(f"agent {self.agent!r} is not installed")
        for key in ("model", "effort"):
            value = getattr(self, key)
            error = None if value is None else option_error(backend, key, value)
            if error:
                raise InvalidChatConfig(f"{error.lstrip('.: ')}")


def clean_chat_text(text: str, has_images: bool) -> str:
    """Like clean_user_message, but a photo may come without text (as in the Claude app)."""
    if has_images and not text.strip():
        return ""
    return clean_user_message(text)


def image_numbers(turns: List[Turn]) -> List[Tuple[int, str]]:
    """(number, name) of every image in the chat, numbered from 1 in the order they were sent."""
    names = [name for turn in turns for name in turn.images]
    return list(enumerate(names, 1))


def _block(turn: Turn, first_number: int) -> str:
    labels = [IMAGE_LABEL.format(n=first_number + i) for i in range(len(turn.images))]
    body = "\n".join(labels + ([turn.content] if turn.content else []))
    return f"### {'USER' if turn.role == USER_ROLE else 'ASSISTANT'}\n{body}"


def history_window(turns: List[Turn], limit: int = MAX_CHAT_HISTORY_CHARS) -> int:
    """Index of the oldest turn format_history keeps."""
    return len(turns) - len(_kept_blocks(turns, limit))


def _kept_blocks(turns: List[Turn], limit: int) -> List[str]:
    """Blocks of the newest turns that fit in ``limit``, newest first; the last turn always."""
    first_numbers, n = [], 1
    for turn in turns:
        first_numbers.append(n)
        n += len(turn.images)
    blocks: List[str] = []
    used = 0
    for index in range(len(turns) - 1, -1, -1):
        block = _block(turns[index], first_numbers[index])
        if blocks and used + len(block) > limit:
            break
        blocks.append(block)
        used += len(block) + 2  # the blank line between blocks
    return blocks


def format_history(turns: List[Turn], limit: int = MAX_CHAT_HISTORY_CHARS) -> str:
    """The newest messages that fit in ``limit`` characters, oldest first. The last message is
    always kept (it is the one to answer); older ones are dropped whole, never cut. Each image
    appears as a numbered "[image N]" line in the message it was sent with."""
    blocks = _kept_blocks(turns, limit)
    omitted = len(turns) - len(blocks)
    if omitted:
        blocks.append(OMITTED_MARKER.format(n=omitted))
    return "\n\n".join(reversed(blocks))


@dataclass
class Chat(EventLog):
    config: ChatConfig
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    state: str = "idle"  # idle | busy
    turns: List[Turn] = field(default_factory=list)
    events: List[Dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    custom_title: Optional[str] = None
    deleted: bool = field(default=False, repr=False)
    draft: str = field(default="", repr=False)    # reply being generated; never saved
    draft_round: int = field(default=-1, repr=False)  # round of the reply being drafted
    files_dir: Optional[str] = field(default=None, repr=False)  # where its images are stored (set by the app)
    observers: List[Callable[["Chat", Dict], None]] = field(default_factory=list, repr=False)
    _cond: threading.Condition = field(default_factory=lambda: threading.Condition(threading.RLock()), repr=False)

    def __post_init__(self) -> None:
        if not self.events:
            self.emit("chat_started", config=asdict(self.config))

    @property
    def finished(self) -> bool:
        """Nothing more will happen until the next message: SSE readers may close."""
        return self.state != "busy"

    @property
    def title(self) -> str:
        if self.custom_title:
            return self.custom_title
        first_turn = next((t for t in self.turns if t.role == USER_ROLE), None)
        first = first_turn.content if first_turn else ""
        if not first and first_turn is not None and first_turn.images:
            return IMAGE_ONLY_TITLE
        if len(first) <= TITLE_CHARS:
            return first
        return first[:TITLE_CHARS - 1].rstrip() + "…"  # visibly cut, not silently

    def add_user_message(self, text: str, images: Sequence[str] = ()) -> None:
        """Record the human's message and mark the chat busy; the caller then runs ChatRunner.reply.
        ``images``: names of images already stored in ``files_dir``."""
        text = clean_chat_text(text, bool(images))
        with self._cond:
            if self.deleted:
                raise ChatDeleted("this chat was deleted")
            if self.state == "busy":
                raise ChatBusy("the agent is still answering; wait for its reply")
            if len(self.turns) + 2 > MAX_CHAT_MESSAGES:  # this message and its reply
                raise TooManyUserMessages(f"at most {MAX_CHAT_MESSAGES} messages per chat; start a new one")
            turn = Turn(role=USER_ROLE, agent=USER_ROLE, round=len(self.turns), content=text, status=None, duration_s=0.0,
                        images=list(images))
            self.turns.append(turn)
            self.draft, self.draft_round = "", len(self.turns)
            self.emit("turn_completed", final_state="busy", turn=asdict(turn))

    def add_reply(self, content: str, duration_s: float) -> Turn:
        with self._cond:
            turn = Turn(role=ASSISTANT_ROLE, agent=self.config.agent, round=len(self.turns), content=content,
                        status=None, duration_s=duration_s)
            self.turns.append(turn)
            self.draft = ""
            self.emit("turn_completed", final_state="idle", turn=asdict(turn))
            return turn

    def fail_reply(self, kind: str, message: str) -> None:
        """Drop the unanswered user message (the agent never read it) and go back to idle."""
        with self._cond:
            if self.turns and self.turns[-1].role == USER_ROLE:
                self.turns.pop()
            self.draft = ""
            self.emit("reply_failed", final_state="idle", kind=kind, message=message)

    def append_draft(self, text: str) -> None:
        """Add a piece of the reply being generated and wake SSE readers."""
        with self._cond:
            if self.state != "busy":
                return  # a late piece after the reply landed or failed: the final text already won
            self.draft += text
            self._cond.notify_all()

    def wait_update(self, start: int, draft_round: int, draft_sent: int,
                    timeout: float) -> Tuple[List[Dict], Optional[Dict]]:
        """Block until there are events beyond ``start`` or draft text the reader has not got yet.
        ``draft_round``/``draft_sent``: the round and length of the draft already sent to this reader.
        Returns (events, piece); piece is {round, offset, text} or None. Events come first: the user
        message that starts a draft is always delivered before that draft."""
        with self._cond:
            def unseen_draft() -> bool:
                return bool(self.draft) and (self.draft_round != draft_round or len(self.draft) > draft_sent)
            self._cond.wait_for(lambda: len(self.events) > start or self.finished or unseen_draft(), timeout=timeout)
            events = list(self.events[start:])
            piece = None
            if unseen_draft():
                offset = draft_sent if self.draft_round == draft_round else 0
                piece = {"round": self.draft_round, "offset": offset, "text": self.draft[offset:]}
            return events, piece

    def rename(self, title: str) -> None:
        """Set (or clear, with "") the title shown in the chat list. Allowed while the agent answers."""
        custom = clean_title(title)
        with self._cond:
            self.custom_title = custom
            self.emit("chat_renamed", title=self.title)

    def mark_deleted(self) -> None:
        """Refuse further messages; the caller holds ``_cond`` and has checked the chat is idle."""
        self.deleted = True

    def snapshot(self) -> Dict:
        with self._cond:
            return {"id": self.id, "kind": "chat", "state": self.state, "config": asdict(self.config),
                    "title": self.title, "custom_title": self.custom_title, "turns": [asdict(t) for t in self.turns],
                    "created_at": self.created_at, "events": list(self.events)}

    @classmethod
    def from_snapshot(cls, snap: Dict) -> "Chat":
        chat = cls(config=ChatConfig(**snap["config"]), id=snap["id"], state=snap["state"],
                   created_at=snap.get("created_at", time.time()), events=list(snap.get("events", [])),
                   custom_title=snap.get("custom_title"))
        chat.turns = [Turn(**t) for t in snap.get("turns", [])]
        if chat.state == "busy":  # server died mid-reply: unlock the chat, or every message is ChatBusy forever
            chat.fail_reply("interrupted", "server stopped during this reply; send your message again")
        return chat


class ChatRunner:
    def __init__(self, registry: Dict[str, Backend]) -> None:
        self.registry = registry

    def reply(self, chat: Chat) -> None:
        """Generate the agent's answer to the last user message. Never leaves the chat busy."""
        cfg = chat.config
        prompt = load_prompt("chat").substitute(history=format_history(chat.turns))
        started = time.monotonic()
        try:
            request = TurnRequest(role=CHAT_ROLE, round=len(chat.turns), prompt=prompt, model=cfg.model,
                                  effort=cfg.effort, images=self.images_to_send(chat))
            raw = self.registry[cfg.agent].stream(request, chat.append_draft)
        except BackendTimeout as exc:
            self._fail(chat, "timeout", str(exc))
            return
        except BackendError as exc:
            self._fail(chat, "backend_error", str(exc))
            return
        chat.add_reply(raw.strip(), round(time.monotonic() - started, 2))

    @staticmethod
    def images_to_send(chat: Chat) -> Tuple[ImageFile, ...]:
        """The newest MAX_IMAGES_PER_REPLY images among the messages the history still holds."""
        start = history_window(chat.turns)
        skipped = sum(len(turn.images) for turn in chat.turns[:start])  # images of omitted messages
        numbered = image_numbers(chat.turns)[skipped:]
        if numbered and chat.files_dir is None:
            raise BackendError("this chat has images but no folder to read them from")
        return tuple(ImageFile(number=n, path=os.path.join(chat.files_dir, name), media_type=media_type_of(name))
                     for n, name in numbered[-MAX_IMAGES_PER_REPLY:])

    @staticmethod
    def _fail(chat: Chat, kind: str, message: str) -> None:
        logger.error("chat=%s reply failed (%s): %s", chat.id, kind, message)
        chat.fail_reply(kind, message)
