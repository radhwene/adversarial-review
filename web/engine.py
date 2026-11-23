"""Debate engine: proposer vs critic over N rounds, then an impartial judge.

Flow::

    question ─► proposer opening (round 0)
                 │
                 ▼
        ┌─► critic (round r) ── VERDICT: ACCEPT ──► judge ─► done
        │        │ CONTEST
        │        ▼
        └── proposer rebuttal (round r)   (until r == max_rounds)
                                             └──────────► judge ─► done

Every step appends an event to ``Debate.events``; the HTTP layer replays them
over SSE from any index, so late or reconnecting clients miss nothing.

The human can add remarks at any time (``Debate.add_user_message``). They wait in a
queue and enter the transcript before the next agent turn. A remark on a finished
debate reopens it for a bonus round (proposer, critic) and a new verdict::

    remark ─► debate running?  ── yes ─► queued, read before the next turn
                    │ no (completed)
                    ▼
              bonus round (proposer, critic) ─► judge ─► completed again
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from string import Template
from typing import Callable, Dict, List, Optional

from backends import Backend, BackendError, BackendTimeout, TurnRequest

logger = logging.getLogger(__name__)

PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
MIN_ROUNDS = 1
MAX_ROUNDS = 5
MAX_QUESTION_CHARS = 4000
MAX_USER_MESSAGE_CHARS = 4000
MAX_USER_MESSAGES = 20       # per debate: bounds the transcript size and the quota spent on bonus rounds
MAX_BONUS_ROUNDS = 3         # reopenings of a finished debate
USER_ROLE = "user"

SEAT_ROLES = ("proposer", "critic", "judge")
ROLE_STATUS_BLOCK = {"proposer": "PROPOSER_STATUS", "critic": "CRITIC_STATUS", "judge": "JUDGE_STATUS"}
VERDICT_ACCEPT = "ACCEPT"

_STATUS_LINE = re.compile(r"^\s*([A-Z_]+)\s*:\s*(.*?)\s*$")


def parse_status_block(text: str, block: str) -> Optional[Dict[str, str]]:
    """Return the ``KEY: value`` pairs of the last ``---BLOCK---`` section, or None if absent."""
    pattern = re.compile(rf"---{block}---\s*\n(.*?)\n\s*---END_{block}---", re.DOTALL)
    matches = pattern.findall(text)
    if not matches:
        return None
    fields: Dict[str, str] = {}
    for line in matches[-1].splitlines():
        m = _STATUS_LINE.match(line)
        if m:
            fields[m.group(1)] = m.group(2)
    return fields


def strip_status_block(text: str, block: str) -> str:
    return re.sub(rf"\n*---{block}---.*?---END_{block}---\s*", "", text, flags=re.DOTALL).strip()


def load_prompt(name: str) -> Template:
    with open(os.path.join(PROMPTS_DIR, f"{name}.md"), encoding="utf-8") as fh:
        return Template(fh.read())


class UserMessageRejected(Exception):
    """Base for remarks the debate cannot take."""


class InvalidUserMessage(UserMessageRejected):
    """Empty or too long."""


class DebateClosed(UserMessageRejected):
    """Failed debate, or a finished one that used all its bonus rounds."""


class TooManyUserMessages(UserMessageRejected):
    """MAX_USER_MESSAGES reached."""


def clean_user_message(text: str) -> str:
    """Stripped remark, or InvalidUserMessage when empty or too long."""
    text = text.strip()
    if not text:
        raise InvalidUserMessage("message is empty")
    if len(text) > MAX_USER_MESSAGE_CHARS:
        raise InvalidUserMessage(f"message exceeds {MAX_USER_MESSAGE_CHARS} chars")
    return text


class InvalidDebateConfig(ValueError):
    pass


@dataclass
class DebateConfig:
    question: str
    proposer: str
    critic: str
    judge: str
    max_rounds: int = 3
    project: str = "default"
    # per seat, e.g. {"proposer": {"model": "opus", "effort": "high"}}; a missing key = backend default
    options: Dict[str, Dict[str, str]] = field(default_factory=dict)

    def validate(self, registry: Dict[str, Backend]) -> None:
        if not self.question.strip():
            raise InvalidDebateConfig("question is empty")
        if len(self.question) > MAX_QUESTION_CHARS:
            raise InvalidDebateConfig(f"question exceeds {MAX_QUESTION_CHARS} chars")
        if not (MIN_ROUNDS <= self.max_rounds <= MAX_ROUNDS):
            raise InvalidDebateConfig(f"max_rounds must be in [{MIN_ROUNDS}, {MAX_ROUNDS}]")
        for role in SEAT_ROLES:
            name = getattr(self, role)
            if name not in registry:
                raise InvalidDebateConfig(f"unknown backend for {role}: {name!r}")
            if not registry[name].available():
                raise InvalidDebateConfig(f"backend {name!r} ({role}) is not installed")
        self._validate_options(registry)

    def _validate_options(self, registry: Dict[str, Backend]) -> None:
        if not isinstance(self.options, dict):
            raise InvalidDebateConfig("options must be an object")
        for role, opts in self.options.items():
            if role not in SEAT_ROLES:
                raise InvalidDebateConfig(f"options: unknown seat {role!r}")
            if not isinstance(opts, dict):
                raise InvalidDebateConfig(f"options.{role} must be an object")
            backend = registry[getattr(self, role)]
            for key, value in opts.items():
                allowed = {"model": backend.models, "effort": backend.efforts}.get(key)
                if allowed is None:
                    raise InvalidDebateConfig(f"options.{role}: unknown option {key!r}")
                if value not in allowed:
                    raise InvalidDebateConfig(f"options.{role}.{key}: {value!r} not offered by {backend.name}"
                                              + (f" (choose {', '.join(allowed)})" if allowed else ""))

    def seat(self, role: str, key: str) -> Optional[str]:
        return self.options.get(role, {}).get(key)


@dataclass
class Turn:
    role: str
    agent: str
    round: int
    content: str
    status: Optional[Dict[str, str]]
    duration_s: float


class EventLog:
    """Append-only event list with blocking readers. Shared by debates and chats.

    Subclasses provide ``events``, ``observers``, ``state``, ``_cond`` and ``finished``."""

    events: List[Dict]
    observers: List[Callable]
    state: str
    _cond: threading.Condition

    @property
    def finished(self) -> bool:  # pragma: no cover - overridden
        raise NotImplementedError

    def emit(self, type_: str, final_state: Optional[str] = None, **data) -> None:
        """Append an event; ``final_state`` is set atomically with it so readers never see
        ``finished`` before the terminal event exists."""
        with self._cond:
            if final_state is not None:
                self.state = final_state
            event = {"seq": len(self.events), "type": type_, "ts": time.time(), **data}
            self.events.append(event)
            # observers run before readers are woken: once a client sees an event, it is already traced
            for observer in self.observers:
                observer(self, event)
            self._cond.notify_all()

    def wait_events(self, start: int, timeout: float) -> List[Dict]:
        """Block until events beyond ``start`` exist (or timeout); return them."""
        with self._cond:
            self._cond.wait_for(lambda: len(self.events) > start or self.finished, timeout=timeout)
            return list(self.events[start:])


@dataclass
class Debate(EventLog):
    config: DebateConfig
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    state: str = "pending"  # pending | running | completed | failed
    turns: List[Turn] = field(default_factory=list)
    events: List[Dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    # called with each new event under the lock, before readers wake (e.g. the trace-debat.csv trace)
    observers: List[Callable[["Debate", Dict], None]] = field(default_factory=list, repr=False)
    bonus_rounds: int = 0
    # remarks not yet read by an agent; drained into ``turns`` before the next turn
    pending_user: List[str] = field(default_factory=list, repr=False)
    # RLock: emit() re-enters it from add_user_message()
    _cond: threading.Condition = field(default_factory=lambda: threading.Condition(threading.RLock()), repr=False)

    @property
    def finished(self) -> bool:
        return self.state in ("completed", "failed")

    def user_message_count(self) -> int:
        return sum(1 for t in self.turns if t.role == USER_ROLE) + len(self.pending_user)

    def add_user_message(self, text: str) -> bool:
        """Queue a remark from the human. Returns True when it reopened a completed debate:
        the caller must then run ``DebateRunner.resume``."""
        text = clean_user_message(text)
        with self._cond:
            if self.state == "failed":
                raise DebateClosed("this debate failed; start a new one")
            if self.user_message_count() >= MAX_USER_MESSAGES:
                raise TooManyUserMessages(f"at most {MAX_USER_MESSAGES} messages per debate")
            reopen = self.state == "completed"
            if reopen and self.bonus_rounds >= MAX_BONUS_ROUNDS:
                raise DebateClosed(f"this debate already had {MAX_BONUS_ROUNDS} bonus rounds")
            self.pending_user.append(text)
            if reopen:
                self.bonus_rounds += 1
                self.emit("debate_resumed", final_state="running", bonus=self.bonus_rounds)
            self.emit("user_message", text=text)
            return reopen

    def take_user_messages(self, round_: int) -> List["Turn"]:
        """Move queued remarks into the transcript, before the turn of ``round_``."""
        with self._cond:
            texts, self.pending_user = self.pending_user, []
            new = [Turn(role=USER_ROLE, agent=USER_ROLE, round=round_, content=t, status=None, duration_s=0.0)
                   for t in texts]
            self.turns.extend(new)
            return new

    def complete_unless_pending(self, **data) -> bool:
        """Mark the debate completed, unless a remark arrived during the last turn: then
        count a bonus round and return False so the runner answers it first."""
        with self._cond:
            if self.pending_user and self.bonus_rounds < MAX_BONUS_ROUNDS:
                self.bonus_rounds += 1
                return False
            # out of bonus rounds: keep the late remarks in the transcript, unanswered
            for remark in self.take_user_messages(self.turns[-1].round if self.turns else 0):
                self.emit("turn_completed", turn=asdict(remark))
            self.emit("debate_completed", final_state="completed", **data)
            return True

    def snapshot(self) -> Dict:
        return {
            "id": self.id,
            "state": self.state,
            "config": asdict(self.config),
            "turns": [asdict(t) for t in self.turns],
            "created_at": self.created_at,
            "bonus_rounds": self.bonus_rounds,
            "events": list(self.events),
        }

    @classmethod
    def from_snapshot(cls, snap: Dict) -> "Debate":
        """Rebuild a finished debate from disk so its events can be replayed over SSE."""
        debate = cls(config=DebateConfig(**snap["config"]), id=snap["id"], state=snap["state"],
                     created_at=snap.get("created_at", time.time()), bonus_rounds=snap.get("bonus_rounds", 0))
        debate.turns = [Turn(**t) for t in snap.get("turns", [])]
        debate.events = list(snap.get("events", []))
        if not debate.finished:  # server died mid-debate: close it so readers stop waiting
            debate.state = "failed"
            debate.events.append({"seq": len(debate.events), "type": "debate_failed", "ts": time.time(),
                                  "kind": "interrupted", "message": "server stopped during this debate"})
        return debate


def format_transcript(turns: List[Turn]) -> str:
    parts = []
    for t in turns:
        label = "Opening" if t.round == 0 else f"Round {t.round}"
        who = "USER (the human who asked the question)" if t.role == USER_ROLE else f"{t.role.upper()} ({t.agent})"
        parts.append(f"### {label}: {who}\n{t.content}")
    return "\n\n".join(parts)


class DebateRunner:
    def __init__(self, registry: Dict[str, Backend], runs_dir: Optional[str] = None) -> None:
        self.registry = registry
        self.runs_dir = runs_dir

    def _turn(self, debate: Debate, role: str, round_: int, prompt: str) -> Turn:
        agent = getattr(debate.config, role)
        debate.emit("turn_started", role=role, agent=agent, round=round_)
        started = time.monotonic()
        cfg = debate.config
        raw = self.registry[agent].complete(TurnRequest(role=role, round=round_, prompt=prompt,
                                                        model=cfg.seat(role, "model"), effort=cfg.seat(role, "effort")))
        block = ROLE_STATUS_BLOCK[role]
        status = parse_status_block(raw, block)
        if status is None:
            logger.warning("debate=%s %s round=%d: missing %s block", debate.id, role, round_, block)
        turn = Turn(role=role, agent=agent, round=round_, content=strip_status_block(raw, block),
                    status=status, duration_s=round(time.monotonic() - started, 2))
        debate.turns.append(turn)
        debate.emit("turn_completed", turn=asdict(turn))
        return turn

    def _speak(self, debate: Debate, role: str, round_: int, template: str, **params) -> Turn:
        """Read the human's queued remarks into the transcript, then run one agent turn on it."""
        for remark in debate.take_user_messages(round_):
            debate.emit("turn_completed", turn=asdict(remark))
        prompt = load_prompt(template).substitute(question=debate.config.question,
                                                  transcript=format_transcript(debate.turns), **params)
        return self._turn(debate, role, round_, prompt)

    def run(self, debate: Debate) -> None:
        cfg = debate.config
        debate.state = "running"
        debate.emit("debate_started", config=asdict(cfg))
        try:
            self._turn(debate, "proposer", 0, load_prompt("proposer_opening").substitute(question=cfg.question))
            converged_round: Optional[int] = None
            for r in range(1, cfg.max_rounds + 1):
                critique = self._speak(debate, "critic", r, "critic", round=r, max_rounds=cfg.max_rounds)
                verdict = (critique.status or {}).get("VERDICT", "").strip().upper()
                if verdict == VERDICT_ACCEPT:
                    converged_round = r
                    debate.emit("converged", round=r)
                    break
                self._speak(debate, "proposer", r, "proposer_rebuttal", round=r, max_rounds=cfg.max_rounds)
            judge = self._speak(debate, "judge", cfg.max_rounds + 1, "judge")
            if debate.complete_unless_pending(converged_round=converged_round, verdict=judge.status):
                return
            self._bonus_rounds(debate)
        except BackendTimeout as exc:
            self._fail(debate, "timeout", str(exc))
        except BackendError as exc:
            self._fail(debate, "backend_error", str(exc))
        finally:
            self._persist(debate)

    def resume(self, debate: Debate) -> None:
        """Answer a remark posted after the verdict: one bonus round, then a new verdict."""
        try:
            self._bonus_rounds(debate)
        except BackendTimeout as exc:
            self._fail(debate, "timeout", str(exc))
        except BackendError as exc:
            self._fail(debate, "backend_error", str(exc))
        finally:
            self._persist(debate)

    def _bonus_rounds(self, debate: Debate) -> None:
        """Proposer answers the remark, critic replies, judge rules again. Repeats while new
        remarks arrive during the round (up to MAX_BONUS_ROUNDS)."""
        while True:
            r = debate.config.max_rounds + debate.bonus_rounds
            debate.emit("bonus_round_started", bonus=debate.bonus_rounds, round=r)
            self._speak(debate, "proposer", r, "proposer_rebuttal", round=r, max_rounds=r)
            self._speak(debate, "critic", r, "critic", round=r, max_rounds=r)
            judge = self._speak(debate, "judge", r + 1, "judge")
            if debate.complete_unless_pending(converged_round=None, verdict=judge.status, bonus=debate.bonus_rounds):
                return

    def _fail(self, debate: Debate, kind: str, message: str) -> None:
        logger.error("debate=%s failed (%s): %s", debate.id, kind, message)
        debate.emit("debate_failed", final_state="failed", kind=kind, message=message)

    def _persist(self, debate: Debate) -> None:
        if not self.runs_dir:
            return
        os.makedirs(self.runs_dir, exist_ok=True)
        path = os.path.join(self.runs_dir, f"{debate.id}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(debate.snapshot(), fh, indent=2, ensure_ascii=False)
