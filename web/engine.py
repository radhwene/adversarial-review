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


@dataclass
class Debate:
    config: DebateConfig
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    state: str = "pending"  # pending | running | completed | failed
    turns: List[Turn] = field(default_factory=list)
    events: List[Dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    # called with each new event under the lock, before readers wake (e.g. the trace-debat.csv trace)
    observers: List[Callable[["Debate", Dict], None]] = field(default_factory=list, repr=False)
    _cond: threading.Condition = field(default_factory=threading.Condition, repr=False)

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

    @property
    def finished(self) -> bool:
        return self.state in ("completed", "failed")

    def snapshot(self) -> Dict:
        return {
            "id": self.id,
            "state": self.state,
            "config": asdict(self.config),
            "turns": [asdict(t) for t in self.turns],
            "created_at": self.created_at,
            "events": list(self.events),
        }

    @classmethod
    def from_snapshot(cls, snap: Dict) -> "Debate":
        """Rebuild a finished debate from disk so its events can be replayed over SSE."""
        debate = cls(config=DebateConfig(**snap["config"]), id=snap["id"], state=snap["state"],
                     created_at=snap.get("created_at", time.time()))
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
        parts.append(f"### {label}: {t.role.upper()} ({t.agent})\n{t.content}")
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

    def run(self, debate: Debate) -> None:
        cfg = debate.config
        debate.state = "running"
        debate.emit("debate_started", config=asdict(cfg))
        q = cfg.question
        try:
            self._turn(debate, "proposer", 0, load_prompt("proposer_opening").substitute(question=q))
            converged_round: Optional[int] = None
            for r in range(1, cfg.max_rounds + 1):
                critique = self._turn(debate, "critic", r, load_prompt("critic").substitute(
                    question=q, round=r, max_rounds=cfg.max_rounds, transcript=format_transcript(debate.turns)))
                verdict = (critique.status or {}).get("VERDICT", "").strip().upper()
                if verdict == VERDICT_ACCEPT:
                    converged_round = r
                    debate.emit("converged", round=r)
                    break
                self._turn(debate, "proposer", r, load_prompt("proposer_rebuttal").substitute(
                    question=q, round=r, max_rounds=cfg.max_rounds, transcript=format_transcript(debate.turns)))
            judge = self._turn(debate, "judge", cfg.max_rounds + 1, load_prompt("judge").substitute(
                question=q, transcript=format_transcript(debate.turns)))
            debate.emit("debate_completed", final_state="completed", converged_round=converged_round, verdict=judge.status)
        except BackendTimeout as exc:
            self._fail(debate, "timeout", str(exc))
        except BackendError as exc:
            self._fail(debate, "backend_error", str(exc))
        finally:
            self._persist(debate)

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
