"""Projects: one sub-folder of the projects root (default ``~/lldebat``) per project.

Layout::

    ~/lldebat/<project>/trace-debat.csv         append-only trace of every debate event (never truncated)
    ~/lldebat/<project>/debates/<id>.json   full snapshot of each debate (turns + events), for replay
"""
from __future__ import annotations

import csv
import json
import logging
import os
import re
import shutil
import threading
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

logger = logging.getLogger("advr.projects")

DEFAULT_PROJECTS_ROOT = os.path.expanduser("~/lldebat")
DEFAULT_PROJECT = "default"
TRACE_FILENAME = "trace-debat.csv"
# Name used before the rename. A legacy file holding data is kept untouched, never rewritten.
LEGACY_TRACE_FILENAME = "FADO.csv"
DEBATES_DIRNAME = "debates"
CHATS_DIRNAME = "chats"
CHAT_FILES_SUFFIX = ".files"  # chats/<id>.files/<sha256>.<ext>
# Letters, digits, '-' and '_' only: no '/', no '..', so a name can never escape the root.
PROJECT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
DEBATE_ID_RE = re.compile(r"^[0-9a-f]{12}$")
TRACE_COLUMNS = ["timestamp", "project", "debate_id", "algorithm", "seq", "event", "role", "seat", "agent",
                   "round", "question", "status", "duration_s", "content"]
# Cells starting with these are run as formulas by Excel/LibreOffice (CSV injection, OWASP).
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


class InvalidProjectName(ValueError):
    pass


class UnknownProject(LookupError):
    pass


class TraceHeaderMismatch(RuntimeError):
    """trace-debat.csv exists with other columns: appending would silently misalign every row."""


def _csv_safe(value) -> str:
    text = "" if value is None else str(value)
    return "'" + text if text.startswith(_FORMULA_PREFIXES) else text


def trace_row(project: str, debate_id: str, algorithm: str, question: str, event: Dict) -> List[str]:
    """Flatten one engine event into a trace-debat.csv row (same column order as TRACE_COLUMNS)."""
    turn = event.get("turn") or {}
    status = turn.get("status") if turn else event.get("verdict")
    content = turn.get("content") or event.get("message") or ""
    row = {
        "timestamp": datetime.fromtimestamp(event.get("ts", time.time()), timezone.utc).isoformat(),
        "project": project,
        "debate_id": debate_id,
        "algorithm": algorithm,
        "seq": event.get("seq"),
        "event": event.get("type"),
        "role": turn.get("role") or event.get("role") or "",
        "seat": turn.get("seat") if turn.get("seat") is not None else "",
        "agent": turn.get("agent") or event.get("agent") or "",
        "round": turn.get("round", event.get("round", "")),
        "question": question if event.get("type") == "debate_started" else "",
        "status": json.dumps(status, ensure_ascii=False) if status else "",
        "duration_s": turn.get("duration_s", ""),
        "content": content,
    }
    return [_csv_safe(row[c]) for c in TRACE_COLUMNS]


class ProjectStore:
    """Creates/lists project folders and appends to their trace-debat.csv (thread-safe)."""

    def __init__(self, root: str = DEFAULT_PROJECTS_ROOT) -> None:
        self.root = os.path.abspath(root)
        self._lock = threading.Lock()
        self._checked_traces: Set[str] = set()  # traces whose header already matched TRACE_COLUMNS
        os.makedirs(self.root, exist_ok=True)
        for name in os.listdir(self.root):
            if PROJECT_NAME_RE.match(name) and os.path.isdir(os.path.join(self.root, name)):
                self._migrate_legacy_trace(os.path.join(self.root, name))

    @staticmethod
    def _migrate_legacy_trace(folder: str) -> None:
        """FADO.csv -> trace-debat.csv. A header-only legacy file is replaced; one with data is kept as is
        (its 12 columns cannot be appended to under the new header) and a fresh trace starts."""
        legacy = os.path.join(folder, LEGACY_TRACE_FILENAME)
        if not os.path.exists(legacy) or os.path.exists(os.path.join(folder, TRACE_FILENAME)):
            return
        with open(legacy, newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            next(reader, None)
            has_data = next(reader, None) is not None
        if has_data:
            logger.warning("legacy %s kept read-only (has data); new events go to %s",
                           legacy, TRACE_FILENAME)
            return
        os.remove(legacy)
        logger.info("removed empty legacy %s (replaced by %s)", legacy, TRACE_FILENAME)

    def _ensure_trace(self, path: str) -> None:
        """Write the header of a new trace, or check an existing one. Caller holds the lock."""
        if path in self._checked_traces:
            return
        if not os.path.exists(path):
            with open(path, "w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerow(TRACE_COLUMNS)
        else:
            with open(path, newline="", encoding="utf-8") as fh:
                header = next(csv.reader(fh), None)
            if header != TRACE_COLUMNS:
                raise TraceHeaderMismatch(f"{path}: header {header} != {TRACE_COLUMNS}")
        self._checked_traces.add(path)

    @staticmethod
    def validate_name(name: str) -> str:
        if not isinstance(name, str) or not PROJECT_NAME_RE.match(name):
            raise InvalidProjectName("project name must be 1-64 chars: letters, digits, '-' or '_'")
        return name

    def path(self, name: str) -> str:
        return os.path.join(self.root, self.validate_name(name))

    def exists(self, name: str) -> bool:
        return os.path.isdir(self.path(name))

    def create(self, name: str) -> bool:
        """Create the project folder and its trace header. Returns False if it already existed."""
        path = self.path(name)
        with self._lock:
            existed = os.path.isdir(path)
            os.makedirs(os.path.join(path, DEBATES_DIRNAME), exist_ok=True)
            self._migrate_legacy_trace(path)
            self._ensure_trace(os.path.join(path, TRACE_FILENAME))
        if not existed:
            logger.info("project created: %s", path)
        return not existed

    def names(self) -> List[str]:
        return [n for n in sorted(os.listdir(self.root))
                if PROJECT_NAME_RE.match(n) and os.path.isdir(os.path.join(self.root, n))]

    def list(self) -> List[Dict]:
        return [{"name": n, "debates": len(self.debate_summaries(n))} for n in self.names()]

    def append_event(self, project: str, debate_id: str, algorithm: str, question: str, event: Dict) -> None:
        if not self.exists(project):
            raise UnknownProject(project)
        row = trace_row(project, debate_id, algorithm, question, event)
        path = os.path.join(self.path(project), TRACE_FILENAME)
        with self._lock:
            self._ensure_trace(path)
            with open(path, "a", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerow(row)

    def _save(self, project: str, dirname: str, snapshot: Dict) -> str:
        folder = os.path.join(self.path(project), dirname)
        os.makedirs(folder, exist_ok=True)  # chats/ is absent in projects created before chats existed
        path = os.path.join(folder, f"{snapshot['id']}.json")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(snapshot, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, path)  # atomic: a reader never sees a half-written snapshot
        return path

    def _load(self, dirname: str, item_id: str) -> Optional[Dict]:
        """Find a saved item in any project (ids are globally unique)."""
        if not DEBATE_ID_RE.match(item_id):
            return None
        for name in sorted(os.listdir(self.root)):
            path = os.path.join(self.root, name, dirname, f"{item_id}.json")
            if PROJECT_NAME_RE.match(name) and os.path.exists(path):
                with open(path, encoding="utf-8") as fh:
                    return json.load(fh)
        return None

    def save_debate(self, project: str, snapshot: Dict) -> str:
        return self._save(project, DEBATES_DIRNAME, snapshot)

    def load_debate(self, debate_id: str) -> Optional[Dict]:
        return self._load(DEBATES_DIRNAME, debate_id)

    def save_chat(self, project: str, snapshot: Dict) -> str:
        return self._save(project, CHATS_DIRNAME, snapshot)

    def load_chat(self, chat_id: str) -> Optional[Dict]:
        return self._load(CHATS_DIRNAME, chat_id)

    def chat_files_dir(self, project: str, chat_id: str) -> str:
        """Folder of a chat's images, beside its JSON (which is rewritten on every event, so it only
        holds their names). May not exist yet."""
        if not DEBATE_ID_RE.match(chat_id):
            raise ValueError(f"invalid chat id: {chat_id!r}")
        return os.path.join(self.path(project), CHATS_DIRNAME, f"{chat_id}{CHAT_FILES_SUFFIX}")

    def save_chat_file(self, project: str, chat_id: str, name: str, data: bytes) -> bool:
        """Store one image under its content-addressed ``name``. Returns False when it was already there."""
        folder = self.chat_files_dir(project, chat_id)
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return False
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)  # atomic: a reader never serves half an image
        return True

    def delete_chat(self, project: str, chat_id: str) -> bool:
        """Remove a saved chat and its images. Its lines in trace-debat.csv stay: the trace is an
        append-only audit log."""
        if not DEBATE_ID_RE.match(chat_id):
            return False
        shutil.rmtree(self.chat_files_dir(project, chat_id), ignore_errors=True)
        try:
            os.remove(os.path.join(self.path(project), CHATS_DIRNAME, f"{chat_id}.json"))
        except FileNotFoundError:
            return False
        return True

    def chat_summaries(self, project: str) -> List[Dict]:
        if not self.exists(project):
            raise UnknownProject(project)
        folder = os.path.join(self.path(project), CHATS_DIRNAME)
        out = []
        for fname in (os.listdir(folder) if os.path.isdir(folder) else []):
            if not fname.endswith(".json"):
                continue
            try:
                with open(os.path.join(folder, fname), encoding="utf-8") as fh:
                    snap = json.load(fh)
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("skipping unreadable chat %s/%s: %s", project, fname, exc)
                continue
            if not snap.get("turns"):
                continue  # first message failed and was handed back: nothing to reopen
            cfg = snap.get("config", {})
            out.append({"id": snap.get("id"), "project": project, "created_at": snap.get("created_at"),
                        "title": snap.get("title", ""),
                        "agent": cfg.get("agent"), "model": cfg.get("model"), "messages": len(snap.get("turns", []))})
        return sorted(out, key=lambda s: s.get("created_at") or 0, reverse=True)

    def all_chat_summaries(self) -> List[Dict]:
        """Every project's chats in one list, newest first: the Conversations tab has no project picker."""
        out = [c for name in self.names() for c in self.chat_summaries(name)]
        return sorted(out, key=lambda s: s.get("created_at") or 0, reverse=True)

    def debate_summaries(self, project: str) -> List[Dict]:
        folder = os.path.join(self.path(project), DEBATES_DIRNAME)
        if not os.path.isdir(folder):
            raise UnknownProject(project)
        out = []
        for fname in os.listdir(folder):
            if not fname.endswith(".json"):
                continue
            try:
                with open(os.path.join(folder, fname), encoding="utf-8") as fh:
                    snap = json.load(fh)
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("skipping unreadable debate %s/%s: %s", project, fname, exc)
                continue
            cfg = snap.get("config", {})
            out.append({"id": snap.get("id"), "state": snap.get("state"), "created_at": snap.get("created_at"),
                        "question": cfg.get("question", ""), "proposer": cfg.get("proposer"),
                        "critic": cfg.get("critic"), "judge": cfg.get("judge")})
        return sorted(out, key=lambda s: s.get("created_at") or 0, reverse=True)
