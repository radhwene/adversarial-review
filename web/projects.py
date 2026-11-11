"""Projects: one sub-folder of the projects root (default ``~/lldebat``) per project.

Layout::

    ~/lldebat/<project>/FADO.csv            append-only trace of every debate event (never truncated)
    ~/lldebat/<project>/debates/<id>.json   full snapshot of each debate (turns + events), for replay
"""
from __future__ import annotations

import csv
import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger("advr.projects")

DEFAULT_PROJECTS_ROOT = os.path.expanduser("~/lldebat")
DEFAULT_PROJECT = "default"
FADO_FILENAME = "FADO.csv"
DEBATES_DIRNAME = "debates"
# Letters, digits, '-' and '_' only: no '/', no '..', so a name can never escape the root.
PROJECT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
DEBATE_ID_RE = re.compile(r"^[0-9a-f]{12}$")
FADO_COLUMNS = ["timestamp", "project", "debate_id", "seq", "event", "role", "agent", "round",
                "question", "status", "duration_s", "content"]
# Cells starting with these are run as formulas by Excel/LibreOffice (CSV injection, OWASP).
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


class InvalidProjectName(ValueError):
    pass


class UnknownProject(LookupError):
    pass


def _csv_safe(value) -> str:
    text = "" if value is None else str(value)
    return "'" + text if text.startswith(_FORMULA_PREFIXES) else text


def fado_row(project: str, debate_id: str, question: str, event: Dict) -> List[str]:
    """Flatten one engine event into a FADO.csv row (same column order as FADO_COLUMNS)."""
    turn = event.get("turn") or {}
    status = turn.get("status") if turn else event.get("verdict")
    content = turn.get("content") or event.get("message") or ""
    row = {
        "timestamp": datetime.fromtimestamp(event.get("ts", time.time()), timezone.utc).isoformat(),
        "project": project,
        "debate_id": debate_id,
        "seq": event.get("seq"),
        "event": event.get("type"),
        "role": turn.get("role") or event.get("role") or "",
        "agent": turn.get("agent") or event.get("agent") or "",
        "round": turn.get("round", event.get("round", "")),
        "question": question if event.get("type") == "debate_started" else "",
        "status": json.dumps(status, ensure_ascii=False) if status else "",
        "duration_s": turn.get("duration_s", ""),
        "content": content,
    }
    return [_csv_safe(row[c]) for c in FADO_COLUMNS]


class ProjectStore:
    """Creates/lists project folders and appends to their FADO.csv (thread-safe)."""

    def __init__(self, root: str = DEFAULT_PROJECTS_ROOT) -> None:
        self.root = os.path.abspath(root)
        self._lock = threading.Lock()
        os.makedirs(self.root, exist_ok=True)

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
        """Create the project folder and its CSV header. Returns False if it already existed."""
        path = self.path(name)
        with self._lock:
            existed = os.path.isdir(path)
            os.makedirs(os.path.join(path, DEBATES_DIRNAME), exist_ok=True)
            fado = os.path.join(path, FADO_FILENAME)
            if not os.path.exists(fado):
                with open(fado, "w", newline="", encoding="utf-8") as fh:
                    csv.writer(fh).writerow(FADO_COLUMNS)
        if not existed:
            logger.info("project created: %s", path)
        return not existed

    def list(self) -> List[Dict]:
        out = []
        for name in sorted(os.listdir(self.root)):
            if PROJECT_NAME_RE.match(name) and os.path.isdir(os.path.join(self.root, name)):
                out.append({"name": name, "debates": len(self.debate_summaries(name))})
        return out

    def append_event(self, project: str, debate_id: str, question: str, event: Dict) -> None:
        if not self.exists(project):
            raise UnknownProject(project)
        row = fado_row(project, debate_id, question, event)
        with self._lock, open(os.path.join(self.path(project), FADO_FILENAME), "a",
                              newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(row)

    def save_debate(self, project: str, snapshot: Dict) -> str:
        path = os.path.join(self.path(project), DEBATES_DIRNAME, f"{snapshot['id']}.json")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(snapshot, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, path)  # atomic: a reader never sees a half-written snapshot
        return path

    def load_debate(self, debate_id: str) -> Optional[Dict]:
        """Find a saved debate in any project (ids are globally unique)."""
        if not DEBATE_ID_RE.match(debate_id):
            return None
        for project in self.list():
            path = os.path.join(self.root, project["name"], DEBATES_DIRNAME, f"{debate_id}.json")
            if os.path.exists(path):
                with open(path, encoding="utf-8") as fh:
                    return json.load(fh)
        return None

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
