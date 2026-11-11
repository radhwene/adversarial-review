"""API keys typed in the web UI, stored on the llm-auth volume (one file per provider, mode 0600).

Only DeepSeek needs one: it has no subscription CLI, every call is billed per token. The key is
never logged and never sent back to the browser; ``status()`` exposes its last four characters.
An environment variable stays a fallback for deployments that inject secrets that way.
"""
from __future__ import annotations

import logging
import os
import re
import tempfile
import threading
from typing import Dict, Optional

logger = logging.getLogger(__name__)

KEY_FILE = "api_key"
HINT_CHARS = 4
SOURCE_FILE = "file"
SOURCE_ENV = "env"


class InvalidApiKey(ValueError):
    """The submitted key does not look like a key for this provider (its value is never echoed)."""


class ApiKeyStore:
    """Read on every call, so a key saved from the UI takes effect without a restart."""

    def __init__(self, directory: str, env_var: str, pattern: "re.Pattern[str]") -> None:
        self.directory = directory
        self.env_var = env_var
        self.pattern = pattern
        self._path = os.path.join(directory, KEY_FILE)
        self._lock = threading.Lock()

    def _from_file(self) -> str:
        try:
            with open(self._path, encoding="utf-8") as fh:
                return fh.read().strip()
        except FileNotFoundError:
            return ""

    def get(self) -> str:
        """The saved key, else the environment variable, else ""."""
        return self._from_file() or os.environ.get(self.env_var, "").strip()

    def source(self) -> Optional[str]:
        if self._from_file():
            return SOURCE_FILE
        return SOURCE_ENV if os.environ.get(self.env_var, "").strip() else None

    def status(self) -> Dict:
        key, source = self.get(), self.source()
        return {"configured": bool(key), "source": source,
                "hint": "…" + key[-HINT_CHARS:] if key else None}

    def set(self, key: str) -> Dict:
        key = key.strip()  # phone pastes carry spaces and newlines
        if not self.pattern.fullmatch(key):
            raise InvalidApiKey("clé refusée : format inattendu")
        with self._lock:
            os.makedirs(self.directory, mode=0o700, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self.directory, prefix=".key-")  # created 0600
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(key)
                os.replace(tmp, self._path)
            except BaseException:
                os.unlink(tmp)
                raise
        logger.info("api key saved in %s (hint …%s)", self.directory, key[-HINT_CHARS:])
        return self.status()

    def clear(self) -> Dict:
        """Delete the saved key. A key from the environment cannot be cleared from here."""
        with self._lock:
            try:
                os.remove(self._path)
                logger.info("api key removed from %s", self.directory)
            except FileNotFoundError:
                pass
        return self.status()
