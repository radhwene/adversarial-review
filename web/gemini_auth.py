"""``gemini`` (Gemini CLI) has no ``login`` / ``status`` / ``logout`` subcommand, so this helper gives it
the same shape as claude and codex for ``AuthManager``:

    python3 gemini_auth.py login   # runs gemini under a pseudo-terminal with NO_BROWSER=true, relays the
                                   # Google sign-in URL to stdout, writes the code read on stdin to the
                                   # terminal, exits 0 once oauth_creds.json is (re)written
    python3 gemini_auth.py status  # JSON {"logged_in": bool, "account": str|null} from the files on disk
    python3 gemini_auth.py logout  # deletes the cached Google credentials

Why a pty: gemini 0.66.0 refuses the manual code flow when stdin is not a TTY ("Manual authorization
is required but the current session is non-interactive"). Only "Login with Google" (oauth-personal)
is configured: ``GEMINI_API_KEY`` (pay-per-use API) is never used.
"""
from __future__ import annotations

import errno
import fcntl
import json
import os
import pty
import re
import select
import signal
import struct
import sys
import tempfile
import termios
import time
from typing import Dict, Optional

AUTH_TYPE = "oauth-personal"  # "Login with Google"
CREDS_FILE = "oauth_creds.json"
ACCOUNTS_FILE = "google_accounts.json"
SETTINGS_FILE = "settings.json"
API_KEY_ENVS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
TERMINAL_COLUMNS = 4000  # wide enough that the CLI never wraps the long sign-in URL
POLL_SECONDS = 0.5
CHILD_EXIT_GRACE_SECONDS = 3
TERMINAL_ESCAPE = re.compile(r"\x1b\[[0-9;?<>=]*[A-Za-z~]|\x1b\][^\x07]*\x07|\x1b[=>()][A-Za-z0-9]?")


def gemini_dir() -> str:
    """Same rule as the CLI: ``$GEMINI_CLI_HOME/.gemini``, else ``~/.gemini``."""
    return os.path.join(os.environ.get("GEMINI_CLI_HOME") or os.path.expanduser("~"), ".gemini")


def _read_json(path: str) -> Optional[Dict]:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def ensure_settings(directory: str) -> None:
    """Select "Login with Google" and turn usage statistics off, keeping any other setting."""
    os.makedirs(directory, mode=0o700, exist_ok=True)
    path = os.path.join(directory, SETTINGS_FILE)
    settings = _read_json(path) or {}
    settings.setdefault("security", {}).setdefault("auth", {})["selectedType"] = AUTH_TYPE
    settings.setdefault("privacy", {})["usageStatisticsEnabled"] = False
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".settings-")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(settings, fh, indent=2)
    os.replace(tmp, path)  # mkstemp already created it 0600


def status(directory: str) -> Dict:
    if any(os.environ.get(name) for name in API_KEY_ENVS):
        return {"logged_in": True, "account": "api_key"}
    creds = _read_json(os.path.join(directory, CREDS_FILE)) or {}
    if not creds.get("refresh_token"):
        return {"logged_in": False, "account": None}
    accounts = _read_json(os.path.join(directory, ACCOUNTS_FILE)) or {}
    active = accounts.get("active")
    return {"logged_in": True, "account": active if isinstance(active, str) else None}


def logout(directory: str) -> None:
    for name in (CREDS_FILE, ACCOUNTS_FILE):
        try:
            os.remove(os.path.join(directory, name))
        except FileNotFoundError:
            pass


def _mtime(path: str) -> float:
    try:
        return os.stat(path).st_mtime
    except FileNotFoundError:
        return -1.0


def _stop(pid: int) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + CHILD_EXIT_GRACE_SECONDS
        while time.monotonic() < deadline:
            if os.waitpid(pid, os.WNOHANG)[0]:
                return
            time.sleep(0.1)


def login(directory: str) -> int:
    """Relay the CLI's sign-in prompt; return 0 once fresh credentials are on disk."""
    ensure_settings(directory)
    creds = os.path.join(directory, CREDS_FILE)
    before = _mtime(creds)
    scratch = tempfile.mkdtemp(prefix="advr-gemini-login-")
    pid, master = pty.fork()
    if pid == 0:  # child: the CLI, in an empty dir so no project GEMINI.md is read
        os.chdir(scratch)
        os.environ["NO_BROWSER"] = "true"
        try:
            os.execvp("gemini", ["gemini"])
        finally:
            os._exit(127)
    fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", 50, TERMINAL_COLUMNS, 0, 0))
    stdin = sys.stdin.fileno()
    pending = ""
    try:
        while True:
            if _mtime(creds) > before and status(directory)["logged_in"]:
                print("Signed in.", flush=True)
                return 0
            ready, _, _ = select.select([master, stdin], [], [], POLL_SECONDS)
            if stdin in ready:
                line = sys.stdin.readline()
                if not line:  # parent closed our stdin: give up
                    return 1
                os.write(master, line.strip().encode() + b"\r")
            if master in ready:
                try:
                    chunk = os.read(master, 65536)
                except OSError as exc:  # EIO: the CLI exited and closed its terminal
                    if exc.errno != errno.EIO:
                        raise
                    chunk = b""
                if not chunk:
                    print("gemini exited before sign-in completed", flush=True)
                    return 1
                pending += TERMINAL_ESCAPE.sub("", chunk.decode("utf-8", "replace")).replace("\r", "\n")
                *lines, pending = pending.split("\n")
                for text in lines:
                    if text.strip():
                        print(text.strip(), flush=True)
                if "authorization code" in pending.lower():  # prompt has no newline: flush it now
                    print(pending.strip(), flush=True)
                    pending = ""
    finally:
        _stop(pid)
        os.close(master)
        try:
            os.rmdir(scratch)
        except OSError:
            pass


def main(args: list) -> int:
    directory = gemini_dir()
    if args == ["login"]:
        return login(directory)
    if args == ["status"]:
        print(json.dumps(status(directory)))
        return 0
    if args == ["logout"]:
        logout(directory)
        return 0
    print("usage: gemini_auth.py login|status|logout", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
