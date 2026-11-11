"""Log the container's agent CLIs into the operator's own subscriptions, from the web UI.

Flow ("give me the URL, I come back with the code"):

    POST /api/auth/claude/login ─► Popen(claude auth login --claudeai) ─► reader thread parses the URL
    UI shows the URL ─► operator signs in, copies the code ─► POST /api/auth/claude/code ─► CLI stdin
    POST /api/auth/codex/login  ─► Popen(codex login --device-auth) ─► URL + one-time code; CLI polls

Only subscription flows exist here: ``--console`` (Claude API billing) and ``--with-api-key``
(OpenAI API billing) are never run. Codes and CLI output are never logged.
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

LOGIN_TIMEOUT_SECONDS = 15 * 60  # codex device codes expire after 15 min; claude's flow gets the same budget
STATUS_TIMEOUT_SECONDS = 20
STATUS_CACHE_SECONDS = 5
OUTPUT_TAIL_LINES = 6

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
URL_PATTERN = re.compile(r"https://[^\s\"'<>]+")
DEVICE_CODE_PATTERN = re.compile(r"\b[A-Z0-9]{4}-[A-Z0-9]{4,6}\b")
# claude's code is "<authorization code>#<state>": URL-safe base64 pieces joined by '#'
CLAUDE_CODE_PATTERN = re.compile(r"^[A-Za-z0-9_\-#.~]{16,1024}$")

# session states
STARTING = "starting"          # CLI spawned, URL not seen yet
AWAITING_USER = "awaiting_user"  # URL (and device code) shown, operator acts in their browser
VERIFYING = "verifying"        # claude code written to stdin, CLI exchanging it
COMPLETED = "completed"
FAILED = "failed"
EXPIRED = "expired"
TERMINAL_STATES = frozenset({COMPLETED, FAILED, EXPIRED})

# account kinds reported by status
SUBSCRIPTION = "subscription"
API_BILLING = "api_billing"  # logged in, but through pay-per-use API credentials


class AuthError(RuntimeError):
    """Base class for login-flow errors that map to a 4xx answer."""


class UnknownProvider(AuthError):
    pass


class NoLoginInProgress(AuthError):
    pass


class InvalidAuthCode(AuthError):
    pass


@dataclass(frozen=True)
class AuthStatus:
    logged_in: bool
    account: Optional[str] = None  # SUBSCRIPTION | API_BILLING | None
    detail: str = ""


def parse_claude_status(returncode: int, stdout: str) -> AuthStatus:
    """``claude auth status --json``: exit 1 when logged out; ``authMethod`` 'claude.ai' = subscription."""
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        return AuthStatus(False, detail="unreadable status output")
    if not isinstance(data, dict) or not data.get("loggedIn"):
        return AuthStatus(False)
    method = str(data.get("authMethod", ""))
    return AuthStatus(True, SUBSCRIPTION if method == "claude.ai" else API_BILLING, method)


CODEX_CHATGPT = "Logged in using ChatGPT"
CODEX_API_KEY = "Logged in using an API key"


def parse_codex_status(returncode: int, stdout: str) -> AuthStatus:
    """``codex login status`` has no JSON and exits 0 either way, so its text is the only signal
    (codex 0.142.5, pinned in the Dockerfile). Unknown text is reported as logged out, never in."""
    text = ANSI_ESCAPE.sub("", stdout)
    if CODEX_CHATGPT in text:
        return AuthStatus(True, SUBSCRIPTION, "chatgpt")
    if CODEX_API_KEY in text:
        return AuthStatus(True, API_BILLING, "api key")
    return AuthStatus(False)


@dataclass(frozen=True)
class ProviderSpec:
    login_argv: Tuple[str, ...]
    status_argv: Tuple[str, ...]
    logout_argv: Tuple[str, ...]
    parse_status: Callable[[int, str], AuthStatus]
    url_hosts: frozenset
    takes_code: bool  # claude: operator pastes a code back; codex: device code, CLI polls


DEFAULT_PROVIDERS: Dict[str, ProviderSpec] = {
    "claude": ProviderSpec(
        login_argv=("claude", "auth", "login", "--claudeai"),
        status_argv=("claude", "auth", "status", "--json"),
        logout_argv=("claude", "auth", "logout"),
        parse_status=parse_claude_status,
        url_hosts=frozenset({"claude.com", "claude.ai"}),
        takes_code=True,
    ),
    "codex": ProviderSpec(
        login_argv=("codex", "login", "--device-auth"),
        status_argv=("codex", "login", "status"),
        logout_argv=("codex", "logout"),
        parse_status=parse_codex_status,
        url_hosts=frozenset({"auth.openai.com"}),
        takes_code=False,
    ),
}


def allowed_url(line: str, hosts: frozenset) -> Optional[str]:
    """First https URL on the line whose host is (a subdomain of) an allowed host."""
    for match in URL_PATTERN.finditer(line):
        host = (urlsplit(match.group(0)).hostname or "").lower()
        if any(host == h or host.endswith("." + h) for h in hosts):
            return match.group(0)
    return None


@dataclass
class LoginSession:
    provider: str
    proc: subprocess.Popen
    started: float = field(default_factory=time.monotonic)
    state: str = STARTING
    url: Optional[str] = None
    user_code: Optional[str] = None
    error: Optional[str] = None
    tail: List[str] = field(default_factory=list)
    submitted_code: Optional[str] = None  # kept only to redact it from any CLI output

    def snapshot(self) -> Dict:
        return {"state": self.state, "url": self.url, "user_code": self.user_code, "error": self.error}


class AuthManager:
    """One login session per provider; status cached briefly because each check spawns a CLI."""

    def __init__(self, providers: Optional[Dict[str, ProviderSpec]] = None,
                 login_timeout: float = LOGIN_TIMEOUT_SECONDS) -> None:
        self.providers = providers if providers is not None else DEFAULT_PROVIDERS
        self.login_timeout = login_timeout
        self._lock = threading.Lock()
        self._sessions: Dict[str, LoginSession] = {}
        self._status_cache: Dict[str, Tuple[float, AuthStatus]] = {}

    def _spec(self, provider: str) -> ProviderSpec:
        spec = self.providers.get(provider)
        if spec is None:
            raise UnknownProvider(provider)
        return spec

    # -- status -------------------------------------------------------------
    def status(self, provider: str) -> AuthStatus:
        spec = self._spec(provider)
        cached = self._status_cache.get(provider)
        if cached and time.monotonic() - cached[0] < STATUS_CACHE_SECONDS:
            return cached[1]
        try:
            proc = subprocess.run(list(spec.status_argv), capture_output=True, text=True,
                                  timeout=STATUS_TIMEOUT_SECONDS, stdin=subprocess.DEVNULL)
            result = spec.parse_status(proc.returncode, proc.stdout)
        except FileNotFoundError:
            result = AuthStatus(False, detail="CLI not installed")
        except subprocess.TimeoutExpired:
            result = AuthStatus(False, detail="status check timed out")
        self._status_cache[provider] = (time.monotonic(), result)
        return result

    def overview(self) -> Dict[str, Dict]:
        out = {}
        for name, spec in self.providers.items():
            st = self.status(name)
            session = self._sessions.get(name)
            out[name] = {"logged_in": st.logged_in, "account": st.account, "detail": st.detail,
                         "takes_code": spec.takes_code,
                         "login": session.snapshot() if session else None}
        return out

    # -- login --------------------------------------------------------------
    def start_login(self, provider: str) -> Dict:
        """Spawn the CLI login, or return the session already running for this provider."""
        spec = self._spec(provider)
        with self._lock:
            current = self._sessions.get(provider)
            if current is not None and current.state not in TERMINAL_STATES:
                return current.snapshot()
            proc = subprocess.Popen(list(spec.login_argv), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
            session = LoginSession(provider=provider, proc=proc)
            self._sessions[provider] = session
        logger.info("auth: %s login started (pid %d)", provider, proc.pid)
        threading.Thread(target=self._read, args=(session, spec), name=f"auth-{provider}", daemon=True).start()
        timer = threading.Timer(self.login_timeout, self._expire, args=(session,))
        timer.daemon = True
        timer.start()
        return session.snapshot()

    def _read(self, session: LoginSession, spec: ProviderSpec) -> None:
        for raw in session.proc.stdout:
            line = ANSI_ESCAPE.sub("", raw).strip()
            if not line:
                continue
            with self._lock:
                if session.submitted_code:
                    line = line.replace(session.submitted_code, "[code]")
                session.tail = (session.tail + [line])[-OUTPUT_TAIL_LINES:]
                if session.url is None:
                    session.url = allowed_url(line, spec.url_hosts)
                if not spec.takes_code and session.user_code is None:
                    m = DEVICE_CODE_PATTERN.search(line)
                    if m:
                        session.user_code = m.group(0)
                ready = session.url is not None and (spec.takes_code or session.user_code is not None)
                if ready and session.state == STARTING:
                    session.state = AWAITING_USER
        returncode = session.proc.wait()
        for pipe in (session.proc.stdin, session.proc.stdout):
            try:
                pipe.close()
            except OSError:  # stdin may already be broken if the CLI died mid-write
                pass
        with self._lock:
            if session.state not in TERMINAL_STATES:  # expired or cancelled keep their reason
                if returncode == 0:
                    session.state = COMPLETED
                else:
                    session.state = FAILED
                    session.error = f"exit {returncode}: " + " / ".join(session.tail[-2:])
            self._status_cache.pop(session.provider, None)
        logger.info("auth: %s login ended: %s", session.provider, session.state)

    def _expire(self, session: LoginSession) -> None:
        with self._lock:
            if session.state in TERMINAL_STATES:
                return
            session.state = EXPIRED
            session.error = f"no answer within {self.login_timeout / 60:.0f} min, start again"
        session.proc.kill()
        logger.info("auth: %s login expired", session.provider)

    def submit_code(self, provider: str, code: str) -> Dict:
        spec = self._spec(provider)
        code = code.strip()
        if not spec.takes_code:
            raise InvalidAuthCode(f"{provider} does not take a code: enter it on the sign-in page")
        if not CLAUDE_CODE_PATTERN.match(code):
            raise InvalidAuthCode("code looks wrong: copy the whole code shown after sign-in")
        with self._lock:
            session = self._sessions.get(provider)
            if session is None or session.state != AWAITING_USER:
                raise NoLoginInProgress(provider)
            session.state = VERIFYING
            session.submitted_code = code
        try:
            session.proc.stdin.write(code + "\n")
            session.proc.stdin.flush()
        except (OSError, ValueError) as exc:  # ValueError: pipe closed by the reader after exit
            raise NoLoginInProgress(f"{provider} login process is gone") from exc
        logger.info("auth: %s code submitted", provider)  # never the code itself
        return session.snapshot()

    def cancel(self, provider: str) -> None:
        self._spec(provider)
        with self._lock:
            session = self._sessions.pop(provider, None)
            if session is not None and session.state not in TERMINAL_STATES:
                session.state = FAILED
                session.error = "cancelled"
        if session is not None and session.proc.poll() is None:
            session.proc.kill()

    def logout(self, provider: str) -> AuthStatus:
        spec = self._spec(provider)
        self.cancel(provider)
        subprocess.run(list(spec.logout_argv), capture_output=True, text=True,
                       timeout=STATUS_TIMEOUT_SECONDS, stdin=subprocess.DEVNULL)
        self._status_cache.pop(provider, None)
        logger.info("auth: %s logged out", provider)
        return self.status(provider)
