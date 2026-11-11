#!/usr/bin/env python3
"""Local web UI for adversarial reasoning (stdlib only, Python 3.9+).

    python3 web/server.py                 # http://127.0.0.1:8765
    python3 web/server.py --port 9000 --codex-model <model>

Routes:
    GET  /                          single-page UI
    GET  /api/backends              which agent CLIs are installed
    POST /api/debates               start a debate -> 202 {"id": ...}
    GET  /api/debates/<id>          JSON snapshot
    GET  /api/debates/<id>/events   Server-Sent Events (replay from ?from=N or Last-Event-ID)
    GET  /api/projects              projects = sub-folders of --projects-dir (default ~/lldebat)
    POST /api/projects              create one: {"name": ...}
    GET  /api/projects/<name>/debates  saved debates of a project, newest first
"""
from __future__ import annotations

import argparse
import base64
import binascii
import hmac
import ipaddress
import json
import logging
import os
import re
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from auth import AuthManager, InvalidAuthCode, NoLoginInProgress, UnknownProvider  # noqa: E402
from backends import DEFAULT_TIMEOUT_SECONDS, Backend, build_registry  # noqa: E402
from engine import Debate, DebateConfig, DebateRunner, InvalidDebateConfig  # noqa: E402
from projects import (DEFAULT_PROJECT, DEFAULT_PROJECTS_ROOT, InvalidProjectName, TraceHeaderMismatch,  # noqa: E402
                      ProjectStore, UnknownProject)

logger = logging.getLogger("advr.server")

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_INDEX = os.path.join(HERE, "static", "index.html")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
MAX_BODY_BYTES = 16 * 1024
MAX_ACTIVE_DEBATES = 2
SSE_HEARTBEAT_SECONDS = 15
AUTH_USER_ENV = "ADVR_USER"
AUTH_PASSWORD_ENV = "ADVR_PASSWORD"
MIN_PASSWORD_CHARS = 12
AUTH_MAX_FAILURES = 10
AUTH_LOCKOUT_SECONDS = 300
FORWARDED_FOR_HEADER = "X-Forwarded-For"

IpNetwork = ipaddress._BaseNetwork  # IPv4Network | IPv6Network


def parse_trusted_proxies(cidrs: Sequence[str]) -> Tuple[IpNetwork, ...]:
    """CIDRs whose requests may name the real client in X-Forwarded-For. Raises ValueError."""
    return tuple(ipaddress.ip_network(c, strict=False) for c in cidrs)


def client_ip(peer: str, forwarded_for: Optional[str], trusted: Tuple[IpNetwork, ...]) -> str:
    """Address the lockout and the logs are keyed on.

    X-Forwarded-For is client-controlled: believed only when the TCP peer is a declared proxy, and
    then only its right-most entry (the one that proxy appended). Anything unparsable falls back to
    the peer, so a malformed header can never mint a fresh lockout bucket per request."""
    try:
        peer_addr = ipaddress.ip_address(peer)
    except ValueError:
        return peer
    if not forwarded_for or not any(peer_addr in net for net in trusted):
        return peer
    last = forwarded_for.split(",")[-1].strip()
    try:
        return str(ipaddress.ip_address(last))
    except ValueError:
        return peer
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_EVENTS_PATH = re.compile(r"^/api/debates/([0-9a-f]{12})/events$")
_DEBATE_PATH = re.compile(r"^/api/debates/([0-9a-f]{12})$")
_PROJECT_DEBATES_PATH = re.compile(r"^/api/projects/([^/]+)/debates$")
_AUTH_ACTION_PATH = re.compile(r"^/api/auth/([a-z]+)/(login|code|logout|cancel)$")
FORWARDED_PROTO_HEADER = "X-Forwarded-Proto"


class BasicAuth:
    """HTTP Basic credentials check with a per-IP lockout after repeated failures."""

    def __init__(self, user: str, password: str, max_failures: int = AUTH_MAX_FAILURES,
                 lockout_seconds: float = AUTH_LOCKOUT_SECONDS) -> None:
        if not user or len(password) < MIN_PASSWORD_CHARS:
            raise ValueError(f"auth needs a user and a password of at least {MIN_PASSWORD_CHARS} chars")
        self._user = user.encode()
        self._password = password.encode()
        self.max_failures = max_failures
        self.lockout_seconds = lockout_seconds
        self._failures: Dict[str, Tuple[int, float]] = {}  # ip -> (count, first_failure_ts)
        self._lock = threading.Lock()

    def locked_out(self, ip: str) -> bool:
        with self._lock:
            count, since = self._failures.get(ip, (0, 0.0))
            if count >= self.max_failures and time.monotonic() - since < self.lockout_seconds:
                return True
            if count and time.monotonic() - since >= self.lockout_seconds:
                self._failures.pop(ip, None)
            return False

    def check(self, ip: str, header: Optional[str]) -> bool:
        ok = False
        if header and header[:6].lower() == "basic ":
            try:
                user, _, password = base64.b64decode(header[6:].strip(), validate=True).partition(b":")
                # evaluate both comparisons: no early exit that would leak which part was wrong
                ok = hmac.compare_digest(user, self._user) & hmac.compare_digest(password, self._password)
            except (binascii.Error, ValueError):
                ok = False
        with self._lock:
            if ok:
                self._failures.pop(ip, None)
            elif header:  # a request without header is the browser's first probe, not a failed guess
                count, since = self._failures.get(ip, (0, time.monotonic()))
                self._failures[ip] = (count + 1, since)
        return ok


def auth_from_env() -> Optional[BasicAuth]:
    user, password = os.environ.get(AUTH_USER_ENV), os.environ.get(AUTH_PASSWORD_ENV)
    if not user and not password:
        return None
    return BasicAuth(user or "", password or "")


class App:
    """Shared state: backend registry, debates in memory, concurrency cap."""

    def __init__(self, registry: Dict[str, Backend], runs_dir: Optional[str],
                 max_active: int = MAX_ACTIVE_DEBATES, auth: Optional[BasicAuth] = None,
                 projects: Optional[ProjectStore] = None,
                 trusted_proxies: Tuple[IpNetwork, ...] = (),
                 llm_auth: Optional[AuthManager] = None) -> None:
        self.registry = registry
        self.llm_auth = llm_auth
        self.auth = auth
        self.trusted_proxies = trusted_proxies
        self.projects = projects
        self.runner = DebateRunner(registry, runs_dir)
        self.debates: Dict[str, Debate] = {}
        self.slots = threading.BoundedSemaphore(max_active)

    def start(self, config: DebateConfig) -> Debate:
        config.validate(self.registry)
        if self.projects is not None and not self.projects.exists(ProjectStore.validate_name(config.project)):
            raise UnknownProject(config.project)
        if not self.slots.acquire(blocking=False):
            raise OverflowError("too many debates running")
        debate = Debate(config=config)
        if self.projects is not None:
            debate.observers.append(self._trace)
        self.debates[debate.id] = debate
        threading.Thread(target=self._run, args=(debate,), name=f"debate-{debate.id}", daemon=True).start()
        return debate

    def _run(self, debate: Debate) -> None:
        try:
            self.runner.run(debate)
        except Exception:  # thread boundary: an engine bug must not leave clients waiting forever
            logger.exception("debate=%s crashed", debate.id)
            if not debate.finished:
                debate.emit("debate_failed", final_state="failed", kind="internal_error", message="internal error, see server log")
        finally:
            self.slots.release()

    def _trace(self, debate: Debate, event: Dict) -> None:
        """Append every event to <project>/trace-debat.csv. A disk error is logged, never fatal to the debate."""
        try:
            self.projects.append_event(debate.config.project, debate.id, "duel", debate.config.question, event)
        except (OSError, UnknownProject, TraceHeaderMismatch):
            logger.exception("debate=%s trace-debat.csv write failed (event %s)", debate.id, event.get("type"))
        if debate.finished:  # terminal event: save the snapshot before any reader can see the end
            try:
                self.projects.save_debate(debate.config.project, debate.snapshot())
            except OSError:
                logger.exception("debate=%s could not be saved", debate.id)

    def get_debate(self, debate_id: str) -> Optional[Debate]:
        """In-memory debate, else a finished one reloaded from its project folder."""
        debate = self.debates.get(debate_id)
        if debate is None and self.projects is not None:
            snap = self.projects.load_debate(debate_id)
            if snap is not None:
                debate = self.debates.setdefault(debate_id, Debate.from_snapshot(snap))
        return debate


def host_without_port(host_header: str) -> str:
    host = host_header.strip().lower()
    if host.startswith("["):  # [::1]:8765
        return host[1:host.find("]")] if "]" in host else host
    return host.rsplit(":", 1)[0] if host.count(":") == 1 else host


def make_handler(app: App, allowed_hosts: frozenset = LOOPBACK_HOSTS):
    class Handler(BaseHTTPRequestHandler):
        server_version = "AdversarialReasoning/0.1"

        def _guard(self) -> bool:
            """Auth first (when configured), else the loopback Host allowlist."""
            if app.auth is None:
                return self._host_allowed()
            ip = self._client_ip()
            if app.auth.locked_out(ip):
                logger.warning("auth lockout active for %s", ip)
                self._error(HTTPStatus.TOO_MANY_REQUESTS, "too many failed logins, retry later")
                return False
            if app.auth.check(ip, self.headers.get("Authorization")):
                return True
            if self.headers.get("Authorization"):
                logger.warning("failed login from %s", ip)
            body = b'{"error": "authentication required"}'
            self.send_response(HTTPStatus.UNAUTHORIZED)
            self.send_header("WWW-Authenticate", 'Basic realm="Adversarial Reasoning", charset="UTF-8"')
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return False

        def _host_allowed(self) -> bool:
            """Reject DNS-rebinding: a hostile page rebound to 127.0.0.1 still sends its own Host.
            With auth enabled the password protects instead (a rebound origin never has it)."""
            if host_without_port(self.headers.get("Host") or "") in allowed_hosts:
                return True
            self._error(HTTPStatus.FORBIDDEN, "host not allowed")
            return False

        def _client_ip(self) -> str:
            return client_ip(self.client_address[0], self.headers.get(FORWARDED_FOR_HEADER),
                             app.trusted_proxies)

        def _secure_transport(self) -> bool:
            """OAuth codes may only travel over TLS: via a trusted proxy that saw https, or loopback."""
            peer = ipaddress.ip_address(self.client_address[0])
            if any(peer in net for net in app.trusted_proxies):
                return (self.headers.get(FORWARDED_PROTO_HEADER) or "").strip().lower() == "https"
            return peer.is_loopback and not self.headers.get(FORWARDED_FOR_HEADER)

        def log_message(self, fmt, *args):  # route http.server logs to logging
            ip = self._client_ip() if getattr(self, "headers", None) is not None else self.address_string()
            logger.info("%s %s", ip, fmt % args)

        # -- helpers ---------------------------------------------------------
        def _json(self, status: int, payload) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status: int, message: str) -> None:
            self._json(status, {"error": message})

        # -- routes ----------------------------------------------------------
        def do_GET(self) -> None:
            if not self._guard():
                return
            path, _, query = self.path.partition("?")
            if path in ("/", "/index.html"):
                with open(STATIC_INDEX, "rb") as fh:
                    body = fh.read()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Content-Security-Policy",
                                 "default-src 'self'; style-src 'self' 'unsafe-inline'; "
                                 "script-src 'self' 'unsafe-inline'; connect-src 'self'")
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/api/auth" and app.llm_auth is not None:
                self._json(HTTPStatus.OK, app.llm_auth.overview())
                return
            if path == "/api/backends":
                self._json(HTTPStatus.OK, {name: {"available": b.available()} for name, b in app.registry.items()})
                return
            m = _EVENTS_PATH.match(path)
            if m:
                self._stream(m.group(1), query)
                return
            m = _DEBATE_PATH.match(path)
            debate = app.get_debate(m.group(1)) if m else None
            if debate is not None:
                self._json(HTTPStatus.OK, debate.snapshot())
                return
            if app.projects is not None:
                if path == "/api/projects":
                    self._json(HTTPStatus.OK, app.projects.list())
                    return
                m = _PROJECT_DEBATES_PATH.match(path)
                if m:
                    try:
                        self._json(HTTPStatus.OK, app.projects.debate_summaries(m.group(1)))
                    except (InvalidProjectName, UnknownProject):
                        self._error(HTTPStatus.NOT_FOUND, "unknown project")
                    return
            self._error(HTTPStatus.NOT_FOUND, "not found")

        def _read_json_body(self) -> Optional[Dict]:
            """Parse a JSON object body, or send the error response and return None."""
            # CSRF: only application/json is accepted. Cross-origin JSON needs a CORS preflight,
            # which this server never answers, so other sites cannot start (paid) debates.
            media_type = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
            if media_type != "application/json":
                self._error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Content-Type must be application/json")
                return None
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_BODY_BYTES:
                self._error(HTTPStatus.BAD_REQUEST, f"body must be 1..{MAX_BODY_BYTES} bytes")
                return None
            try:
                body = json.loads(self.rfile.read(length))
            except json.JSONDecodeError as exc:
                self._error(HTTPStatus.BAD_REQUEST, f"invalid request: {exc}")
                return None
            if not isinstance(body, dict):
                self._error(HTTPStatus.BAD_REQUEST, "invalid request: body must be a JSON object")
                return None
            return body

        def do_POST(self) -> None:
            if not self._guard():
                return
            if self.path == "/api/projects" and app.projects is not None:
                self._create_project()
                return
            m = _AUTH_ACTION_PATH.match(self.path)
            if m and app.llm_auth is not None:
                self._auth_action(m.group(1), m.group(2))
                return
            if self.path != "/api/debates":
                self._error(HTTPStatus.NOT_FOUND, "not found")
                return
            body = self._read_json_body()
            if body is None:
                return
            try:
                config = DebateConfig(
                    question=str(body["question"]),
                    proposer=str(body["proposer"]),
                    critic=str(body["critic"]),
                    judge=str(body["judge"]),
                    max_rounds=int(body.get("max_rounds", 3)),
                    project=str(body.get("project", DEFAULT_PROJECT)),
                )
                debate = app.start(config)
            except UnknownProject as exc:
                self._error(HTTPStatus.BAD_REQUEST, f"unknown project: {exc}")
                return
            except (KeyError, TypeError, ValueError) as exc:
                # InvalidDebateConfig and InvalidProjectName are ValueErrors with user-facing messages
                self._error(HTTPStatus.BAD_REQUEST, str(exc) if isinstance(exc, (InvalidDebateConfig, InvalidProjectName))
                            else f"invalid request: {exc}")
                return
            except OverflowError as exc:
                self._error(HTTPStatus.TOO_MANY_REQUESTS, str(exc))
                return
            logger.info("debate=%s started project=%s proposer=%s critic=%s judge=%s rounds=%d", debate.id,
                        config.project, config.proposer, config.critic, config.judge, config.max_rounds)
            self._json(HTTPStatus.ACCEPTED, {"id": debate.id})

        def _auth_action(self, provider: str, action: str) -> None:
            if not self._secure_transport():
                self._error(HTTPStatus.FORBIDDEN, "LLM sign-in requires HTTPS")
                return
            body = self._read_json_body()
            if body is None:
                return
            try:
                if action == "login":
                    result = app.llm_auth.start_login(provider)
                elif action == "code":
                    result = app.llm_auth.submit_code(provider, str(body.get("code", "")))
                elif action == "cancel":
                    app.llm_auth.cancel(provider)
                    result = {"state": "cancelled"}
                else:
                    st = app.llm_auth.logout(provider)
                    result = {"logged_in": st.logged_in}
            except UnknownProvider:
                self._error(HTTPStatus.NOT_FOUND, "unknown provider")
                return
            except NoLoginInProgress:
                self._error(HTTPStatus.CONFLICT, "no sign-in waiting for a code: start again")
                return
            except InvalidAuthCode as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            except FileNotFoundError:
                self._error(HTTPStatus.SERVICE_UNAVAILABLE, f"{provider} CLI not installed")
                return
            self._json(HTTPStatus.OK, result)

        def _create_project(self) -> None:
            body = self._read_json_body()
            if body is None:
                return
            try:
                created = app.projects.create(str(body.get("name", "")))
            except InvalidProjectName as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            self._json(HTTPStatus.CREATED if created else HTTPStatus.OK, {"name": body["name"], "created": created})

        def _stream(self, debate_id: str, query: str) -> None:
            debate = app.get_debate(debate_id)
            if debate is None:
                self._error(HTTPStatus.NOT_FOUND, "unknown debate")
                return
            start = 0
            m = re.search(r"(?:^|&)from=(\d+)", query)
            if m:
                start = int(m.group(1))
            elif (self.headers.get("Last-Event-ID") or "").isdigit():
                start = int(self.headers["Last-Event-ID"]) + 1
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                while True:
                    events = debate.wait_events(start, timeout=SSE_HEARTBEAT_SECONDS)
                    if not events:
                        if debate.finished:
                            break
                        self.wfile.write(b": heartbeat\n\n")
                    for ev in events:
                        self.wfile.write(f"id: {ev['seq']}\ndata: {json.dumps(ev)}\n\n".encode())
                    self.wfile.flush()
                    start += len(events)
                    if debate.finished and start >= len(debate.events):
                        break
            except (BrokenPipeError, ConnectionResetError):
                logger.info("debate=%s SSE client disconnected", debate_id)

    return Handler


def make_server(app: App, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    allowed = LOOPBACK_HOSTS | {host.lower()}
    server = ThreadingHTTPServer((host, port), make_handler(app, frozenset(allowed)))
    server.daemon_threads = True
    return server


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--claude-model", default=None)
    parser.add_argument("--codex-model", default=None, help="model for codex exec -m (account-dependent)")
    parser.add_argument("--deepseek-model", default=None, help="DeepSeek model (default deepseek-chat); needs DEEPSEEK_API_KEY")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS, help="seconds per agent turn")
    parser.add_argument("--mock-delay", type=float, default=0.8, help="simulated latency of the mock backend")
    parser.add_argument("--trusted-proxy", action="append", default=[], metavar="CIDR",
                        help="reverse proxy allowed to set X-Forwarded-For (repeatable), e.g. the docker gateway")
    parser.add_argument("--projects-dir", default=DEFAULT_PROJECTS_ROOT,
                        help="one sub-folder per project, each with trace-debat.csv + debates/")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        auth = auth_from_env()
    except ValueError as exc:
        parser.error(str(exc))
    if args.host.lower() not in LOOPBACK_HOSTS:
        if auth is None:
            parser.error(f"refusing to bind {args.host} without auth: set {AUTH_USER_ENV} and {AUTH_PASSWORD_ENV}")
        logger.warning("Exposed on %s over plain HTTP: the password crosses the network base64-encoded "
                       "(readable by anyone on the path). Prefer an SSH tunnel or an HTTPS reverse proxy.",
                       args.host)
    try:
        trusted_proxies = parse_trusted_proxies(args.trusted_proxy)
    except ValueError as exc:
        parser.error(f"--trusted-proxy: {exc}")
    registry = build_registry(codex_model=args.codex_model, claude_model=args.claude_model,
                              timeout=args.timeout, mock_delay=args.mock_delay,
                              deepseek_model=args.deepseek_model)
    projects = ProjectStore(args.projects_dir)
    projects.create(DEFAULT_PROJECT)
    server = make_server(App(registry, None, auth=auth, projects=projects, trusted_proxies=trusted_proxies,
                             llm_auth=AuthManager()),
                         args.host, args.port)
    logger.info("projects: %s", projects.root)
    logger.info("auth: %s", "HTTP Basic enabled" if auth else "disabled (loopback only)")
    logger.info("trusted proxies: %s", ", ".join(map(str, trusted_proxies)) or "none")
    logger.info("Serving on http://%s:%d", args.host, server.server_address[1])
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down")


if __name__ == "__main__":
    main()
