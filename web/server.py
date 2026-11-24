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
    POST /api/debates/<id>/messages {"text"}  the human's remark (bonus round after the verdict)
    POST /api/chats {"agent", "model"?, "effort"?, "project"?, "text"}  new chat + first message -> 202 {"id"}
    POST /api/chats/<id>/messages {"text"}    next message (409 while the agent is answering)
    GET  /api/chats/<id>            JSON snapshot
    PATCH /api/chats/<id> {"title"}  rename ("" = back to the first message) -> 200 snapshot
    DELETE /api/chats/<id>          forget it (409 while the agent is answering) -> 204
    GET  /api/chats/<id>/events     Server-Sent Events, closes when the reply has landed
    GET  /api/projects/<name>/chats saved chats of a project, newest first
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
from typing import Callable, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api_keys import ApiKeyStore, InvalidApiKey  # noqa: E402
from auth import (DEFAULT_PROVIDERS, AuthManager, InvalidAuthCode, NoLoginInProgress, ProviderSpec,  # noqa: E402
                  UnknownProvider)
from backends import (DEFAULT_KEYS_DIR, DEFAULT_TIMEOUT_SECONDS, Backend, build_registry,  # noqa: E402
                      deepseek_key_store)
from chat import Chat, ChatBusy, ChatConfig, ChatDeleted, ChatRunner, InvalidChatConfig, InvalidChatTitle  # noqa: E402
from engine import (Debate, DebateClosed, DebateConfig, DebateRunner, InvalidDebateConfig,  # noqa: E402
                    EventLog, InvalidUserMessage, TooManyUserMessages, clean_user_message)
from projects import (DEFAULT_PROJECT, DEFAULT_PROJECTS_ROOT, InvalidProjectName, TraceHeaderMismatch,  # noqa: E402
                      ProjectStore, UnknownProject)

logger = logging.getLogger("advr.server")

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_INDEX = os.path.join(HERE, "static", "index.html")
STATIC_LOGO = os.path.join(HERE, "static", "logo.svg")
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
_MESSAGES_PATH = re.compile(r"^/api/debates/([0-9a-f]{12})/messages$")
_PROJECT_DEBATES_PATH = re.compile(r"^/api/projects/([^/]+)/debates$")
_CHAT_EVENTS_PATH = re.compile(r"^/api/chats/([0-9a-f]{12})/events$")
_CHAT_PATH = re.compile(r"^/api/chats/([0-9a-f]{12})$")
_CHAT_MESSAGES_PATH = re.compile(r"^/api/chats/([0-9a-f]{12})/messages$")
_PROJECT_CHATS_PATH = re.compile(r"^/api/projects/([^/]+)/chats$")
_AUTH_ACTION_PATH = re.compile(r"^/api/auth/([a-z]+)/(login|code|logout|cancel)$")
_KEY_ACTION_PATH = re.compile(r"^/api/keys/([a-z]+)/(save|clear)$")
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
                 llm_auth: Optional[AuthManager] = None,
                 api_keys: Optional[Dict[str, ApiKeyStore]] = None) -> None:
        self.registry = registry
        self.llm_auth = llm_auth
        self.api_keys = api_keys or {}  # provider -> key typed in the UI (pay-per-use APIs)
        self.auth = auth
        self.trusted_proxies = trusted_proxies
        self.projects = projects
        self.runner = DebateRunner(registry, runs_dir)
        self.debates: Dict[str, Debate] = {}
        self.chat_runner = ChatRunner(registry)
        self.chats: Dict[str, Chat] = {}
        # shared by debates and chat replies: each holds one agent CLI subprocess at a time
        self.slots = threading.BoundedSemaphore(max_active)

    def signed_out_agents(self, agents: Sequence[str]) -> List[str]:
        """Agents whose CLI is not logged in. Checked before spawning: an unauthenticated codex falls back
        to the pay-per-use API and fails with a bare 401 mid-debate."""
        if self.llm_auth is None:
            return []
        out = []
        for agent in dict.fromkeys(agents):
            try:
                if not self.llm_auth.status(agent).logged_in:
                    out.append(agent)
            except UnknownProvider:  # mock, deepseek: no CLI login
                continue
        return out

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

    def post_user_message(self, debate: Debate, text: str) -> bool:
        """Queue the human's remark. On a completed debate it takes a slot and starts a bonus round.
        Returns True when it reopened the debate. Raises UserMessageRejected subclasses, or
        OverflowError when no slot is free for the bonus round."""
        text = clean_user_message(text)  # a bad message must never cost a slot
        # under the debate lock: the judge cannot complete between "is it finished?" and queueing the remark
        with debate._cond:
            needs_slot = debate.state == "completed"
            if needs_slot and not self.slots.acquire(blocking=False):
                raise OverflowError("too many debates running")
            try:
                reopened = debate.add_user_message(text)
            except BaseException:
                if needs_slot:
                    self.slots.release()
                raise
        if reopened:
            if self.projects is not None and self._trace not in debate.observers:  # reloaded from disk
                debate.observers.append(self._trace)
            threading.Thread(target=self._run, args=(debate, self.runner.resume),
                             name=f"debate-{debate.id}-bonus", daemon=True).start()
        return reopened

    def _run(self, debate: Debate, step: Optional[Callable[[Debate], None]] = None) -> None:
        try:
            (step or self.runner.run)(debate)
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

    def start_chat(self, config: ChatConfig, text: str) -> Chat:
        """Create a chat and send its first message (no empty chats on disk)."""
        config.validate(self.registry)
        if self.projects is not None and not self.projects.exists(ProjectStore.validate_name(config.project)):
            raise UnknownProject(config.project)
        text = clean_user_message(text)
        chat = Chat(config=config)
        if self.projects is not None:
            chat.observers.append(self._chat_trace)
        self.post_chat_message(chat, text)
        self.chats[chat.id] = chat
        return chat

    def post_chat_message(self, chat: Chat, text: str) -> None:
        """Record the message and start the reply in a thread. Raises UserMessageRejected subclasses
        (ChatBusy while a reply runs), or OverflowError when no slot is free."""
        text = clean_user_message(text)  # a bad message must never cost a slot
        with chat._cond:  # busy check, slot and queueing as one step (same lesson as debates)
            if chat.state == "busy":
                raise ChatBusy("the agent is still answering; wait for its reply")
            if not self.slots.acquire(blocking=False):
                raise OverflowError("too many agents running")
            try:
                chat.add_user_message(text)
            except BaseException:
                self.slots.release()
                raise
        threading.Thread(target=self._run_chat, args=(chat,), name=f"chat-{chat.id}", daemon=True).start()

    def _run_chat(self, chat: Chat) -> None:
        try:
            self.chat_runner.reply(chat)
        except Exception:  # thread boundary: an engine bug must not leave the chat busy forever
            logger.exception("chat=%s crashed", chat.id)
            if not chat.finished:
                chat.fail_reply("internal_error", "internal error, see server log")
        finally:
            self.slots.release()

    def _chat_trace(self, chat: Chat, event: Dict) -> None:
        """Trace and save on every event: a restart mid-reply must find the busy chat (with the user's
        message) on disk, so reloading it records an "interrupted" failure instead of losing the message."""
        try:
            self.projects.append_event(chat.config.project, chat.id, "chat", "", event)
        except (OSError, UnknownProject, TraceHeaderMismatch):
            logger.exception("chat=%s trace-debat.csv write failed (event %s)", chat.id, event.get("type"))
        self._save_chat(chat)

    def _save_chat(self, chat: Chat) -> None:
        try:
            self.projects.save_chat(chat.config.project, chat.snapshot())
        except OSError:
            logger.exception("chat=%s could not be saved", chat.id)

    def delete_chat(self, chat: Chat) -> None:
        """Forget a chat in memory and on disk. Raises ChatBusy while the agent answers: the reply
        thread would otherwise save it back. A message racing this call gets ChatDeleted."""
        with chat._cond:
            if chat.state == "busy":
                raise ChatBusy("the agent is still answering; delete the chat after its reply")
            chat.mark_deleted()
            self.chats.pop(chat.id, None)
            if self.projects is not None:
                self.projects.delete_chat(chat.config.project, chat.id)

    def get_chat(self, chat_id: str) -> Optional[Chat]:
        """In-memory chat, else a saved one reloaded from its project folder (memory included)."""
        chat = self.chats.get(chat_id)
        if chat is None and self.projects is not None:
            snap = self.projects.load_chat(chat_id)
            if snap is not None:
                loaded = Chat.from_snapshot(snap)  # a busy snapshot comes back idle + reply_failed "interrupted"
                loaded.observers.append(self._chat_trace)
                if snap.get("state") != loaded.state:
                    self._save_chat(loaded)  # persist that failure so event seqs stay stable across reloads
                chat = self.chats.setdefault(chat_id, loaded)
        return chat

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
                self.send_header("Cache-Control", "no-cache")  # phones must pick up a new UI after a rebuild
                self.send_header("Content-Security-Policy",
                                 "default-src 'self'; style-src 'self' 'unsafe-inline'; "
                                 "script-src 'self' 'unsafe-inline'; connect-src 'self'")
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/favicon.svg":
                with open(STATIC_LOGO, "rb") as fh:
                    body = fh.read()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/svg+xml")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "max-age=86400")
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/api/auth" and app.llm_auth is not None:
                self._json(HTTPStatus.OK, app.llm_auth.overview())
                return
            if path == "/api/keys":
                self._json(HTTPStatus.OK, {name: store.status() for name, store in app.api_keys.items()})
                return
            if path == "/api/backends":
                self._json(HTTPStatus.OK, {name: {"available": b.available(), "models": list(b.models),
                                                   "efforts": list(b.efforts)} for name, b in app.registry.items()})
                return
            m = _EVENTS_PATH.match(path)
            if m:
                self._stream(app.get_debate(m.group(1)), query)
                return
            m = _CHAT_EVENTS_PATH.match(path)
            if m:
                self._stream(app.get_chat(m.group(1)), query)
                return
            m = _CHAT_PATH.match(path)
            chat = app.get_chat(m.group(1)) if m else None
            if chat is not None:
                self._json(HTTPStatus.OK, chat.snapshot())
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
                m = _PROJECT_DEBATES_PATH.match(path) or _PROJECT_CHATS_PATH.match(path)
                if m:
                    summaries = (app.projects.chat_summaries if path.endswith("/chats")
                                 else app.projects.debate_summaries)
                    try:
                        self._json(HTTPStatus.OK, summaries(m.group(1)))
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
            m = _KEY_ACTION_PATH.match(self.path)
            if m:
                self._key_action(m.group(1), m.group(2))
                return
            m = _MESSAGES_PATH.match(self.path)
            if m:
                self._user_message(m.group(1))
                return
            if self.path == "/api/chats":
                self._chat_message(None)
                return
            m = _CHAT_MESSAGES_PATH.match(self.path)
            if m:
                self._chat_message(m.group(1))
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
                    options=body.get("options") or {},
                )
                signed_out = app.signed_out_agents((config.proposer, config.critic, config.judge))
                if signed_out:
                    self._error(HTTPStatus.CONFLICT, ", ".join(signed_out)
                                + " non connecté : ouvre « Comptes LLM » et clique Connecter")
                    return
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

        def _user_message(self, debate_id: str) -> None:
            """The human's remark in a debate: read before the next turn, or a bonus round after the verdict."""
            debate = app.get_debate(debate_id)
            if debate is None:
                self._error(HTTPStatus.NOT_FOUND, "unknown debate")
                return
            body = self._read_json_body()
            if body is None:
                return
            text = body.get("text")
            if not isinstance(text, str):
                self._error(HTTPStatus.BAD_REQUEST, "text must be a string")
                return
            try:
                reopened = app.post_user_message(debate, text)
            except InvalidUserMessage as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            except DebateClosed as exc:
                self._error(HTTPStatus.CONFLICT, str(exc))
                return
            except (TooManyUserMessages, OverflowError) as exc:
                self._error(HTTPStatus.TOO_MANY_REQUESTS, str(exc))
                return
            # length only: the remark itself is user content, not log material
            logger.info("debate=%s user message chars=%d reopened=%s", debate.id, len(text.strip()), reopened)
            self._json(HTTPStatus.ACCEPTED, {"reopened": reopened})

        def _key_action(self, provider: str, action: str) -> None:
            """Save or delete an API key. The value never reaches a log line or a response body."""
            store = app.api_keys.get(provider)
            if store is None:
                self._error(HTTPStatus.NOT_FOUND, "unknown provider")
                return
            if not self._secure_transport():
                self._error(HTTPStatus.FORBIDDEN, "API keys may only be sent over HTTPS")
                return
            body = self._read_json_body()
            if body is None:
                return
            key = body.get("key")
            if action == "save" and not isinstance(key, str):
                self._error(HTTPStatus.BAD_REQUEST, "key must be a string")
                return
            try:
                status = store.clear() if action == "clear" else store.set(key)
            except InvalidApiKey as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            except OSError:  # unwritable volume: the traceback names the path, never the key
                logger.exception("could not %s the %s API key", action, provider)
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "could not store the key on the server")
                return
            self._json(HTTPStatus.OK, status)

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

        def do_PATCH(self) -> None:
            """Rename a chat: {"title": "..."}; an empty title goes back to the first message."""
            if not self._guard():
                return
            m = _CHAT_PATH.match(self.path)
            chat = app.get_chat(m.group(1)) if m else None
            if chat is None:
                self._error(HTTPStatus.NOT_FOUND, "unknown chat")
                return
            body = self._read_json_body()
            if body is None:
                return
            title = body.get("title")
            if not isinstance(title, str):
                self._error(HTTPStatus.BAD_REQUEST, "title must be a string")
                return
            try:
                chat.rename(title)
            except InvalidChatTitle as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            logger.info("chat=%s renamed chars=%d", chat.id, len(chat.title))
            self._json(HTTPStatus.OK, chat.snapshot())

        def do_DELETE(self) -> None:
            # CSRF: DELETE is not a "simple" method, so a cross-origin page needs a CORS preflight,
            # which this server never answers.
            if not self._guard():
                return
            m = _CHAT_PATH.match(self.path)
            chat = app.get_chat(m.group(1)) if m else None
            if chat is None:
                self._error(HTTPStatus.NOT_FOUND, "unknown chat")
                return
            try:
                app.delete_chat(chat)
            except ChatBusy as exc:
                self._error(HTTPStatus.CONFLICT, str(exc))
                return
            logger.info("chat=%s deleted", chat.id)
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _chat_message(self, chat_id: Optional[str]) -> None:
            """A chat message: the first one creates the chat (chat_id None), later ones continue it."""
            chat = None
            if chat_id is not None:
                chat = app.get_chat(chat_id)
                if chat is None:
                    self._error(HTTPStatus.NOT_FOUND, "unknown chat")
                    return
            body = self._read_json_body()
            if body is None:
                return
            text = body.get("text")
            if not isinstance(text, str):
                self._error(HTTPStatus.BAD_REQUEST, "text must be a string")
                return
            try:
                if chat is None:
                    config = ChatConfig(agent=str(body["agent"]), project=str(body.get("project", DEFAULT_PROJECT)),
                                        model=body.get("model") or None, effort=body.get("effort") or None)
                agent = chat.config.agent if chat is not None else config.agent
                if app.signed_out_agents((agent,)):
                    self._error(HTTPStatus.CONFLICT, agent + " non connecté : ouvre « Comptes LLM » et clique Connecter")
                    return
                if chat is None:
                    chat = app.start_chat(config, text)
                else:
                    app.post_chat_message(chat, text)
            except KeyError as exc:
                self._error(HTTPStatus.BAD_REQUEST, f"invalid request: missing {exc}")
                return
            except UnknownProject as exc:
                self._error(HTTPStatus.BAD_REQUEST, f"unknown project: {exc}")
                return
            except (InvalidChatConfig, InvalidProjectName, InvalidUserMessage) as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            except ChatBusy as exc:
                self._error(HTTPStatus.CONFLICT, str(exc))
                return
            except ChatDeleted:
                self._error(HTTPStatus.NOT_FOUND, "unknown chat")
                return
            except (TooManyUserMessages, OverflowError) as exc:
                self._error(HTTPStatus.TOO_MANY_REQUESTS, str(exc))
                return
            # length only: the message is user content, not log material
            logger.info("chat=%s message chars=%d agent=%s turns=%d", chat.id, len(text.strip()),
                        chat.config.agent, len(chat.turns))
            self._json(HTTPStatus.ACCEPTED, {"id": chat.id})

        def _stream(self, log: Optional[EventLog], query: str) -> None:
            if log is None:
                self._error(HTTPStatus.NOT_FOUND, "unknown debate or chat")
                return
            debate = log  # debate or chat: same event protocol
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
                logger.info("%s SSE client disconnected", debate.id)

    return Handler


def without_agents(registry: Dict[str, Backend], providers: Dict[str, ProviderSpec],
                   disabled: List[str]) -> Tuple[Dict[str, Backend], Dict[str, ProviderSpec]]:
    """Drop agents switched off by the operator (e.g. a provider that refuses this client) from both
    the debate seats and the Comptes LLM panel. An unknown or required name is a ValueError."""
    unknown = sorted(set(disabled) - (set(registry) - {"mock"}))
    if unknown:
        raise ValueError(f"unknown agent(s) {', '.join(unknown)}; choose from "
                         f"{', '.join(sorted(set(registry) - {'mock'}))}")
    gone = set(disabled)
    return ({n: b for n, b in registry.items() if n not in gone},
            {n: s for n, s in providers.items() if n not in gone})


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
    parser.add_argument("--deepseek-model", default=None, help="DeepSeek model (default deepseek-chat); key from the UI or DEEPSEEK_API_KEY")
    parser.add_argument("--keys-dir", default=DEFAULT_KEYS_DIR,
                        help="where API keys typed in the UI are stored (one 0600 file per provider)")
    parser.add_argument("--gemini-model", default=None, help="Gemini model alias or id (default: the CLI's own); Login with Google")
    parser.add_argument("--disable-agent", action="append", default=[], metavar="NAME",
                        help="hide an agent from the seats and the Comptes LLM panel (repeatable), e.g. gemini")
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
    deepseek_keys = deepseek_key_store(args.keys_dir)
    registry = build_registry(codex_model=args.codex_model, claude_model=args.claude_model,
                              timeout=args.timeout, mock_delay=args.mock_delay,
                              deepseek_model=args.deepseek_model, gemini_model=args.gemini_model,
                              deepseek_keys=deepseek_keys)
    try:
        registry, providers = without_agents(registry, DEFAULT_PROVIDERS, args.disable_agent)
    except ValueError as exc:
        parser.error(f"--disable-agent: {exc}")
    if args.disable_agent:
        logger.info("agents disabled: %s", ", ".join(args.disable_agent))
    projects = ProjectStore(args.projects_dir)
    projects.create(DEFAULT_PROJECT)
    server = make_server(App(registry, None, auth=auth, projects=projects, trusted_proxies=trusted_proxies,
                             llm_auth=AuthManager(providers), api_keys={"deepseek": deepseek_keys}),
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
