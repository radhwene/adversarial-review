"""agy-relay: the only process that holds the Google (Antigravity) OAuth token.

The debate server never sees that token. It POSTs a prompt here; we run ``agy`` once in
headless mode and hand back the final text, nothing else.

Why a separate container: in print mode agy is an *agent*. With its default permission
mode it ran shell commands that a prompt asked for (tested 2026-11-13). In a debate the
prompt carries the other agent's text, so a hostile answer could ask agy to read and
send out the token. Defence in depth, from inside out:

1. ``toolPermission: strict`` plus an explicit deny list (and ``*``) on every call;
2. a fresh empty working directory per call;
3. agy's stderr goes to /dev/null: it prints the account e-mail;
4. this container publishes no port and holds no other secret.

Endpoints:
  GET  /health    -> 200 {"ok": true}           (no auth: reveals nothing)
  POST /complete  -> 200 {"text", "usage"}      Authorization: Bearer <AGY_RELAY_TOKEN>
                     body {"prompt": str, "model": str}
Errors are JSON {"error": {"kind": <one of ERROR_KINDS>, "message": str}}.
"""
from __future__ import annotations

import argparse
import hmac
import json
import logging
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

log = logging.getLogger("agy-relay")

ALLOWED_MODELS = ("gemini-3.8-flash-low", "gemini-3.8-flash-medium", "gemini-3.8-flash-high")

# Every tool agy 1.5.7 advertised in its ``init`` event, plus names it was seen calling
# that the list did not show (``write_file``) and the action names used in
# ``denied_actions``. Strict mode already refuses them; the list is the second lock.
DENIED_TOOLS = (
    "*",
    "ask_custom_permission", "ask_permission", "ask_question", "browser_click_element",
    "browser_drag_pixel_to_pixel", "browser_get_dom", "browser_get_network_request",
    "browser_input", "browser_list_network_requests", "browser_mouse_down", "browser_mouse_up",
    "browser_move_mouse", "browser_press_key", "browser_refresh_page", "browser_resize_window",
    "browser_scroll", "browser_scroll_dom", "browser_select_option", "browser_subagent",
    "call_mcp_tool", "capture_browser_console_logs", "capture_browser_screenshot",
    "click_browser_pixel", "command_status", "define_subagent", "delete_knowledge",
    "execute_browser_javascript", "find_by_name", "generate_image", "grep_search",
    "invoke_subagent", "list_browser_pages", "list_dir", "list_permissions",
    "list_plugin_accounts", "list_resources", "manage_inbox", "manage_subagents", "manage_task",
    "multi_replace_file_content", "notebook_edit", "notebook_execution", "open_browser_url",
    "read_browser_page", "read_resource", "read_url_content", "replace_file_content",
    "run_command", "run_workflow", "schedule", "search_marketplace", "search_web", "sed_file",
    "send_command_input", "send_message", "view_file", "wait", "wait_5_seconds",
    "write_to_file", "write_file", "read_file", "command",
)

SETTINGS_JSON = json.dumps({"toolPermission": "strict", "permissions": {"deny": list(DENIED_TOOLS)}},
                           separators=(",", ":"))

MAX_BODY_BYTES = 512 * 1024
MAX_PROMPT_CHARS = 200_000
MAX_STDOUT_BYTES = 8 * 1024 * 1024
MAX_ERROR_CHARS = 300
MIN_TOKEN_CHARS = 32
DEFAULT_TIMEOUT_S = 240
# agy ignored its own --print-timeout in one test, so we kill it ourselves this long after.
KILL_GRACE_S = 15
QUEUE_WAIT_S = 30

ERROR_KINDS = ("unauthorized", "bad_request", "busy", "timeout", "agent_error", "empty_response",
               "protocol")

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


class AgyFailure(Exception):
    """agy did not produce a usable answer. ``kind`` is one of ERROR_KINDS."""

    def __init__(self, kind: str, message: str, denied: int = 0):
        assert kind in ERROR_KINDS, kind
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.denied = denied  # tool calls the permission lock refused before agy gave up


@dataclass(frozen=True)
class AgyAnswer:
    text: str
    usage: dict
    denied: int = 0  # tool calls agy attempted and the permission lock refused


def redact(text: str) -> str:
    """Drop e-mail addresses and cap the length before anything leaves this process."""
    return _EMAIL.sub("<redacted>", text)[:MAX_ERROR_CHARS]


def build_argv(agy_bin: str, model: str, timeout_s: int) -> list[str]:
    return [agy_bin, "--settings", SETTINGS_JSON, "--model", model, "--disable-slash-commands",
            "--input-format", "stream-json", "--output-format", "stream-json",
            f"--print-timeout={timeout_s}s", "--print="]


def encode_prompt(prompt: str) -> bytes:
    return (json.dumps({"event": "user", "message": {"content": prompt}}) + "\n").encode()


def parse_stream(stdout: bytes) -> AgyAnswer:
    """Find the ``result`` event in agy's NDJSON output."""
    result = None
    for line in stdout.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("event") == "result":
            result = event.get("result")
    if not isinstance(result, dict):
        raise AgyFailure("protocol", "agy ended without a result event")
    usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
    denied = result.get("denied_actions") if isinstance(result.get("denied_actions"), list) else []
    if result.get("status") != "SUCCESS":
        raise AgyFailure("agent_error", redact(str(result.get("error") or "agy reported an error")))
    text = result.get("response")
    if not isinstance(text, str) or not text.strip():
        hint = " (it tried a forbidden tool)" if denied else ""
        raise AgyFailure("empty_response", f"agy returned no text{hint}", denied=len(denied))
    return AgyAnswer(text=text, usage=usage, denied=len(denied))


class AgyRunner:
    """Runs one agy turn at a time: the quota is small and agy keeps local state."""

    def __init__(self, agy_bin: str, timeout_s: int = DEFAULT_TIMEOUT_S,
                 queue_wait_s: float = QUEUE_WAIT_S):
        self.agy_bin = agy_bin
        self.timeout_s = timeout_s
        self.queue_wait_s = queue_wait_s
        self._slot = threading.BoundedSemaphore(1)

    def complete(self, prompt: str, model: str) -> AgyAnswer:
        if not self._slot.acquire(timeout=self.queue_wait_s):
            raise AgyFailure("busy", "another Gemini turn is running, retry later")
        try:
            return self._run(prompt, model)
        finally:
            self._slot.release()

    def _run(self, prompt: str, model: str) -> AgyAnswer:
        with tempfile.TemporaryDirectory(prefix="agy-turn-") as workdir:
            proc = subprocess.Popen(
                build_argv(self.agy_bin, model, self.timeout_s), cwd=workdir,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                start_new_session=True)  # own process group, so a kill takes its children too
            try:
                stdout, _ = proc.communicate(encode_prompt(prompt),
                                             timeout=self.timeout_s + KILL_GRACE_S)
            except subprocess.TimeoutExpired:
                _kill_group(proc)
                raise AgyFailure("timeout", f"agy did not answer within {self.timeout_s}s") from None
        if len(stdout) > MAX_STDOUT_BYTES:
            raise AgyFailure("protocol", "agy output too large")
        if proc.returncode != 0:
            parse_stream(stdout)  # agy prints an ERROR result before exiting non-zero: raise that
            raise AgyFailure("agent_error", f"agy exited with code {proc.returncode}")
        return parse_stream(stdout)


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


STATUS_BY_KIND = {
    "unauthorized": HTTPStatus.UNAUTHORIZED,
    "bad_request": HTTPStatus.BAD_REQUEST,
    "busy": HTTPStatus.SERVICE_UNAVAILABLE,
    "timeout": HTTPStatus.GATEWAY_TIMEOUT,
    "agent_error": HTTPStatus.BAD_GATEWAY,
    "empty_response": HTTPStatus.BAD_GATEWAY,
    "protocol": HTTPStatus.BAD_GATEWAY,
}


def make_handler(runner: AgyRunner, token: str):
    expected = f"Bearer {token}".encode()

    class Handler(BaseHTTPRequestHandler):
        server_version = "agy-relay"
        sys_version = ""

        def log_message(self, fmt, *args):  # replaced by our own one-line access log
            pass

        def _send(self, status: HTTPStatus, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _fail(self, failure: AgyFailure) -> None:
            self._send(STATUS_BY_KIND[failure.kind],
                       {"error": {"kind": failure.kind, "message": failure.message}})

        def do_GET(self):
            if self.path == "/health":
                self._send(HTTPStatus.OK, {"ok": True})
            else:
                self._send(HTTPStatus.NOT_FOUND, {"error": {"kind": "bad_request", "message": "not found"}})

        def do_POST(self):
            started = time.monotonic()
            model, outcome, denied = "-", "ok", 0
            try:
                if self.path != "/complete":
                    raise AgyFailure("bad_request", "not found")
                given = self.headers.get("Authorization", "").encode()
                if not hmac.compare_digest(given, expected):
                    raise AgyFailure("unauthorized", "missing or wrong relay token")
                prompt, model = self._read_request()
                answer = runner.complete(prompt, model)
                denied = answer.denied
                self._send(HTTPStatus.OK, {"text": answer.text, "usage": answer.usage})
            except AgyFailure as failure:
                outcome, denied = failure.kind, failure.denied
                self._fail(failure)
            finally:
                # Never log the prompt or the answer: they are the operator's debate content.
                log.info("POST %s model=%s outcome=%s denied=%d %.1fs", self.path, model, outcome,
                         denied, time.monotonic() - started)

        def _read_request(self) -> tuple[str, str]:
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                raise AgyFailure("bad_request", "Content-Length required") from None
            if length <= 0 or length > MAX_BODY_BYTES:
                raise AgyFailure("bad_request", f"body must be 1..{MAX_BODY_BYTES} bytes")
            try:
                data = json.loads(self.rfile.read(length))
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise AgyFailure("bad_request", "body is not JSON") from None
            if not isinstance(data, dict):
                raise AgyFailure("bad_request", "body must be a JSON object")
            prompt, model = data.get("prompt"), data.get("model")
            if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > MAX_PROMPT_CHARS:
                raise AgyFailure("bad_request", f"prompt must be 1..{MAX_PROMPT_CHARS} chars")
            if model not in ALLOWED_MODELS:
                raise AgyFailure("bad_request", f"model must be one of {', '.join(ALLOWED_MODELS)}")
            return prompt, model

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--agy-bin", default="agy")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S,
                        help="seconds one agy turn may take")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    token = os.environ.get("AGY_RELAY_TOKEN", "")
    if len(token) < MIN_TOKEN_CHARS:
        log.error("AGY_RELAY_TOKEN must be at least %d characters", MIN_TOKEN_CHARS)
        return 2
    server = ThreadingHTTPServer((args.host, args.port),
                                 make_handler(AgyRunner(args.agy_bin, args.timeout), token))
    log.info("listening on %s:%d", args.host, args.port)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
